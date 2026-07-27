# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the hf3fs per-rank sliced loader.

Covers the pure-logic pieces only (no libhf3fs, no GPU): range expansion,
safetensors header parsing, column-segment math, and shard planning.
"""

import json
import struct

import pytest
import torch

from sglang.srt.model_loader.hf3fs_loader import (
    _CHUNK,
    _KIND_COLUMN,
    _KIND_REPLICATE,
    _KIND_ROW_FUSE,
    ShardPlanner,
    TensorLoc,
    _expand_ranges_to_chunks,
    column_segment_ranges,
    parse_safetensors_header,
)

# ---------------------------------------------------------------------------
# _expand_ranges_to_chunks
# ---------------------------------------------------------------------------


class TestExpandRangesToChunks:
    def test_aligned_single_range(self):
        chunks, total = _expand_ranges_to_chunks([(0, 8192)])
        assert total == 8192
        assert chunks == [(0, 8192, 0, 0, 8192)]

    def test_head_padding(self):
        # offset 1000 -> aligned 0, head 1000; logical 4096 bytes
        chunks, total = _expand_ranges_to_chunks([(1000, 5096)])
        assert total == 4096
        assert len(chunks) == 1
        file_off, read_len, src_skip, out_off, copy_len = chunks[0]
        assert file_off == 0
        assert read_len == 5096  # head(1000) + logical(4096)
        assert src_skip == 1000
        assert out_off == 0
        assert copy_len == 4096

    def test_multi_ranges_pack_contiguously(self):
        chunks, total = _expand_ranges_to_chunks([(100, 4196), (8192, 12288)])
        assert total == 4096 + 4096
        # second range's out_off must follow the first range's logical size
        assert chunks[0][3] == 0 and chunks[0][4] == 4096
        assert chunks[1][3] == 4096 and chunks[1][4] == 4096

    def test_large_range_splits(self, monkeypatch):
        import sglang.srt.model_loader.hf3fs_loader as mod

        monkeypatch.setattr(mod, "_CHUNK", 4096)
        try:
            # logical 8192 at aligned offset 0 -> two 4KB chunks
            chunks, total = mod._expand_ranges_to_chunks([(0, 8192)])
            assert total == 8192
            assert [c[1] for c in chunks] == [4096, 4096]
            assert [c[3] for c in chunks] == [0, 4096]
            assert [c[4] for c in chunks] == [4096, 4096]
        finally:
            monkeypatch.setattr(mod, "_CHUNK", _CHUNK)

    def test_empty_and_zero_ranges(self):
        chunks, total = _expand_ranges_to_chunks([(500, 500), (100, 100)])
        assert chunks == [] and total == 0


# ---------------------------------------------------------------------------
# parse_safetensors_header
# ---------------------------------------------------------------------------


def _write_safetensors(path, tensors):
    """tensors: {name: (dtype_str, shape, payload_bytes)}"""
    header = {}
    offset = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for name, (_, _, payload) in tensors.items():
            f.write(payload)


class TestParseHeader:
    def test_offsets_and_dtypes(self, tmp_path):
        p = tmp_path / "model-00001-of-00001.safetensors"
        _write_safetensors(
            p,
            {
                "a.weight": ("BF16", (4, 8), b"\x00" * 64),
                "b.weight": ("F8_E4M3", (2, 2), b"\x01" * 4),
            },
        )
        idx = parse_safetensors_header(str(p))
        assert idx["a.weight"].torch_dtype == torch.bfloat16
        assert idx["a.weight"].shape == (4, 8)
        assert idx["a.weight"].nbytes == 64
        assert idx["b.weight"].start == idx["a.weight"].end
        assert idx["b.weight"].torch_dtype == torch.float8_e4m3fn

    def test_metadata_skipped(self, tmp_path):
        p = tmp_path / "m.safetensors"
        header = {
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        hjson = json.dumps(header).encode()
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(b"\x00" * 4)
        idx = parse_safetensors_header(str(p))
        assert list(idx) == ["w"]


# ---------------------------------------------------------------------------
# column_segment_ranges
# ---------------------------------------------------------------------------


def _loc(name, shape, dtype=torch.bfloat16, file="/fake/shard.safetensors", start=8192):
    nbytes = 1
    for d in shape:
        nbytes *= d
    nbytes *= torch.tensor([], dtype=dtype).element_size()
    return TensorLoc(
        file=file, start=start, end=start + nbytes, shape=shape, torch_dtype=dtype
    )


class TestColumnSegment:
    def test_basic(self):
        loc = _loc("w", (16, 4))  # 16 rows, 8B/row
        # world=4 semantics via rows_local=4; rank 1 -> rows 4..8
        ranges = column_segment_ranges(loc, tp_rank=1, rows_local=4)
        assert ranges == [(loc.start + 4 * 8, loc.start + 8 * 8)]

    def test_vocab_padding_partial(self):
        loc = _loc("w", (10, 4))  # ckpt has 10 rows; world implies 4 rows/rank
        # rank 2 wants rows 8..12 but only 8..10 exist
        ranges = column_segment_ranges(loc, tp_rank=2, rows_local=4)
        assert ranges == [(loc.start + 8 * 8, loc.start + 10 * 8)]

    def test_vocab_padding_full_rank(self):
        loc = _loc("w", (10, 4))
        # rank 3 wants rows 12..16 -> fully padding
        assert column_segment_ranges(loc, tp_rank=3, rows_local=4) == []


# ---------------------------------------------------------------------------
# ShardPlanner
# ---------------------------------------------------------------------------

N = 8  # tp_size
H = 2048
Q_ROWS, KV_ROWS, I = 4096, 512, 768
E = 4  # experts


def _qwen3_like_ckpt() -> dict:
    """Minimal Qwen3-MoE-like checkpoint index (unfused, HF layout)."""
    idx = {}
    p = "model.layers.0"
    idx[f"{p}.self_attn.q_proj.weight"] = _loc("q", (Q_ROWS, H))
    idx[f"{p}.self_attn.k_proj.weight"] = _loc("k", (KV_ROWS, H))
    idx[f"{p}.self_attn.v_proj.weight"] = _loc("v", (KV_ROWS, H))
    idx[f"{p}.self_attn.o_proj.weight"] = _loc("o", (H, Q_ROWS))
    idx[f"{p}.mlp.gate.weight"] = _loc("router", (E, H))
    for e in range(E):
        idx[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _loc(f"g{e}", (I, H))
        idx[f"{p}.mlp.experts.{e}.up_proj.weight"] = _loc(f"u{e}", (I, H))
        idx[f"{p}.mlp.experts.{e}.down_proj.weight"] = _loc(f"d{e}", (H, I))
    idx[f"{p}.input_layernorm.weight"] = _loc("ln", (H,))
    idx["model.embed_tokens.weight"] = _loc("emb", (151936, H))
    idx["model.norm.weight"] = _loc("fn", (H,))
    return idx


def _planner(idx, rank=0):
    return ShardPlanner(idx, tp_rank=rank, tp_size=N, slice_threshold=1 << 20)


class TestShardPlanner:
    def test_qkv_unfused_assembly(self):
        idx = _qwen3_like_ckpt()
        pl = _planner(idx, rank=1)
        plan = pl.plan_param(
            "model.layers.0.self_attn.qkv_proj.weight",
            ((Q_ROWS + 2 * KV_ROWS) // N, H),
        )
        kinds = [pt.kind for pt in plan.parts]
        assert kinds == [_KIND_COLUMN] * 3
        q_local, kv_local = Q_ROWS // N, KV_ROWS // N
        assert [pt.dst_dim0_offset for pt in plan.parts] == [
            0,
            q_local,
            q_local + kv_local,
        ]
        assert [pt.rows_local for pt in plan.parts] == [q_local, kv_local, kv_local]

    def test_qkv_fused_ckpt_rejected(self):
        idx = _qwen3_like_ckpt()
        p = "model.layers.0.self_attn"
        for comp in ("q_proj", "k_proj", "v_proj"):
            del idx[f"{p}.{comp}.weight"]
        idx[f"{p}.qkv_proj.weight"] = _loc("qkv", (Q_ROWS + 2 * KV_ROWS, H))
        with pytest.raises(NotImplementedError):
            _planner(idx).plan_param(
                "model.layers.0.self_attn.qkv_proj.weight",
                ((Q_ROWS + 2 * KV_ROWS) // N, H),
            )

    def test_w13_gate_front_up_back(self):
        idx = _qwen3_like_ckpt()
        plan = _planner(idx).plan_param(
            "model.layers.0.mlp.experts.w13_weight", (E, 2 * (I // N), H)
        )
        assert len(plan.parts) == 2 * E
        i_local = I // N
        for e in range(E):
            gate = next(
                pt
                for pt in plan.parts
                if pt.ckpt_name.endswith(f"{e}.gate_proj.weight")
            )
            up = next(
                pt for pt in plan.parts if pt.ckpt_name.endswith(f"{e}.up_proj.weight")
            )
            assert gate.expert_id == e and gate.dst_dim0_offset == 0
            assert up.expert_id == e and up.dst_dim0_offset == i_local
            assert gate.rows_local == i_local

    def test_w2_row_fuse(self):
        idx = _qwen3_like_ckpt()
        plan = _planner(idx).plan_param(
            "model.layers.0.mlp.experts.w2_weight", (E, H, I // N)
        )
        assert len(plan.parts) == E
        assert all(pt.kind == _KIND_ROW_FUSE for pt in plan.parts)
        assert {pt.expert_id for pt in plan.parts} == set(range(E))

    def test_o_proj_row_fuse_by_shape_inference(self):
        idx = _qwen3_like_ckpt()
        plan = _planner(idx).plan_param(
            "model.layers.0.self_attn.o_proj.weight", (H, Q_ROWS // N)
        )
        assert plan.parts[0].kind == _KIND_ROW_FUSE
        assert plan.parts[0].ckpt_name == "model.layers.0.self_attn.o_proj.weight"

    def test_small_tensor_replicate(self):
        idx = _qwen3_like_ckpt()
        plan = _planner(idx).plan_param("model.layers.0.input_layernorm.weight", (H,))
        assert plan.parts[0].kind == _KIND_REPLICATE

    def test_router_gate_replicate(self):
        idx = _qwen3_like_ckpt()
        plan = _planner(idx).plan_param("model.layers.0.mlp.gate.weight", (E, H))
        assert plan.parts[0].kind == _KIND_REPLICATE

    def test_embed_vocab_padding_column(self):
        idx = _qwen3_like_ckpt()
        # param rows padded: ckpt 151936 rows / 8 -> param may hold 18992 rows
        plan = _planner(idx).plan_param("model.embed_tokens.weight", (151936 // N, H))
        assert plan.parts[0].kind == _KIND_COLUMN
        assert plan.parts[0].rows_local == 151936 // N

    def test_lm_head_tied_fallback(self):
        idx = _qwen3_like_ckpt()
        # tied checkpoint: no lm_head.weight in ckpt
        plan = _planner(idx).plan_param("lm_head.weight", (151936 // N, H))
        assert plan is not None
        assert plan.parts[0].kind == _KIND_COLUMN
        assert plan.parts[0].ckpt_name == "model.embed_tokens.weight"

    def test_missing_returns_none(self):
        idx = _qwen3_like_ckpt()
        plan = _planner(idx).plan_param(
            "model.layers.0.self_attn.rotary_emb.cos_sin_cache", (4096, 64)
        )
        assert plan is None

    def test_plan_all_splits_missing(self):
        idx = _qwen3_like_ckpt()
        shapes = {
            "model.layers.0.self_attn.qkv_proj.weight": (
                (Q_ROWS + 2 * KV_ROWS) // N,
                H,
            ),
            "model.layers.0.self_attn.rotary_emb.cos_sin_cache": (4096, 64),
        }
        plans, missing = _planner(idx).plan_all(shapes)
        assert list(plans) == ["model.layers.0.self_attn.qkv_proj.weight"]
        assert missing == ["model.layers.0.self_attn.rotary_emb.cos_sin_cache"]

    def test_shape_mismatch_raises(self):
        idx = _qwen3_like_ckpt()
        with pytest.raises(AssertionError):
            _planner(idx).plan_param(
                "model.layers.0.self_attn.qkv_proj.weight", (12345, H)
            )
