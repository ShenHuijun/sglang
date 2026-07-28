# SPDX-License-Identifier: Apache-2.0
"""Hf3fsModelLoader — per-rank sliced weight loading from a 3FS-mounted model dir.

Instead of reading every checkpoint tensor in full on each TP rank (the default
safetensors path), this loader reads only the bytes this rank owns:

- Column-style tensors (sharded on dim0: q/k/v_proj, gate/up_proj, MoE expert
  gate/up, embed_tokens/lm_head): each rank USRBIO range-reads its own 1/N
  contiguous dim0 segment (direct IO, bypasses FUSE/page cache).
- Row-style tensors (sharded on dim1: o_proj, dense/shared down_proj, MoE
  expert down_proj -> w2): a dim1 slice is a strided read with per-row
  granularity; under the USRBIO 4KB alignment constraint the read amplification
  is ~20x (e.g. 192B logical per row vs 4KB physical), so v1 falls back to a
  full FUSE read. On a single node all TP ranks share the page cache, so the
  network cost of Row tensors is ~1x model size, same as the default loader.
- Small tensors (< slice_threshold bytes, e.g. norms, router gate, fp8 scales):
  full FUSE read (replicate).

Fused-parameter assembly (no hf_config needed — shard sizes are inferred from
the ratio between param shape and checkpoint tensor shape):
  qkv_proj.weight      <- q_proj/k_proj/v_proj
  gate_up_proj.weight  <- gate_proj/up_proj
  experts.w13_weight   <- experts.{e}.gate_proj (front half) + up_proj (back)
  experts.w2_weight    <- experts.{e}.down_proj (dim1 slice)

v1 scope: Qwen3-MoE-style architectures, TP-only (no EP), checkpoints with
unfused q/k/v and gate/up tensors (the HF layout). Already-fused qkv_proj /
gate_up_proj checkpoint tensors are rejected explicitly. DSv3 MLA naming,
fused shared experts into w13, and GGUF/AWQ on-disk quant layouts are not
supported yet.
"""

from __future__ import annotations

import ctypes
import glob
import json
import logging
import os
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    ShardedStateLoader,
    _get_quantization_config,
    _initialize_model,
)

try:
    from sglang.srt.utils import set_default_torch_dtype
except ImportError:  # older releases (e.g. 0.5.11)
    from sglang.srt.model_loader.utils import set_default_torch_dtype

try:  # sglang main (>=0.5.x post)
    from sglang.srt.runtime_context import get_parallel

    def _tp_rank() -> int:
        return get_parallel().tp_rank

    def _tp_size() -> int:
        return get_parallel().tp_size

except ImportError:  # older releases (e.g. 0.5.11)
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    def _tp_rank() -> int:
        return get_tensor_model_parallel_rank()

    def _tp_size() -> int:
        return get_tensor_model_parallel_world_size()


try:
    from sglang.srt.model_loader.loader import _post_load_weights
except ImportError:  # older releases without the shared helper

    def _post_load_weights(model: nn.Module) -> None:
        # Loaders that bypass model.load_weights() must trigger the post-load
        # fixup explicitly (see loader.py in newer releases).
        if hasattr(model, "post_load_weights"):
            model.post_load_weights()


logger = logging.getLogger(__name__)

_ALIGN = 4096
_IOV_SIZE = 2 << 30  # 2GB registered buffer, reused in batches
_CHUNK = 32 << 20  # max bytes per single prep_io

_ST_DTYPE_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _ST_DTYPE_TO_TORCH["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _ST_DTYPE_TO_TORCH["F8_E5M2"] = torch.float8_e5m2


# ---------------------------------------------------------------------------
# USRBIO reader (ctypes binding of libhf3fs_api_shared.so)
# ---------------------------------------------------------------------------


class _Hf3fsIov(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.POINTER(ctypes.c_uint8)),
        ("iovh", ctypes.c_void_p),
        ("id", ctypes.c_char * 16),
        ("mount_point", ctypes.c_char * 256),
        ("size", ctypes.c_size_t),
        ("block_size", ctypes.c_size_t),
        ("numa", ctypes.c_int),
    ]


class _Hf3fsIor(ctypes.Structure):
    _fields_ = [
        ("iov", _Hf3fsIov),
        ("iorh", ctypes.c_void_p),
        ("mount_point", ctypes.c_char * 256),
        ("for_read", ctypes.c_bool),
        ("io_depth", ctypes.c_int),
        ("priority", ctypes.c_int),
        ("timeout", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    ]


class _Hf3fsCqe(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("result", ctypes.c_int64),
        ("userdata", ctypes.c_void_p),
    ]


_LIB = None


def _load_libhf3fs():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL("/usr/lib/libhf3fs_api_shared.so")
    lib.hf3fs_extract_mount_point.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    lib.hf3fs_extract_mount_point.restype = ctypes.c_int
    lib.hf3fs_iovcreate.argtypes = [
        ctypes.POINTER(_Hf3fsIov),
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.hf3fs_iovcreate.restype = ctypes.c_int
    lib.hf3fs_iorcreate4.argtypes = [
        ctypes.POINTER(_Hf3fsIor),
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.hf3fs_iorcreate4.restype = ctypes.c_int
    lib.hf3fs_reg_fd.argtypes = [ctypes.c_int, ctypes.c_uint64]
    lib.hf3fs_reg_fd.restype = ctypes.c_int
    lib.hf3fs_prep_io.argtypes = [
        ctypes.POINTER(_Hf3fsIor),
        ctypes.POINTER(_Hf3fsIov),
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    lib.hf3fs_prep_io.restype = ctypes.c_int
    lib.hf3fs_submit_ios.argtypes = [ctypes.POINTER(_Hf3fsIor)]
    lib.hf3fs_submit_ios.restype = ctypes.c_int
    lib.hf3fs_wait_for_ios.argtypes = [
        ctypes.POINTER(_Hf3fsIor),
        ctypes.POINTER(_Hf3fsCqe),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.hf3fs_wait_for_ios.restype = ctypes.c_int
    lib.hf3fs_dereg_fd.argtypes = [ctypes.c_int]
    lib.hf3fs_iovdestroy.argtypes = [ctypes.POINTER(_Hf3fsIov)]
    lib.hf3fs_iordestroy.argtypes = [ctypes.POINTER(_Hf3fsIor)]
    _LIB = lib
    return lib


def _expand_ranges_to_chunks(
    ranges: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int, int, int, int]], int]:
    """Expand logical ranges into 4KB-aligned read chunks.

    Returns (chunks, total_logical) where each chunk is
    (file_off, read_len, src_skip, out_off, copy_len):
    - file_off/read_len: aligned read to issue (read_len <= _CHUNK)
    - src_skip: bytes to skip at the front of the read buffer (alignment head)
    - out_off/copy_len: where the logical payload lands in the output buffer
    """
    chunks: List[Tuple[int, int, int, int, int]] = []
    out_off = 0
    for s, e in ranges:
        logical = e - s
        if logical <= 0:
            continue
        a_s = (s // _ALIGN) * _ALIGN
        head = s - a_s
        pos = 0
        while pos < head + logical:
            ln = min(_CHUNK, head + logical - pos)
            src_skip = max(0, head - pos)
            copy_off = max(0, pos - head)
            copy_len = min(ln - src_skip, logical - copy_off)
            chunks.append((a_s + pos, ln, src_skip, out_off + copy_off, copy_len))
            pos += ln
        out_off += logical
    return chunks, out_off


class UsrbioReader:
    """Batched range reads over 3FS USRBIO into one registered iov buffer."""

    def __init__(
        self,
        mount_point: bytes,
        iov_size: int = _IOV_SIZE,
        entries: int = 1024,
        numa: int = -1,
    ):
        self.lib = _load_libhf3fs()
        self.iov_size = iov_size
        self.entries = entries
        self.iov = _Hf3fsIov()
        rc = self.lib.hf3fs_iovcreate(
            ctypes.byref(self.iov), mount_point, iov_size, 0, numa
        )
        assert rc == 0, f"hf3fs_iovcreate failed: {rc}"
        self.ior = _Hf3fsIor()
        rc = self.lib.hf3fs_iorcreate4(
            ctypes.byref(self.ior), mount_point, entries, True, 0, 0, numa, 0
        )
        assert rc == 0, f"hf3fs_iorcreate4 failed: {rc}"
        self.base = ctypes.cast(self.iov.base, ctypes.c_void_p).value
        self.bytes_read = 0  # physical bytes (incl. 4KB-alignment padding)

    def close(self):
        self.lib.hf3fs_iordestroy(ctypes.byref(self.ior))
        self.lib.hf3fs_iovdestroy(ctypes.byref(self.iov))

    def read_ranges(self, fd: int, ranges: List[Tuple[int, int]], out: np.ndarray):
        """Read [(start, end)] (absolute file offsets, end-exclusive) into
        `out`, packed contiguously by logical bytes in range order. Each start
        is aligned down to 4KB; the alignment head is read but discarded."""
        chunks, total_logical = _expand_ranges_to_chunks(ranges)
        assert total_logical <= out.nbytes, (
            f"out buffer {out.nbytes}B < logical read {total_logical}B"
        )

        batch = []  # (buf_slot, chunk)
        used = 0

        def flush():
            nonlocal batch, used
            if not batch:
                return
            lib = self.lib
            for buf_slot, (file_off, read_len, _, _, _) in batch:
                rc = lib.hf3fs_prep_io(
                    ctypes.byref(self.ior),
                    ctypes.byref(self.iov),
                    True,
                    ctypes.c_void_p(self.base + buf_slot),
                    fd,
                    file_off,
                    read_len,
                    None,
                )
                assert rc >= 0, f"hf3fs_prep_io: {rc} ({os.strerror(-rc)})"
            rc = lib.hf3fs_submit_ios(ctypes.byref(self.ior))
            assert rc == 0, f"hf3fs_submit_ios: {rc}"
            n = len(batch)
            cqes = (_Hf3fsCqe * n)()
            done = 0
            while done < n:
                rc = lib.hf3fs_wait_for_ios(
                    ctypes.byref(self.ior), cqes, n, n - done, None
                )
                assert rc > 0, f"hf3fs_wait_for_ios: {rc}"
                for j in range(rc):
                    assert cqes[j].result >= 0, f"io error: {cqes[j].result}"
                    self.bytes_read += cqes[j].result
                done += rc
            u8 = out.view(np.uint8)
            for buf_slot, (_, read_len, src_skip, o_off, copy_len) in batch:
                src = np.frombuffer(
                    (ctypes.c_uint8 * read_len).from_address(self.base + buf_slot),
                    dtype=np.uint8,
                )
                u8[o_off : o_off + copy_len] = src[src_skip : src_skip + copy_len]
            batch = []
            used = 0

        for chunk in chunks:
            read_len = chunk[1]
            if used + read_len > self.iov_size or len(batch) >= self.entries:
                flush()
            batch.append((used, chunk))
            used += read_len
        flush()

    def read_tensor(
        self, fd: int, ranges: List[Tuple[int, int]], torch_dtype: torch.dtype
    ) -> torch.Tensor:
        total = sum(e - s for s, e in ranges)
        out = np.empty(max(total, 1), dtype=np.uint8)
        self.read_ranges(fd, ranges, out)
        return torch.from_numpy(out[:total]).view(torch_dtype)


def extract_mount_point(path: str) -> bytes:
    mp_buf = ctypes.create_string_buffer(256)
    rc = _load_libhf3fs().hf3fs_extract_mount_point(mp_buf, 256, path.encode())
    assert rc > 0, f"hf3fs_extract_mount_point failed for {path}"
    return mp_buf.value


# ---------------------------------------------------------------------------
# Checkpoint index (safetensors headers only — no payload reads)
# ---------------------------------------------------------------------------


@dataclass
class TensorLoc:
    file: str
    start: int  # absolute offset of payload in file
    end: int
    shape: Tuple[int, ...]
    torch_dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def parse_safetensors_header(path: str) -> Dict[str, TensorLoc]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        dtype = _ST_DTYPE_TO_TORCH.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype {meta['dtype']} ({name})")
        out[name] = TensorLoc(
            file=path,
            start=base + s,
            end=base + e,
            shape=tuple(meta["shape"]),
            torch_dtype=dtype,
        )
    return out


def build_ckpt_index(model_dir: str) -> Dict[str, TensorLoc]:
    index: Dict[str, TensorLoc] = {}
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise RuntimeError(f"no safetensors found under {model_dir}")
    for sh in shards:
        for name, loc in parse_safetensors_header(sh).items():
            index[name] = loc
    return index


# ---------------------------------------------------------------------------
# Shard planning
# ---------------------------------------------------------------------------

_KIND_COLUMN = "column"  # dim0 contiguous 1/N segment via USRBIO
_KIND_ROW_FUSE = "row_fuse"  # full FUSE read, then dim1 narrow (v1 fallback)
_KIND_REPLICATE = "replicate"  # full FUSE read


@dataclass
class LoadPart:
    """One checkpoint tensor contributing (a slice of) one param."""

    ckpt_name: str
    kind: str
    rows_local: int = 0  # for COLUMN: rows this rank owns from this tensor
    dst_dim0_offset: int = 0  # where this part lands in the param (or param[e])
    expert_id: Optional[int] = None


@dataclass
class ParamPlan:
    param_name: str
    parts: List[LoadPart] = field(default_factory=list)


_QKV_RE = re.compile(r"^(?P<prefix>.*\.)qkv_proj\.weight$")
_GATE_UP_RE = re.compile(r"^(?P<prefix>.*\.)gate_up_proj\.weight$")
_W13_RE = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.)w13_weight$")
_W2_RE = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.)w2_weight$")
_EMBED_RE = re.compile(r"(embed_tokens|lm_head)\.weight$")


def _div(a: int, b: int) -> int:
    assert a % b == 0, f"{a} not divisible by {b}"
    return a // b


def column_segment_ranges(
    loc: TensorLoc, tp_rank: int, rows_local: int
) -> List[Tuple[int, int]]:
    """Byte ranges of this rank's dim0 segment of a row-major tensor.
    Clips against the actual checkpoint rows (vocab-padding safe)."""
    rows_full = loc.shape[0]
    row_bytes = loc.nbytes // rows_full
    start_row = tp_rank * rows_local
    if start_row >= rows_full:
        return []  # this rank is fully padding
    end_row = min(start_row + rows_local, rows_full)
    return [(loc.start + start_row * row_bytes, loc.start + end_row * row_bytes)]


class ShardPlanner:
    """Map each local param to the checkpoint bytes this rank must read.

    All shard sizes are inferred from shape ratios (param vs checkpoint), so no
    hf_config / model-specific constants are needed. Params with no matching
    checkpoint tensor (model-internal buffers like cos_sin_cache) are planned
    as None by plan_all and skipped by the caller.
    """

    def __init__(
        self,
        ckpt_index: Dict[str, TensorLoc],
        tp_rank: int,
        tp_size: int,
        slice_threshold: int = 1 << 20,
    ):
        self.ckpt = ckpt_index
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.slice_threshold = slice_threshold
        self._expert_ids = sorted(
            {
                int(m.group(1))
                for name in ckpt_index
                for m in [re.search(r"\.experts\.(\d+)\.", name)]
                if m
            }
        )

    def _infer_kind_by_shape(
        self, param_shape: Tuple[int, ...], loc: TensorLoc
    ) -> Optional[str]:
        if tuple(param_shape) == loc.shape:
            return _KIND_REPLICATE
        if len(param_shape) == 2 and len(loc.shape) == 2:
            if (
                param_shape[1] == loc.shape[1]
                and loc.shape[0] == param_shape[0] * self.tp_size
            ):
                return _KIND_COLUMN
            if (
                param_shape[0] == loc.shape[0]
                and loc.shape[1] == param_shape[1] * self.tp_size
            ):
                return _KIND_ROW_FUSE
        if (
            len(param_shape) == 1
            and len(loc.shape) == 1
            and loc.shape[0] == param_shape[0] * self.tp_size
        ):
            return _KIND_COLUMN
        return None

    def plan_param(
        self, param_name: str, param_shape: Tuple[int, ...]
    ) -> Optional[ParamPlan]:
        plan = ParamPlan(param_name)
        world = self.tp_size

        m = _QKV_RE.match(param_name)
        if m:
            prefix = m.group("prefix")
            if f"{prefix}qkv_proj.weight" in self.ckpt:
                raise NotImplementedError(
                    f"{param_name}: already-fused qkv in checkpoint is not "
                    f"supported in v1 (shard boundaries inside the fused tensor "
                    f"require model-specific head sizes)"
                )
            dst = 0
            for comp in ("q_proj", "k_proj", "v_proj"):
                ckpt_name = f"{prefix}{comp}.weight"
                loc = self.ckpt[ckpt_name]
                rows_local = _div(loc.shape[0], world)
                plan.parts.append(
                    LoadPart(
                        ckpt_name=ckpt_name,
                        kind=_KIND_COLUMN,
                        rows_local=rows_local,
                        dst_dim0_offset=dst,
                    )
                )
                dst += rows_local
            assert dst == param_shape[0], (
                f"{param_name}: assembled rows {dst} != param rows {param_shape[0]}"
            )
            return plan

        m = _GATE_UP_RE.match(param_name)
        if m and ".experts." not in param_name:
            prefix = m.group("prefix")
            if f"{prefix}gate_up_proj.weight" in self.ckpt:
                raise NotImplementedError(
                    f"{param_name}: already-fused gate_up in checkpoint is not "
                    f"supported in v1"
                )
            dst = 0
            for comp in ("gate_proj", "up_proj"):
                ckpt_name = f"{prefix}{comp}.weight"
                loc = self.ckpt[ckpt_name]
                rows_local = _div(loc.shape[0], world)
                plan.parts.append(
                    LoadPart(
                        ckpt_name=ckpt_name,
                        kind=_KIND_COLUMN,
                        rows_local=rows_local,
                        dst_dim0_offset=dst,
                    )
                )
                dst += rows_local
            assert dst == param_shape[0], (
                f"{param_name}: assembled rows {dst} != param rows {param_shape[0]}"
            )
            return plan

        m = _W13_RE.match(param_name)
        if m:
            prefix = m.group("prefix")
            # param: [E, 2*I_local, H]; ckpt per expert: gate/up [I, H]
            for e in self._expert_ids:
                for comp, half in (("gate_proj", 0), ("up_proj", 1)):
                    ckpt_name = f"{prefix}{e}.{comp}.weight"
                    loc = self.ckpt[ckpt_name]
                    rows_local = _div(loc.shape[0], world)
                    assert param_shape[1] == 2 * rows_local, (
                        f"{param_name}: param dim1 {param_shape[1]} != "
                        f"2*I/N {2 * rows_local}"
                    )
                    plan.parts.append(
                        LoadPart(
                            ckpt_name=ckpt_name,
                            kind=_KIND_COLUMN,
                            rows_local=rows_local,
                            dst_dim0_offset=half * rows_local,
                            expert_id=e,
                        )
                    )
            return plan

        m = _W2_RE.match(param_name)
        if m:
            prefix = m.group("prefix")
            for e in self._expert_ids:
                ckpt_name = f"{prefix}{e}.down_proj.weight"
                loc = self.ckpt[ckpt_name]
                cols_local = _div(loc.shape[1], world)
                assert param_shape[2] == cols_local, (
                    f"{param_name}: param dim2 {param_shape[2]} != I/N {cols_local}"
                )
                plan.parts.append(
                    LoadPart(ckpt_name=ckpt_name, kind=_KIND_ROW_FUSE, expert_id=e)
                )
            return plan

        if param_name in self.ckpt:
            loc = self.ckpt[param_name]
            if loc.nbytes < self.slice_threshold:
                plan.parts.append(LoadPart(ckpt_name=param_name, kind=_KIND_REPLICATE))
                return plan
            kind = self._infer_kind_by_shape(param_shape, loc)
            if kind == _KIND_COLUMN and _EMBED_RE.search(param_name):
                # vocab padding: param rows * N may exceed ckpt rows; use param
                # rows as the local shard size and clip at read time
                plan.parts.append(
                    LoadPart(
                        ckpt_name=param_name,
                        kind=_KIND_COLUMN,
                        rows_local=param_shape[0],
                    )
                )
                return plan
            if kind is None:
                raise ValueError(
                    f"cannot infer shard rule for {param_name}: "
                    f"param {tuple(param_shape)} vs ckpt {loc.shape}"
                )
            rows_local = loc.shape[0] // world if kind == _KIND_COLUMN else 0
            plan.parts.append(
                LoadPart(ckpt_name=param_name, kind=kind, rows_local=rows_local)
            )
            return plan

        # tied lm_head: checkpoint stores the embedding only once, under
        # embed_tokens.weight; the tied param survives _filter_subtensors as
        # "lm_head.weight" (lexicographically smaller than model.embed_tokens...)
        if param_name.endswith("lm_head.weight"):
            for cand in self.ckpt:
                if cand.endswith("embed_tokens.weight"):
                    loc = self.ckpt[cand]
                    plan.parts.append(
                        LoadPart(
                            ckpt_name=cand,
                            kind=_KIND_COLUMN,
                            rows_local=param_shape[0],
                        )
                    )
                    return plan

        return None  # no checkpoint tensor — model-internal buffer

    def plan_all(
        self, state_dict_shapes: Dict[str, Tuple[int, ...]]
    ) -> Tuple[Dict[str, ParamPlan], List[str]]:
        plans, missing = {}, []
        for name, shape in state_dict_shapes.items():
            p = self.plan_param(name, shape)
            if p is None:
                missing.append(name)
            else:
                plans[name] = p
        return plans, missing


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class Hf3fsModelLoader(BaseModelLoader):
    """Load model weights from a 3FS mount with per-rank sliced reads."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        allowed = {"slice_threshold", "iov_size", "io_entries", "numa"}
        unexpected = set(extra.keys()) - allowed
        if unexpected:
            raise ValueError(f"unexpected hf3fs loader config keys: {unexpected}")
        self.slice_threshold = int(extra.get("slice_threshold", 1 << 20))
        self.iov_size = int(extra.get("iov_size", _IOV_SIZE))
        self.io_entries = int(extra.get("io_entries", 1024))
        self.numa = int(extra.get("numa", -1))

    def download_model(self, model_config: ModelConfig) -> None:
        if not os.path.isdir(model_config.model_path):
            raise ValueError(
                f"hf3fs loader expects a local (3FS-mounted) model dir, "
                f"got {model_config.model_path}"
            )

    def _fuse_read_tensor(self, loc: TensorLoc) -> torch.Tensor:
        with open(loc.file, "rb") as f:
            f.seek(loc.start)
            buf = f.read(loc.nbytes)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return torch.from_numpy(arr.copy()).view(loc.torch_dtype).reshape(loc.shape)

    def _read_column_segment(
        self,
        reader: UsrbioReader,
        fd_cache: Dict[str, int],
        loc: TensorLoc,
        rows_local: int,
        tp_rank: int,
    ) -> torch.Tensor:
        ranges = column_segment_ranges(loc, tp_rank, rows_local)
        if not ranges:
            return None  # fully-padding rank
        fd = fd_cache.get(loc.file)
        if fd is None:
            fd = os.open(loc.file, os.O_RDONLY)
            rc = reader.lib.hf3fs_reg_fd(fd, 0)
            assert rc <= 0, f"hf3fs_reg_fd: {rc}"
            fd_cache[loc.file] = fd
        row_bytes = loc.nbytes // loc.shape[0]
        nrows = sum(e - s for s, e in ranges) // row_bytes
        t = reader.read_tensor(fd, ranges, loc.torch_dtype)
        return t.reshape(nrows, *loc.shape[1:])

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config,
    ) -> nn.Module:
        local_model_path = model_config.model_path
        if not os.path.isdir(local_model_path):
            raise ValueError(
                f"hf3fs loader expects a local (3FS-mounted) model dir, "
                f"got {local_model_path}"
            )

        quant_config = _get_quantization_config(model_config, self.load_config)
        tp_rank = _tp_rank()
        tp_size = _tp_size()

        t0 = time.perf_counter()
        ckpt_index = build_ckpt_index(local_model_path)
        logger.info(
            "hf3fs: indexed %d tensors from %s in %.2fs",
            len(ckpt_index),
            local_model_path,
            time.perf_counter() - t0,
        )

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)

            state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
            planner = ShardPlanner(
                ckpt_index,
                tp_rank=tp_rank,
                tp_size=tp_size,
                slice_threshold=self.slice_threshold,
            )
            plans, missing = planner.plan_all(
                {k: tuple(v.shape) for k, v in state_dict.items()}
            )
            if missing:
                logger.warning(
                    "hf3fs: %d params have no checkpoint tensor and keep their "
                    "initialized values (expected for model-internal buffers "
                    "like cos_sin_cache): %s",
                    len(missing),
                    missing,
                )

            first_shard = next(iter(ckpt_index.values())).file
            mp = extract_mount_point(first_shard)
            reader = UsrbioReader(
                mount_point=mp,
                iov_size=self.iov_size,
                entries=self.io_entries,
                numa=self.numa,
            )

            fd_cache: Dict[str, int] = {}
            n_column = n_fuse = n_rep = 0
            bytes_column = bytes_fuse = 0
            t_read0 = time.perf_counter()
            try:
                for param_name, param in list(state_dict.items()):
                    plan = plans.get(param_name)
                    if plan is None:
                        state_dict.pop(param_name)
                        continue
                    for part in plan.parts:
                        loc = ckpt_index[part.ckpt_name]
                        dst = (
                            param.data[part.expert_id]
                            if part.expert_id is not None
                            else param.data
                        )
                        if part.kind == _KIND_COLUMN:
                            seg = self._read_column_segment(
                                reader, fd_cache, loc, part.rows_local, tp_rank
                            )
                            if seg is None:
                                continue  # fully-padding rank
                            view = dst.narrow(0, part.dst_dim0_offset, seg.shape[0])
                            assert view.shape == seg.shape, (
                                f"{param_name}/{part.ckpt_name}: dst "
                                f"{tuple(view.shape)} vs seg {tuple(seg.shape)}"
                            )
                            view.copy_(seg)
                            n_column += 1
                            bytes_column += seg.numel() * seg.element_size()
                        elif part.kind == _KIND_ROW_FUSE:
                            full = self._fuse_read_tensor(loc)
                            cols_local = full.shape[1] // tp_size
                            seg = full.narrow(1, tp_rank * cols_local, cols_local)
                            assert dst.shape == seg.shape, (
                                f"{param_name}/{part.ckpt_name}: dst "
                                f"{tuple(dst.shape)} vs seg {tuple(seg.shape)}"
                            )
                            dst.copy_(seg)
                            n_fuse += 1
                            bytes_fuse += full.numel() * full.element_size()
                        else:  # replicate
                            full = self._fuse_read_tensor(loc)
                            assert dst.shape == full.shape, (
                                f"{param_name}/{part.ckpt_name}: dst "
                                f"{tuple(dst.shape)} vs full {tuple(full.shape)}"
                            )
                            dst.copy_(full)
                            n_rep += 1
                            bytes_fuse += full.numel() * full.element_size()
                    state_dict.pop(param_name)
            finally:
                for fd in fd_cache.values():
                    reader.lib.hf3fs_dereg_fd(fd)
                    os.close(fd)
                physical_read = reader.bytes_read
                reader.close()

            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

            _post_load_weights(model)

        dt = time.perf_counter() - t_read0
        logger.info(
            "hf3fs: rank %d loaded in %.2fs — column parts %d (%.2fGB logical), "
            "fuse parts %d + replicate %d (%.2fGB), usrbio physical read %.2fGB",
            tp_rank,
            dt,
            n_column,
            bytes_column / 1e9,
            n_fuse,
            n_rep,
            bytes_fuse / 1e9,
            physical_read / 1e9,
        )
        return model.eval()
