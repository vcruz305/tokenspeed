"""Read-only safetensors mappings and Triton gathers for PLE tables."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import triton
import triton.language as tl


def safetensors_header(path):
    """Return the header and absolute data-section offset after validation."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > 128 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header length {header_len}")
        encoded = handle.read(header_len)
        if len(encoded) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(encoded)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return header, 8 + header_len


def safetensors_tensor_data_offset(path, name):
    header, data_start = safetensors_header(path)
    entry = header.get(name)
    if not isinstance(entry, dict) or "data_offsets" not in entry:
        raise KeyError(f"{path}: tensor {name!r} is absent from safetensors header")
    start, end = entry["data_offsets"]
    if start < 0 or end < start or data_start + end > os.path.getsize(path):
        raise ValueError(f"{path}: invalid data offsets for {name!r}")
    return data_start + start


def shard_payload_offsets(path):
    """Return absolute payload offsets for PLE n-gram shard tensors."""
    header, data_start = safetensors_header(path)
    result = {}
    for name, entry in header.items():
        if ".ngram_embedding.shard_" not in name or not name.endswith(".weight"):
            continue
        start, end = entry["data_offsets"]
        if start < 0 or end < start or data_start + end > os.path.getsize(path):
            raise ValueError(f"{path}: invalid data offsets for {name!r}")
        result[name] = data_start + start
    return result


def open_mmap_table(path, shard_names):
    """Open a checkpoint file read-only and return its mapping and row bases."""
    mapping = np.memmap(Path(path), mode="r", dtype=np.uint8)
    offsets = shard_payload_offsets(path)
    missing = [name for name in shard_names if name not in offsets]
    if missing:
        raise KeyError(f"{path}: missing PLE shard tensors: {missing[:4]}")
    bases = [int(mapping.ctypes.data) + offsets[name] for name in shard_names]
    return mapping, offsets, bases


@triton.jit
def _ple_mmap_gather_kernel(
    ids_ptr,
    bases_ptr,
    out_ptr,
    scale,
    rows_per_shard: tl.constexpr,
    num_shards: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One n-gram row per program, read straight from the file-backed mapping.

    Same addressing as TokenSpeed's ``_ple_host_gather_kernel``: the shard base is a host virtual
    address (valid on the device under ATS / unified addressing), cast once as a scalar to an FP8
    pointer, plus row and column offsets. ``scale`` is the checkpoint's per-tensor dequant factor,
    applied exactly as the pinned-host FP8 path applies it.
    """
    row = tl.program_id(0)
    logical_id = tl.load(ids_ptr + row)
    valid = (logical_id >= 0) & (logical_id < rows_per_shard * num_shards)
    safe_id = tl.where(valid, logical_id, 0)
    shard = safe_id // rows_per_shard
    local = safe_id % rows_per_shard
    base = tl.load(bases_ptr + shard)
    table = base.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < head_dim
    values = tl.load(table + local * head_dim + offsets, mask=mask & valid, other=0.0).to(tl.float32)
    values = values * scale
    tl.store(
        out_ptr + row * head_dim + offsets,
        tl.where(valid, values, 0.0).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_mmap_gather(ids, bases, out, *, rows_per_shard, head_dim, scale):
    """Gather FP8 rows from read-only file-backed mappings into ``out``."""
    rows = ids.numel()
    if rows == 0:
        return out
    _ple_mmap_gather_kernel[(rows,)](
        ids, bases, out, float(scale),
        rows_per_shard=int(rows_per_shard), num_shards=int(bases.numel()), head_dim=int(head_dim),
        BLOCK_D=triton.next_power_of_2(int(head_dim)),
        num_warps=1,
    )
    return out
