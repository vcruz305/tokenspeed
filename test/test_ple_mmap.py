import json
import struct
from types import SimpleNamespace

import numpy as np
import torch

from tokenspeed.runtime.layers.ple_lookup import PLELookup
from tokenspeed.runtime.layers.ple_mmap import open_mmap_table, safetensors_tensor_data_offset


def _write_safetensors(path, arrays):
    payload = bytearray()
    header = {}
    for name, array in arrays.items():
        start = len(payload)
        payload.extend(array.tobytes())
        header[name] = {
            "dtype": "F8_E4M3",
            "shape": list(array.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_safetensors_absolute_offsets_preserve_payload_order(tmp_path):
    names = [
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
    ]
    arrays = [np.full((3, 4), 11, np.uint8), np.full((3, 4), 7, np.uint8)]
    path = tmp_path / "weights.safetensors"
    _write_safetensors(path, dict(zip(names, arrays)))
    mapping, offsets, _ = open_mmap_table(path, names)
    for name, array in zip(names, arrays):
        assert bytes(mapping[offsets[name] : offsets[name] + array.nbytes]) == array.tobytes()
        assert offsets[name] == safetensors_tensor_data_offset(path, name)


def test_mmap_row_order_matches_normal_ple_shard_loader(tmp_path):
    rows_per_shard, width, shard_count = 3, 4, 128
    names = [
        f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
        for i in range(shard_count)
    ]
    # Physical order is deliberately shuffled independently of logical IDs.
    arrays = {
        name: np.full((rows_per_shard, width), i, np.uint8)
        for i, name in reversed(list(enumerate(names)))
    }
    path = tmp_path / "shuffled.safetensors"
    _write_safetensors(path, arrays)
    mapping, _, bases = open_mmap_table(path, names)

    lookup = PLELookup.__new__(PLELookup)
    torch.nn.Module.__init__(lookup)
    lookup.ngram_embedding = SimpleNamespace(
        org_vocab_size=rows_per_shard * shard_count,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=rows_per_shard * shard_count,
        ),
        weight=torch.nn.Parameter(
            torch.empty(rows_per_shard * shard_count, width, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    lookup.ngram_embedding_scale = None
    lookup.embed_store_dtype = torch.float8_e4m3fn
    lookup.offload_embedding = False
    lookup._checkpoint_weight_scale = 1.0

    # This is the production loader's shard-to-logical-row placement path.
    for shard_id in range(shard_count):
        source = torch.from_numpy(arrays[names[shard_id]].copy()).view(torch.float8_e4m3fn)
        lookup.load_shard(source, shard_id, shard_count)

    rng = np.random.default_rng(431)
    for logical_id in rng.integers(0, rows_per_shard * shard_count, size=10_000):
        shard_id, local_row = divmod(int(logical_id), rows_per_shard)
        mapped = np.ndarray(
            (width,), dtype=np.uint8, buffer=mapping,
            offset=bases[shard_id] - int(mapping.ctypes.data) + local_row * width,
        )
        expected = lookup.ngram_embedding.weight[logical_id].view(torch.uint8).numpy()
        assert np.array_equal(mapped, expected)


def test_qwen4_exp_runtime_module_imports():
    import tokenspeed.runtime.models.qwen4_exp  # noqa: F401


def test_missing_optional_mxfp4_kernels_fail_only_when_used():
    from tokenspeed.runtime.layers.dense.mxfp4 import mxfp4_linear

    try:
        mxfp4_linear(None, None)
    except ImportError as error:
        assert "MXFP4 dense kernels are unavailable" in str(error)
    else:
        # A complete kernel installation may expose the optional API.
        assert callable(mxfp4_linear)
