# Qwen3.8-Flash-Next on one DGX Spark (GB10) with the PLE n-gram table read from disk

Branch `ple-disk-offload`. Status: **experimental**. It serves and answers correctly, but it is far slower
than the EXL3 path on the same machine (numbers below).

## What this branch adds

- `ple_mmap_table` (hf override): the PLE n-gram embedding table (47.7 GiB of FP8 in
  `model-fp8-mtp-ple.safetensors`) is mapped read-only from the checkpoint file instead of being copied into
  pinned host memory. Lookups gather rows straight from the file mapping (`ple_mmap.py`). The loader skips the
  128 shard tensors, so they never enter RAM.
  Measured: 20,000 random rows gathered on the GPU are bit-identical to a CPU safetensors decode. In the
  running server the device memory summary shows no n-gram table resident.
- Kernel fixes needed on sm_121:
  - NVFP4 CUTLASS MoE: activation global scales passed as 0-D tensors.
  - Prebuilt CUDA modules are only reported available when they contain SASS for the current GPU.
    The nightly `tokenspeed-kernel` wheel ships sm_90a/100a/103a only, so on GB10 build it from source:
    `TOKENSPEED_CUDA_ARCH=121a pip install --no-build-isolation -e tokenspeed-kernel/python` (about 3 min).

## Launch that served (one GB10, 121.7 GiB unified memory)

```bash
tokenspeed serve nvidia/Qwen3.8-Flash-Next-NVFP4 --tensor-parallel-size 1 --trust-remote-code \
  --moe-backend auto --gpu-memory-utilization 0.80 --max-model-len 262144 --max-num-seqs 1 \
  --hf-overrides '{"ple_offload_embedding":true,"ple_embed_dtype":"float8_e4m3fn","ple_mmap_table":true}' \
  --reasoning-parser qwen3 --tool-call-parser qwen --grammar-backend xgrammar --retraction-snapshot-ratio 0
```

- Leave `--quantization` unset: the checkpoint is ModelOpt `MIXED_PRECISION` and is detected from its metadata.
- `--retraction-snapshot-ratio 0`: the retraction pool is pinned *host* memory, which is the same pool on GB10.
- Drop the checkpoint files' page cache while the server starts (`posix_fadvise(DONTNEED)`, no root).
  CUDA's free-memory query on unified memory does not count reclaimable page cache, so without it the
  KV-cache sizing fails with "no cache budget left".
- Prebuild FlashInfer's `fused_moe_120` JIT module before the first launch (about 5 min with no model
  loaded). Compiling it during startup on top of the loaded weights drove free memory under 2 GiB.
- MTP is off: the MTP draft layer's FP8 MoE selects a `flashinfer_trtllm` path that has no GB10 kernel.

## Measured (14 Oct 2026, greedy, thinking off, client-side timing from the SSE stream)

| | TokenSpeed, this branch | EXL3 3.05 bpw + TabbyAPI, same Spark |
|---|---:|---:|
| Decode tok/s, code / prose / math (400 tok, median of 3) | 16.9 / 15.8 / 16.2 | 94.0 / 56.4 / 93.8 |
| Cold prefill, 16K-token prompt | 457-505 tok/s | about 500 tok/s |
| Max input at `--gpu-memory-utilization` 0.80 / 0.85 | 39,678 / 119,550 tokens | 262,144 |
| Tool calls (7 checks) | 5/7 (no parallel calls) | 7/7 |
| Decode tok/s with the 51 GB table fully in page cache vs evicted every request | about 17 either way | — |

Why decode is slow: in this checkpoint only the routed experts are FP4. Linear attention, full attention,
shared experts, the LM head and the rest are BF16. That is about 9.4 GiB read per decoded token, which caps decode
near 27 tok/s at the GB10's 273 GB/s on any engine. A decode profile puts 46% of GPU time in BF16 GEMV
kernels and 14% in the disk-mapped n-gram gather.

Open problems:

- At 0.85 utilization (119K input) a 16K prefill pushed free memory from 10.5 GB to 5 GB, mostly mapped
  table pages (`Mapped` 5.9 GB during prefill at 0.80).
- Parallel tool calls return none.
- The QSA streaming top-k kernel requests 136 KB of shared memory, against the GB10 limit of 99 KB, on long
  prompts. A fallback is not yet verified.
