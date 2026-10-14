#!/usr/bin/env python3
"""GPU feasibility + exactness of the disk-mmap PLE gather on the real checkpoint (< 1 GiB GPU memory).
1. Bit-exact: 20,000 random logical ids gathered on the GPU straight from the mmap'd 53.7 GB file vs the
   same rows decoded on the CPU from safetensors (via TokenSpeed's own shard order: shard = id // R).
2. Cold vs warm: evict the table's pages (posix_fadvise DONTNEED, no sudo), time first and second gathers.
3. RSS / MemAvailable before and after, to show the table is not materialized."""
import json, os, random, re, time, struct
import numpy as np, torch
from safetensors import safe_open
from tokenspeed.runtime.layers.ple_mmap import open_mmap_table, ple_mmap_gather, safetensors_header

M = os.path.expanduser("~/tokenspeed-qwen/models/Qwen3.8-Flash-Next-NVFP4")
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
sh = {int(re.search(r"shard_(\d+)\.weight$", k).group(1)): (k, v) for k, v in idx.items() if ".ngram_embedding.shard_" in k}
assert sorted(sh) == list(range(128))
path = f"{M}/{sh[0][1]}"; names = [sh[i][0] for i in range(128)]
hdr, _ = safetensors_header(path)
R, D = hdr[names[0]]["shape"]; scale_name = [k for k in hdr if k.endswith("ngram_embedding.weight_scale")][0]
def mem(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
def rss(): return int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) // 1024
print(f"file={os.path.basename(path)} rows/shard={R} dim={D} shards=128 total_rows={R*128}")
with safe_open(path, "pt", device="cpu") as f: scale = float(f.get_tensor(scale_name).float().item())
print(f"weight_scale={scale}")
fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
m0, r0 = mem(), rss()
owner, offsets, bases = open_mmap_table(path, names)
dev = torch.device("cuda")
bt = torch.tensor(bases, dtype=torch.int64, device=dev)
print(f"after mmap: MemAvailable {mem()} MiB (was {m0}), RSS {rss()} MiB (was {r0})")

rng = random.Random(1234); N = 20000
ids_list = [rng.randrange(R * 128) for _ in range(N)]
ids = torch.tensor(ids_list, dtype=torch.int64, device=dev)
out = torch.empty((N, D), dtype=torch.bfloat16, device=dev)
torch.cuda.synchronize(); t = time.perf_counter()
ple_mmap_gather(ids, bt, out, rows_per_shard=R, head_dim=D, scale=scale); torch.cuda.synchronize()
cold = time.perf_counter() - t
t = time.perf_counter()
ple_mmap_gather(ids, bt, out, rows_per_shard=R, head_dim=D, scale=scale); torch.cuda.synchronize()
warm = time.perf_counter() - t
# a different, unseen id set at decode-like size (ngram heads x tokens): typical per-step gather
ids2 = torch.tensor([rng.randrange(R * 128) for _ in range(16 * 8)], dtype=torch.int64, device=dev)
out2 = torch.empty((ids2.numel(), D), dtype=torch.bfloat16, device=dev)
ts = []
for _ in range(50):
    torch.cuda.synchronize(); t = time.perf_counter()
    ple_mmap_gather(ids2, bt, out2, rows_per_shard=R, head_dim=D, scale=scale); torch.cuda.synchronize()
    ts.append(time.perf_counter() - t)
print(f"gather {N} rows: cold {cold*1e3:.1f} ms, warm {warm*1e3:.2f} ms; 128-row gather median {sorted(ts)[25]*1e6:.0f} us")

# CPU reference: decode the same rows from the safetensors tensors (TokenSpeed concatenation order)
ref = torch.empty((N, D), dtype=torch.float32)
by_shard = {}
for i, g in enumerate(ids_list): by_shard.setdefault(g // R, []).append(i)
with safe_open(path, "pt", device="cpu") as f:
    for s_i, rows in by_shard.items():
        sl = f.get_slice(names[s_i])
        for i in rows:
            local = ids_list[i] % R
            ref[i] = sl[local:local + 1, :].to(torch.float32)[0] * scale
ref_bf16 = ref.to(torch.bfloat16)
got = out.cpu()
mism = (got.view(torch.int16) != ref_bf16.view(torch.int16)).any(dim=1).sum().item()
print(f"BIT-EXACT rows: {N - mism}/{N} (mismatched rows: {mism})")
print(f"end: MemAvailable {mem()} MiB, RSS {rss()} MiB, GPU allocated {torch.cuda.max_memory_allocated() >> 20} MiB")
print("FEASIBILITY_PASS" if mism == 0 else "FEASIBILITY_FAIL")
