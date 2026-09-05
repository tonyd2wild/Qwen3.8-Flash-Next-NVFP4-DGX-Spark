# Qwen3.8-Flash-Next NVFP4 on ONE DGX Spark, vLLM, TP1

NVIDIA's own checkpoint, `nvidia/Qwen3.8-Flash-Next-NVFP4` (124 GB on disk, 125B-A6B hybrid MoE + 51B n-gram table + 4B MTP), served from a single DGX Spark (GB10, 128 GB unified memory) with upstream vLLM main and a three-file patch (`ple_mmap.py`, hooks in `ple_layer.py`, one line in `compilation.py`) plus a two-fix `modelopt.py` overlay, all written for this lane. No requantizing, no repacking: the checkpoint is used byte for byte.

Status: serving as of 2026-09-05. Numbers below are from the harness in `tools/`, cold prefill only, real prompts only; the count-to-100 ceiling is reported once at the bottom and never as a headline.

## Why it fits

| Piece | Size | Where it lives |
|---|---|---|
| Model weights (experts NVFP4, attention and shared experts BF16, MTP experts FP8, vision tower) | ~76 GiB | GPU (unified pool) |
| PLE n-gram table (320,001,536 rows x 160 B, FP8, one global scale) | 47.68 GiB | **Left on the NVMe.** Each token needs exactly 16 rows (2.5 KB), read on demand |
| KV cache | what is left inside `gpu-memory-utilization` x pool | GPU |

Upstream vLLM main (nightly `8a728663`, 2026-09-04) has the NVIDIA-checkpoint loader (PR #53896, #54882) but no PLE offload of any kind: the four table-relief PRs are all open. Without a patch the skeleton allocates the full 47.68 GiB table and nothing fits.

## The patch (ours)

`patch/ple_mmap.py` (new file, mounted as `vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py`)
- Parses the safetensors headers through `model.safetensors.index.json`, finds the 128 row shards of the PLE layer, keeps one file descriptor per file (`POSIX_FADV_RANDOM`).
- Gathers rows with positional reads (`preadv`) on a thread pool into a pinned staging buffer, then one async copy into a GPU buffer. An mmap version was written first and measured: page faults serialize on the process `mmap_lock` (10-20K rows/s cold regardless of thread count); `preadv` scales with NVMe queue depth (131K rows/s cold at 64 threads on the Spark's NVMe).
- Returns the FP8 rows unchanged; vLLM's stock global-scale dequant runs as before.

`patch/ple_layer.py` (nightly file + hooks, see `patch/ple_layer.diff`)
- Allocates a 1-row placeholder instead of the 47.68 GiB table at construction.
- Validates and drops the shard tensors at load time (they would otherwise hit the loader's TP-range copy).
- Looks rows up through a registered custom op (`vllm::qwen4_exp_ple_mmap_gather`) so torch.compile treats the host gather as opaque and, with `patch/compilation.py` (one line added to `_attention_ops`), as a piecewise CUDA-graph split point. This is the same graph mode blazux's recipe uses (its `vllm_ple_mmap.py` runs as an opaque splitting op under PIECEWISE) and Trosfy's early #54129 revision; here it comes from vLLM's own `_attention_ops` list, which already carries `qwen4_exp_compute_ple_ngram_ids`.
- Builds the reader once at weight-load time; the traced forward touches only tensors and constants (Dynamo cannot construct thread pools).

Measured reader cost on Reddie (CPU-only container, cold = after `drop_caches`, byte-exact against raw reads): 8K-token prefill (131,072 rows) 1.0 s cold, 0.64 s warm; 64-token decode batch 23 ms cold, 8 ms warm; single token ~0 ms warm.

## Upstream overlays (credited, unmodified upstream code)

`patch/upstream-overlays/` contains two upstream diffs applied onto the nightly's files, because they were not in the image we run:
- **PR #55375, peakcrosser7, merged 2026-09-05**: fused PLE conv state-index stride fix. Without it, MTP plus more than one concurrent prefill writes conv state to the wrong slot.
- **PR #54846, andreasgru, open**: fp8_e4m3 and nvfp4 KV cache on the QSA attention path (`--kv-cache-dtype fp8_e4m3`). Stock vLLM refuses anything but BF16 KV on this model. Related RFC #54426 by Nanetnounou documented the same four sm_121 integration points independently.

Both are upstream vLLM contributions by their authors; nothing else in this lane is derived from anyone's repo.

`patch/upstream-overlays/modelopt.py` is different: it is the nightly's file plus **two fixes of ours** for loading NVIDIA's MTP head (see `patch/modelopt_mtp_index.diff`):
- NVIDIA's `hf_quant_config.json` keys the MTP experts as `mtp.layers.0.mlp.experts` while vLLM (and the checkpoint's own tensor names) number that layer `mtp.layers.48`; the mixed-precision resolver never matched, so the experts were built without their scales. We add the draft-local index as a lookup candidate.
- The mixed-precision config had no branch for `FP8_BLOCK_SCALES` experts at all (only NVFP4, per-tensor FP8, MXFP8); the MTP experts are 128x128 block-scaled FP8. We route that algo to vLLM's generic block-FP8 MoE method.
Both are being reported upstream (`research/upstream-issue-modelopt-mtp-index.md`). The same two gaps were fixed independently the same day by sfxnz (`docker/modelopt.py` in `sfxnz/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark`, MIT, commit 13:51 UTC, about two hours before ours) and by MiaAI-Lab (`files/patch_modelopt_fp8_block_moe.py` in `Qwen3.8-Flash-Next-Dual-DGX-Sparks`, AGPL-3.0, commit 16:07 UTC); the three implementations share no code. The idea of leaving the table on disk is shared with vLLM PR #54129 (Trosfy) and with community single-Spark runs; the implementation here is independent (see `research/` for the prior-art notes and credit map).

## Launch

```bash
# on the Spark that holds the checkpoint locally (random reads over NFS would crawl)
IMAGE=vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c \
PLE_MODE=mmap PLE_WORKERS=64 GRAPHS=piecewise MTP=4 KV_DTYPE=fp8_e4m3 \
GMU=0.80 MAXLEN=262144 SEQS=8 bash launch/qwen38fn-nvidia-tp1.sh   # these are also the script's defaults
```

Knobs: `PLE_MODE` (mmap | none), `GRAPHS` (eager | piecewise), `MTP` (0 | N), `KV_DTYPE` (auto | fp8_e4m3 | nvfp4), `OVERLAYS` (1 | 0), `GMU`, `MAXLEN`, `SEQS`, `PATCH_DIR`, `EXTRA`. Prefix caching is off by default (`PREFIX_CACHE_ARG`) until the GDN prefix-cache crash reported in vLLM #54173 (brainatworkharris) is verified fixed in this image. The launcher sets `VLLM_USE_DEEP_GEMM=0` (DeepGEMM faults on sm_121: vLLM issue #54125, jschmied), `VLLM_USE_V2_MODEL_RUNNER=1` (pins the V2 runner the nightly already selects), and `--no-enable-flashinfer-autotune` (FlashInfer autotune-cache failures on GB10 reported by jschmied); `CUTE_DSL_ARCH=sm_121a`, `TORCH_CUDA_ARCH_LIST=12.1a` and `expandable_segments` are the standard Spark vLLM environment (Flaviu Vlaicu's playbook, eugr's images, and our own earlier recipes).

Endpoint: `http://<spark>:8000/v1`, model `qwen3.8-flash-next`, thinking off by default (`enable_thinking` per request).

## Memory settings measured (KV pool ledger)

One row per boot. Same weights resident in every row; only the KV pool moves.

| gmu | Context | KV dtype | Graphs | MTP | KV pool (tokens) | KV GiB | Free after boot | Result |
|---|---|---|---|---|---|---|---|---|
| 0.72 | 131,072 | bf16 | eager | 0 | 250,868 | 6.6 | ~28 GB | serves (first boot) |
| 0.78 | 262,144 | bf16 | eager | 0 | 715,213 | 17.5 | ~18 GB | serves |
| 0.78 | 262,144 | bf16 | piecewise | 0 | 581,8xx | ~14.2 | ~21 GB | serves; torch.compile + graphs reserve ~130K tokens |
| 0.78 | 262,144 | fp8_e4m3 | eager | 0 | 1,136,939 | 14.7 | ~19 GB | serves; 200K prefill stress passed |
| 0.82 | 262,144 | fp8_e4m3 | eager | 0 | 1,605,263 | 20.8 | ~13 GB | serves; 176K prefill stress passed (1,660 tok/s) |
| **0.80** | **262,144** | **fp8_e4m3** | **piecewise** | **4** | **995,129** | **16.6** | **~16 GB** | **shipped default (this README's numbers)** |
| 0.85 | 262,144 | fp8_e4m3 | piecewise | 4 | 1,170,740 | ~19.5 | ~10 GB | serves and passed the 176K stress (1,571 tok/s), but MemAvailable sat at 9.8 GB afterwards: works, tight, not the default |

MTP4's draft head costs roughly 290K tokens of pool at the same gmu. `free -g` on a GB10 counts page cache as used; the "free after boot" column is `MemAvailable`. Community reports put the danger line around 10 GB available: MiaAI-Lab's single-Spark README says to keep MemAvailable at or above ~10 GiB under load, and blazux reports gmu 0.85 drifting into swap after a day and 0.875 OOM-killed on a 300K prefill with MTP. So 0.80 with MTP is the shipped setting and 0.82 is the ceiling we would recommend without MTP.

Weights load in ~9-10 minutes: the loader still streams the 47.68 GiB PLE file through RAM before the shards are dropped. Skipping those shards in the iterator is the next load-time improvement.

## Results (shipped default: MTP4 + FP8 KV + piecewise graphs, gmu 0.80, 262K)

Harness: 40 real prompts, 8 categories x 5, thinking off, no prefix cache, one Spark (GB10). "Per stream" is the median speed of one reply; "aggregate" is total tokens/s the box delivers across all concurrent streams (all categories mixed). Cold prefill only. Counting prompts are not in any of these numbers.

### Single stream (x1), per category

| Category | tok/s | TTFT | Auto score |
|---|---|---|---|
| Prose | 21.7 | 0.30 s | 0.35 (word-range overruns, text reads well) |
| Coding | 34.4 | 0.27 s | 1.00 |
| Math / logic | 36.0 | 0.31 s | 1.00 |
| JSON | 43.7 | 0.34 s | 1.00 |
| HTML | 42.4 | 0.32 s | 0.93 |
| Narrative | 18.7 | 0.30 s | 0.69 (length) |
| Summary | 21.2 | 0.27 s | 0.75 |
| Format | 22.2 | 0.29 s | 1.00 |
| Median of all 40 | 32.5 | 0.30 s | 0.84 |

MTP4 acceptance on this checkpoint: mean accepted length 3.56, 64% overall, per position 0.90 / 0.66 / 0.53 / 0.40. Structured output (code, JSON, HTML) accepts drafts often and roughly doubles the eager floor of 15.4 tok/s; free prose accepts less and gains about 1.4x.

### Concurrent load

| Load | TTFT (median) | Per stream (median, all prompts) | Prose per stream | Aggregate |
|---|---|---|---|---|
| x1 | 0.30 s | 32.5 | 21.7 | 32.5 |
| x2 | 0.35 s | 31.0 | 18.3 | 38.8 |
| x4 | 0.44 s | 23.0 | 19.7 | 42.0 |
| x6 | 0.69 s | 19.2 | 14.7 | 62.5 |

### Cold prefill (one request, no prefix cache, needle question answered correctly every time)

| Prompt | Time to first token | Prefill rate |
|---|---|---|
| 7K tokens | 5.9 s | 1,206 tok/s |
| 28K tokens | 17.1 s | 1,654 tok/s |
| 113K tokens | 68.6 s | 1,643 tok/s |
| 176K tokens (gmu 0.82, no MTP) | 106 s | 1,660 tok/s |
| 200K tokens (gmu 0.78, no MTP) | 104 s | ~1,900 tok/s |

### Eager floor (for reference)

No graphs, no MTP, bf16 KV, gmu 0.72: 15.4 tok/s single stream on every category, 26.9 tok/s aggregate at x4, TTFT 0.26 s.

### Counting ceiling (footnote, not a benchmark)

Count-to-100 sweep (`results/sweep_qwen38fn_tp1_mtp4_fp8_080.json`), aggregate tok/s: x1 43.8, x2 89.9, x3 128.8, x4 151.9, x5 175.4, x6 194.0. This is the best case for speculative decoding (near-perfect draft acceptance) and is not representative of real work, which is why it is not in the tables above.

## Files

- `patch/` the two patched files, their diffs against the nightly, the compilation overlay, `upstream-overlays/`
- `launch/qwen38fn-nvidia-tp1.sh` single-Spark launcher
- `tools/` benchmark harness (`bench_lane.sh`, `bench_categories.py`, `bench_sweep.py`, `stress_prefill.py`, `probe-qwen38fn.sh`)
- `results/` raw JSON per run, the KV pool ledger, the harness and stress logs, and the chart images (`chart_*_all.png` is the one-image summary; `tools/make_tweet_charts.py` builds them from the JSON)
- `research/` runner internals notes and the GB10 prior-art / pitfalls survey with the credit map

## TP2 lane (two Sparks, same stack)

`launch/qwen38fn-nvidia-tp2.sh <0|1>` runs the identical stack across two Sparks over the RoCE fabric: rank 1 = worker (Spark4, 192.168.192.4, `--headless`), rank 0 = head (Reddie, 192.168.192.2, serves :8000). Start the worker first, then the head. Each rank mounts its own local copy of the checkpoint: the disk-backed table is read locally on every rank (random reads over NFS would crawl), and our lookup returns full rows on each rank, so no vocab-parallel all-reduce is needed for the table. Networking is the same NCCL-over-RoCE block the GLM TP4 lane uses (`NCCL_IB_HCA=rocep1s0f0`, GID 3, `enp1s0f0np0`).

Boot: weights 321 s + draft head 67 s + graphs 5 s, about 10 minutes launch to serve.

KV pool at gmu 0.80 (the default): **5,874,061 tokens (52 GiB), 22 concurrent 262K requests**. The head node sits at ~10.5 GB available at 0.80. If 0.80 gives you trouble on your boxes, `GMU=0.75` is the stable fallback (roughly 4.5M tokens of KV, scaled from the 0.80 measurement, not measured separately).

(TP2 harness results: filled in below once the run completes)

