# Run Qwen 3.8 Flash on a Single DGX Spark GB10 on the OFFICIAL NVIDIA NVFP4 Quant

NVIDIA's own checkpoint, `nvidia/Qwen3.8-Flash-Next-NVFP4` (124 GB on disk: 125B-A6B hybrid MoE, 51B n-gram table, 4B MTP head, vision tower), served from **one DGX Spark (GB10, 128 GB unified memory)** with upstream vLLM and a small patch written for this lane. No requantizing, no repacking: the checkpoint is used byte for byte.

It fits because the 47.68 GiB FP8 n-gram (PLE) table never gets loaded: our patch leaves it on the NVMe and reads the 16 rows each token needs on demand. About 76 GiB of real weights stay resident, the rest of the pool is KV cache.

**Default lane: [`single-spark-vllm-tp1/`](single-spark-vllm-tp1/)** (patch, launcher, harness, results, research and credits). The original 2x Spark SGLang TP2 deployment lives on unchanged in [`lanes/sglang-tp2/`](lanes/sglang-tp2/).

![Qwen3.8-Flash-Next NVFP4 on one DGX Spark: prose by load, cold prefill, per-category speeds, KV pool by gmu](single-spark-vllm-tp1/results/chart_qwen38fn_tp1_mtp4_fp8_080_all.png)

## Shipped configuration

MTP4 on NVIDIA's own draft head, FP8 KV cache, piecewise CUDA graphs, `gpu-memory-utilization 0.80`, 262,144-token context (1M with YaRN is possible, not measured here), thinking off by default, prefix caching off, vLLM nightly `8a728663` (2026-09-04).

```bash
# on a Spark that holds the checkpoint on its local NVMe
bash single-spark-vllm-tp1/launch/qwen38fn-nvidia-tp1.sh      # defaults = this configuration
# endpoint: http://<spark>:8000/v1   model: qwen3.8-flash-next
```

KV pool at this setting: **995,129 tokens** (3.8 concurrent 262K requests). Weights load in about 10 minutes.

## Numbers (measured on the box, 2026-09-05)

Harness: 40 real prompts, 8 categories x 5, thinking off, no prefix cache, cold prefill. Per stream = the speed of one reply; aggregate = total tokens/s across all concurrent streams. Counting prompts are excluded from every number; the count-to-100 sweep is a footnote in the lane README.

### Single stream

| Category | tok/s | TTFT |
|---|---|---|
| Prose | 21.7 | 0.30 s |
| Coding | 34.4 | 0.27 s |
| Math / logic | 36.0 | 0.31 s |
| JSON | 43.7 | 0.34 s |
| HTML | 42.4 | 0.32 s |
| Narrative | 18.7 | 0.30 s |
| Summary | 21.2 | 0.27 s |
| Format | 22.2 | 0.29 s |
| Median of all 40 prompts | 32.5 | 0.30 s |

MTP4 acceptance: 64% overall, 3.56 tokens accepted per step on average. Eager floor without MTP or graphs is 15.4 tok/s on every category. MTP3 was A/B'd against MTP4 on the same 8 prompts (`single-spark-vllm-tp1/results/ab_mtp3_vs_mtp4.log`): a tie on real prompts (overall median 31.8 vs 30.6), MTP4 ahead only on easy-draft text, so MTP4 stays the default.

### Concurrent load

| Load | TTFT | Per stream (all prompts) | Prose per stream | Aggregate |
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
| 176K tokens | 106 s | 1,660 tok/s |
| 200K tokens | 104 s | ~1,900 tok/s |

### Decode latency tail (streaming, prose replies of ~250 words, `tools/itl_probe.py`)

| Streams | TTFT | Step latency p50 / p90 / p99 | Tokens per step | Per-token latency p50 / p99 | Per stream | Aggregate (prose) |
|---|---|---|---|---|---|---|
| 1 | 260 ms | 105 / 113 / 121 ms | 2.16 | 49 / 56 ms | 20.6 tok/s | 20.6 tok/s |
| 6 | 569 ms | 186 / 194 / 203 ms | 2.16 | 86 / 94 ms | 11.6 tok/s | 70.0 tok/s |

One SSE chunk is one MTP step. p99 sits within 10% of p50 at six streams: the on-demand table reads never show up as a tail.

### KV pool by gpu-memory-utilization (262K context, same weights in every row)

| gmu | KV dtype | MTP | KV pool (tokens) | MemAvailable after boot | Note |
|---|---|---|---|---|---|
| 0.72 | bf16 | off | 250,868 | ~28 GB | first boot |
| 0.78 | bf16 | off | 715,213 | ~18 GB | |
| 0.78 | fp8_e4m3 | off | 1,136,939 | ~19 GB | 200K prefill stress passed |
| 0.82 | fp8_e4m3 | off | 1,605,263 | ~13 GB | 176K prefill stress passed |
| **0.80** | **fp8_e4m3** | **MTP4** | **995,129** | **~16 GB** | **shipped default** |
| 0.85 | fp8_e4m3 | MTP4 | 1,170,740 | ~10 GB | serves, passed the 176K stress, but sits at the memory floor |

The MTP head costs roughly 290K tokens of pool at the same gmu. Pick your own headroom; around 10 GB available is where single-Spark runs start to swap.

## What is in the patch, and credits

- `ple_mmap.py` and `ple_layer.py`: ours. Disk-backed PLE table (positional reads on a thread pool, byte-exact against the checkpoint), 1-row placeholder weight, shard drop at load, a custom op so the gather is a CUDA-graph split point.
- `modelopt.py` overlay: ours. Two fixes so NVIDIA's MTP head loads (draft-local layer index in the quant config; a branch for 128x128 block-scaled FP8 experts). Being reported upstream.
- Credited upstream overlays, unmodified: vLLM PR #55375 (peakcrosser7, MTP conv-state stride fix) and PR #54846 (andreasgru, FP8 KV on the QSA attention path). Leaving the table on disk as an idea is shared with vLLM PR #54129 (Trosfy) and other single-Spark runs; the implementation here is independent. Full prior-art notes and credit map in `single-spark-vllm-tp1/research/`.

Details, diffs, launcher knobs, the harness and every raw result: [`single-spark-vllm-tp1/README.md`](single-spark-vllm-tp1/README.md).
