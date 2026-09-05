# Qwen3.8-Flash-Next NVFP4 on DGX Spark

Two lanes, one repo:

| Lane | What | Status |
|---|---|---|
| **[`single-spark-vllm-tp1/`](single-spark-vllm-tp1/)** | **NVIDIA's official `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint on ONE DGX Spark, vLLM main, TP1.** Our disk-backed PLE table patch keeps the 47.68 GiB n-gram table on the NVMe so the 124 GB checkpoint fits a single GB10, byte for byte, no requantizing. FP8 KV, MTP, credited upstream overlays, launcher, harness, KV ladder. | Default. Serving 2026-09-05. |
| [`lanes/sglang-tp2/`](lanes/sglang-tp2/) | The original day-0 deployment: Inferact NVFP4 on 2x DGX Spark TP2 via SGLang, with the SM121 QSA kernel-guard fix, launcher, benchmarks and the full report. | Kept as is, archived lane. |

Start with the single-Spark lane README. Numbers there are measured on the box, cold prefill only, real prompts, and the counting ceiling is reported once at the bottom, never as a headline.

Credits for anything borrowed are listed inside the lane README and in `single-spark-vllm-tp1/research/`.
