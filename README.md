# Qwen3.8-Flash-Next-NVFP4 on 2× DGX Spark (TP2 + MTP4, SGLang)

Day-zero deployment of **Qwen3.8-Flash-Next** (NVFP4 quant) served across **two NVIDIA DGX Spark (GB10 / SM121)** boxes as tensor-parallel-2, over the 200G ConnectX fabric. Brought up live on **2026-08-26**, the day the model dropped.

**Live result:** serving on `:8000`, 262,144-token context, **~33 decode tok/s with MTP4 speculative decoding on** (up from ~20 tok/s without — about **64% faster**), TTFT ~0.5s.

The model ships a built-in MTP layer (`mtp_num_hidden_layers: 1`), so speculative decode needs **no separate draft checkpoint** — it self-speculates via NEXTN/EAGLE. See the launcher and the "MTP4" section below.

---

## TL;DR for anyone trying this on DGX Spark

1. **vLLM cannot serve this model yet.** The checkpoint declares `Qwen4ExpForConditionalGeneration`; neither the arm64 vLLM image nor the current spark-vllm nightly registers that architecture. **Use SGLang.**
2. **The official SGLang day-0 image needs one patch to run on DGX Spark.** Its QSA sparse-decode resolver gates the fast FlashInfer TRT-LLM kernel behind `is_sm100_supported()`, so on SM121 (GB10) it falls back to an FA4 CUTE path that **dies in final warmup** with an MLIR congruence error. Fix = a one-line guard extension that also accepts `is_sm120_supported()` after verifying the TRT-LLM kernel passes a real head-shape probe on the SM121 GPU.
3. **It's a big model.** 328GB at FP8 / ~126GB at NVFP4. NVFP4 across TP2 = ~76GB weights per rank, which fits a GB10's ~120GB unified pool with room for a 484K-token KV cache.

---

## Hardware

| | |
|---|---|
| Nodes | 2× DGX Spark (GB10, compute capability **12.1 / SM121**) |
| Head | `192.168.192.1` (rank 0, serves `:8000`) |
| Worker | `192.168.192.3` (rank 1) |
| Interconnect | ConnectX over the 192.168.192.0/24 fabric, NCCL IB |
| Weights | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (~126GB on disk, ModelOpt NVFP4 W4A4, expert-scoped) |

The weights live on an NFS export the worker mounts, so there is **one copy**, read by both nodes — no second download.

## Software

| | |
|---|---|
| Base image | `lmsysorg/sglang:qwen38flashnext` (day-0, `qwen4_exp` native, sglang `0.0.0.dev1+gd91c3682b`, transformers 5.12.1) |
| Deployed image | `radixark/sglang-qwen38flashnext:sm121-qsa` — the base + the one-line SM121 QSA guard patch |
| Quant loader | `--quantization modelopt_fp4`, `--fp4-gemm-backend flashinfer_cutlass` |

The SM121 patch changes **only** the architecture guard that selects the bundled TRT-LLM sparse-decode kernel. Model, checkpoint, quant loader, and serve args are unchanged from the official image.

---

## Deploy

Copy [`launch-qwen38fn-sglang-tp2.sh`](./launch-qwen38fn-sglang-tp2.sh) to `~` on **both** nodes.

**Pre-launch on both nodes (mandatory):**
```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```
DGX Spark is unified memory (GPU + CPU share one pool). A warm page cache starves the GPU allocator ~20 min into load. `free -g` under-reports on GB10 — drop caches regardless.

**Launch, worker first:**
```bash
# Worker (192.168.192.3), rank 1
~/launch-qwen38fn-sglang-tp2.sh 1
# wait ~20s, confirm the container stays Up
# Head (192.168.192.1), rank 0
~/launch-qwen38fn-sglang-tp2.sh 0
```

Ready in **~6 minutes** end to end (206 shards, ~282s weight load). Verify:
```bash
curl http://192.168.192.1:8000/v1/models        # -> qwen3.8-flash-next
curl http://192.168.192.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello and name yourself."}],"max_tokens":40}'
```

---

## Engine receipts (what a healthy bring-up looks like)

```text
type=Qwen4ExpForConditionalGeneration, quant=modelopt_fp4, quant_algo=NVFP4
Load weight end. elapsed=282.07 s, avail mem=35.23 GB, mem usage=76.11 GB
Mamba Cache allocated: ssm_state 5.17GB
KV Cache allocated: #tokens 484032, K 2.77GB, V 2.77GB
max_total_num_tokens=484032, context_len=262144, max_running_requests=6
Uvicorn running on http://0.0.0.0:8000
```

## Benchmark

OpenAI streaming chat-completions, `max_tokens=200`, temp 0, TTFT to first content/reasoning delta, decode = `(tokens-1)/(end-first_token)`:

Baseline (no speculative decode):

| Run | TTFT | Decode tok/s |
|---:|---:|---:|
| 1 | 0.358s | 20.57 |
| 2 | 0.657s | 20.04 |
| 3 | 1.178s | 19.43 |

**With MTP4 speculative decode on** (NEXTN/EAGLE, 3 steps, top-k 1, 4 draft tokens, ReplaySSM):

| Run | TTFT | Decode tok/s |
|---:|---:|---:|
| 1 | 0.485s | 26.24 |
| 2 | 0.695s | 32.78 |
| 3 | 0.182s | 35.57 |

**Median jumps from 20.0 → 32.8 decode tok/s (~64% faster).** Draft acceptance ran ~0.36–0.56 across live traffic. MTP weights add only ~1.9GB and load in ~56s. CUDA graphs are **disabled** in this first stable bring-up (`--disable-cuda-graph`); re-enabling is a separate benchmarked change.

---

## Gotchas that cost us time

- **`max_tokens` includes hidden reasoning tokens** for this model — a small value can truncate the visible answer. The smoke reply used 21 reasoning + 10 content tokens.
- The first boot **loaded all 206 shards then died in warmup** on the FA4 CUTE fallback (`MLIRError: expects coord and shape of view are weakly congruent` → SIGQUIT). That's the SM121 kernel-guard issue above.
- Day-0 warnings that are **non-fatal**: deprecated mamba / CUDA-graph flags, unknown multimodal RoPE keys, missing `torchcodec` (audio not qualified). Image input works; audio is not.
- Validate the worker's NFS mount read **and** write before every launch (empty mount → HFValidationError; read-only mount → hung compile).

## Using it from an OpenAI-compatible client

See [`models.yml`](./models.yml) for an OMP provider block. It's a plain `/v1` endpoint, `auth: none`, model id `qwen3.8-flash-next`, 262144 context, multimodal (text + image), reasoning-enabled.

---

Brought up by the 2Wild fleet ([@Tech2Wild](https://x.com/Tech2Wild)) on race day, 2026-08-26.
Full deployment report with every timestamp and failure/fix: [`DEPLOY-REPORT.md`](./DEPLOY-REPORT.md).
