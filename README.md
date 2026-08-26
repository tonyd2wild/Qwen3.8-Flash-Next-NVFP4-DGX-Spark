# Qwen3.8-Flash-Next-NVFP4 On 2× DGX Spark (TP2 + MTP4, SGLang)

Day-zero deployment of **Qwen3.8-Flash-Next** (NVFP4 quant) served across **two NVIDIA DGX Spark (GB10 / SM121)** boxes as tensor-parallel-2, over the 200G ConnectX fabric. Brought up live on **2026-08-26**, the day the model dropped.

**Live result:** serving on `:8000`, 262,144-token context, TTFT ~0.2s warmed. With MTP4 speculative decoding on, decode is **content-dependent: ~33 tok/s on freeform prose, up to ~55 tok/s on code / structured output** — versus **~20 tok/s with MTP off**. That's a **1.65x–2.7x** speedup, and it's why real agent/tool workloads (which generate structured, predictable output) feel like the top of that range. Full warmed benchmark below.

The model ships a built-in MTP layer (`mtp_num_hidden_layers: 1`), so speculative decode needs **no separate draft checkpoint** — it self-speculates via NEXTN/EAGLE. See the launcher and the "MTP4" section below.

---

## TL;DR for anyone trying this on DGX Spark

1. **vLLM cannot serve this model yet.** The checkpoint declares `Qwen4ExpForConditionalGeneration`; neither the arm64 vLLM image nor the current spark-vllm nightly registers that architecture. **Use SGLang.**
2. **The official SGLang day-0 image needs one patch to run on DGX Spark.** Its QSA sparse-decode resolver gates the fast FlashInfer TRT-LLM kernel behind `is_sm100_supported()`, so on SM121 (GB10) it falls back to an FA4 CUTE path that **dies in final warmup** with an MLIR congruence error. Fix = a one-line guard extension that also accepts `is_sm120_supported()` after verifying the TRT-LLM kernel passes a real head-shape probe on the SM121 GPU.
3. **It's a big model.** 328GB at FP8 / ~126GB at NVFP4. NVFP4 across TP2 = ~76GB weights per rank, which fits a GB10's ~120GB unified pool with room for a **561K-token KV cache** at `--mem-fraction-static 0.82` (still ~16GB free per node).

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
Load weight end. elapsed=282.07 s, mem usage=76.11 GB (weights, per rank)
Mamba Cache allocated: ssm_state 5.17GB
KV Cache allocated: #tokens 561024, K 3.21GB, V 3.21GB
max_total_num_tokens=561024, context_len=262144, max_running_requests=6, available_gpu_mem=16.07 GB
Uvicorn running on http://0.0.0.0:8000
```

(KV at `--mem-fraction-static 0.82`: **561,024 tokens**, up 44% from 390,208 at 0.78, with ~16GB still free per node — no GB10 memory cliff. Bump the fraction when the pair is dedicated; back off toward 0.78 if you co-tenant another model.)

## Benchmark

OpenAI streaming chat-completions, `max_tokens` 200–500, temp 0, warmed, run **on the head node against localhost** (no Tailscale hop polluting the decode timing). Decode = `(completion_tokens - 1) / (t_last_token - t_first_token)`.

**No-MTP baseline** — plain autoregressive, deterministic (doesn't warm up):

| Run | TTFT | Decode tok/s |
|---:|---:|---:|
| median | 0.66s | **20.0** |

**With MTP4 on** (NEXTN/EAGLE, 3 steps, top-k 1, 4 draft tokens, ReplaySSM) — decode swings with draft acceptance, which tracks how predictable the output is:

| Content type | Decode tok/s | vs baseline |
|---|---:|---:|
| Counting / highly predictable | **54.6** | 2.7x |
| Alphabet / repetitive | 50.7 | 2.5x |
| Code generation | 46.9 | 2.3x |
| Mixed / typical prompts (median of 10) | 33.2 | 1.65x |
| Dense freeform reasoning prose | 34.1 | 1.7x |

**Read this correctly:** MTP4 self-drafts ~4 tokens per forward pass off the built-in MTP layer and verifies them in one shot. When the model can predict its own next tokens — code, lists, tool arguments, structured output — draft acceptance approaches 100% and you get the **~50–55 tok/s** top end. On unpredictable freeform prose acceptance drops to ~0.36–0.56 and you land ~33. **Agentic/tool workloads live in the high-acceptance regime**, so real agent use sits near the top. MTP weights add only ~1.9GB and load in ~56s.

**Headroom — CUDA graphs are OFF here** (`--disable-cuda-graph` in this first stable bring-up; the SM121 QSA guard is a single-line kernel-selector patch, not yet verified graph-capture-safe). Enabling decode CUDA graphs is the known lever to lift the whole curve — the MiaAI-Lab dual-Spark recipe reports ~64 tok/s single-stream / ~117 dual-stream with graphs on and a full cuda-graph-safe Triton QSA fallback. See [Roadmap](#roadmap-to-match-the-fastest-recipes).

---

## Gotchas that cost us time

- **`max_tokens` includes hidden reasoning tokens** for this model — a small value can truncate the visible answer. The smoke reply used 21 reasoning + 10 content tokens.
- The first boot **loaded all 206 shards then died in warmup** on the FA4 CUTE fallback (`MLIRError: expects coord and shape of view are weakly congruent` → SIGQUIT). That's the SM121 kernel-guard issue above.
- Day-0 warnings that are **non-fatal**: deprecated mamba / CUDA-graph flags, unknown multimodal RoPE keys, missing `torchcodec` (audio not qualified). Image input works; audio is not.
- Validate the worker's NFS mount read **and** write before every launch (empty mount → HFValidationError; read-only mount → hung compile).

## ⚠️ Day-0 bug: the endless `!!!!` loop (and the fix)

**Symptom:** mid-session, agentic requests (thinking + tools) degrade into an endless stream of `!!!!!!!!` until `max_tokens`, wedging multiple concurrent sessions while fresh simple requests keep working.

**Root cause — not hardware/NaN/thermal/the SM121 patch.** It's an upstream day-0 SGLang bug: [sgl-project/sglang#36537](https://github.com/sgl-project/sglang/issues/36537). When **thinking mode + OpenAI `tools` + `--tool-call-parser qwen3_coder`** interact, the model emits token ID 0 in a deterministic loop, and token 0 in Qwen's vocab decodes as `!`. Every agentic session sends tools and thinks by default, so all sessions eventually trip it: the speculative accept rate flatlines to 0.00 and all request slots fill with zombie generations (clients disconnected, server keeps generating to max_tokens).

Two faces of the same immature stack: **without** the `qwen3_coder` parser, raw `<tool_call>` XML leaks into content instead; **with** it, you get this loop. You need the parser *and* the thinking-off default.

**The fix (already in [`launch-qwen38fn-sglang-tp2.sh`](./launch-qwen38fn-sglang-tp2.sh)):** a server-side default that turns thinking off for every request:
```
--tool-call-parser qwen3_coder \
--default-chat-template-kwargs '{"enable_thinking": false}'
```
Tool calling stays structured (verified returning real `tool_calls` with correct JSON, `reasoning_tokens: 0`), and turns are faster with no hidden reasoning burn. A caller *can* still opt back into thinking with `"chat_template_kwargs": {"enable_thinking": true}` — **do not do this in sessions that carry tools** until #36537 is fixed upstream.

**Wedge signature to watch for** (bounce the server if you see it sustained): `Decode batch … accept len: 1.00, accept rate: 0.00` across all running requests + `Received output … but the state was deleted in TokenizerManager` spam.

**Relaunch trap:** tear down **both** nodes first (`docker rm -f sglang_qwen38fn` on the head AND the worker) before launching. If the old head is still up when the new worker starts, the worker rendezvouses with the dying head, dies with "Connection reset by peer" (exit 0, deceptively clean), and the new head hangs forever at "Init torch distributed begin." Capture `docker logs` **before** tearing down — `docker rm -f` destroys the evidence.

Full write-up: [`KNOWN-ISSUES-thinking-tool-loop.md`](./KNOWN-ISSUES-thinking-tool-loop.md).

## Using it from an OpenAI-compatible client

See [`models.yml`](./models.yml) for an OMP provider block. It's a plain `/v1` endpoint, `auth: none`, model id `qwen3.8-flash-next`, 262144 context, multimodal (text + image). **Thinking is off by default server-side** (see the bug above); the model card still advertises thinking levels, but leave them unused in tool-carrying sessions for now.

---

## Roadmap to match the fastest recipes

This deploy prioritized a **stable, agent-safe** bring-up on day zero over squeezing peak tok/s. Known levers, in order of expected payoff:

1. **Enable decode CUDA graphs.** Biggest win. Currently off because our SM121 QSA fix is a one-line guard that re-selects the bundled TRT-LLM kernel, and that path is not yet verified safe under graph capture. The [MiaAI-Lab dual-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) ships a full cuda-graph-compatible Triton FlashDecoding varlen QSA fallback and reports **~64 tok/s single-stream / ~117 dual-stream** with graphs on — proof the headroom is real. Path: verify our kernel under capture, or adopt a graph-safe fallback, then drop `--disable-cuda-graph`.
2. ~~Raise `--mem-fraction-static` 0.78 → 0.82.~~ **Done** — this deploy runs 0.82, giving **561,024 KV tokens** (+44% over 0.78's 390,208), ~16GB still free per node, no GB10 cliff. Decode speed is unchanged (KV headroom, not throughput). Back off toward 0.78 only if you co-tenant another model on the pair.
3. **YaRN-scale context past native 262K.** Mia runs 900K (RoPE factor 4.0). Only worth it if you actually feed >262K-token prompts; prefill time on Spark is the real limit at that scale, not memory.

What this repo has that the raw-speed recipes don't: the day-0 **`!!!!` token-0 loop fix** (agent-safe tool calling), the **both-nodes-down relaunch trap**, and a companion **3090 GGUF lane** for non-Blackwell hardware ([fleet repo](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy)).

---

Brought up by the 2Wild fleet ([@Tech2Wild](https://x.com/Tech2Wild)) on race day, 2026-08-26.
Full deployment report with every timestamp and failure/fix: [`DEPLOY-REPORT.md`](./DEPLOY-REPORT.md). Warmed re-benchmark + the day-0 loop fix verified live the same evening.
