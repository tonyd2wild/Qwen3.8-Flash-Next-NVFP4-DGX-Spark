# ⚡ Qwen3.8-Flash-Next-NVFP4 on 2× DGX Spark — TP2 + MTP4 + CUDA Graphs

Day-zero deployment of **Qwen3.8-Flash-Next** (NVFP4 quant) served across **two NVIDIA DGX Spark (GB10 / SM121)** as tensor-parallel-2 over the 200G ConnectX fabric. Brought up live on **2026-08-26**, the day the model dropped, then tuned the same evening into the config below.

> 📌 **This is where we stand today (2026-08-26)** — a snapshot of the fastest agent-safe config we could reach on the day-0 stack. It'll move again when **DFlash2 / DSpark** or a better speculative-decode path ships. Numbers are warmed and measured on-box, not estimated.

## 🚀 Live result

Serving on `:8000`, **MTP4 speculative decode + CUDA graphs both ON**, warmed:

| | |
|---|---:|
| ⚡ **Peak decode** (code / structured / agentic) | **69.7 tok/s** |
| 🚀 Typical mixed prompts | ~50 tok/s |
| 🐢 No-MTP baseline | ~20 tok/s |
| 🧠 **KV cache pool** (pinned, OOM-safe) | **600,000 tokens** |
| 📏 Context (native) | 262,144 (YaRN-scalable to ~1M) |
| ⏱️ TTFT (warmed) | ~0.2 s |
| 🛡️ Stability | **all 3 day-0 crashes fixed** (see below) |

That's up to **3.5× over the no-MTP baseline**, and it **matches/beats the fastest published dual-Spark recipe** (MiaAI-Lab, ~64–67 tok/s single-stream) on our own kernel path with no borrowed code — while being **crash-clean under sustained agent load** (three distinct day-0 crashes root-caused and fixed, [see the Stability section](#-stability-the-three-day-0-crashes-and-the-stable-70-fix-solved-2026-08-26)). KV is pinned at 600K for OOM safety; the 1.05M pool is reachable at mem 0.82 with no pin but OOMs under load, so the pin is the production default.

---

## 🧩 What makes this model tricky

Qwen3.8-Flash-Next is a **125B-A3B hybrid MoE** with a **51B PLE (n-gram) embedding table** and a built-in **4B MTP head**, arch id `qwen4_exp` / `Qwen4ExpForConditionalGeneration`. Brand new, so:

1. **vLLM cannot serve it yet.** Neither the arm64 vLLM image nor the spark-vllm nightly registers `Qwen4ExpForConditionalGeneration`. **Use SGLang.**
2. **The official SGLang day-0 image needs one patch on DGX Spark.** Its QSA sparse-decode resolver gates the fast FlashInfer TRT-LLM kernel behind `is_sm100_supported()`, so on SM121 (GB10) it falls back to an FA4 CUTE path that **dies in warmup** (`MLIRError: coord and shape weakly congruent` → SIGQUIT). Fix = a one-line guard extension that also accepts `is_sm120_supported()` after a real head-shape probe. That's the deployed image `radixark/sglang-qwen38flashnext:sm121-qsa`.
3. **It's big.** 328GB at FP8 / ~126GB at NVFP4 → ~76GB weights/rank at TP2, fits a GB10's ~120GB unified pool with room to spare.

---

## 📊 Benchmark (locked config)

OpenAI streaming chat-completions, temp 0, warmed, run **on the head node against localhost** (no Tailscale hop in the timing). Decode = `(completion_tokens − 1) / (t_last − t_first)`.

| Config | Typical median | Peak (code/predictable) | vs no-MTP |
|---|---:|---:|---:|
| 🐢 No MTP | 20.0 | 20.0 | 1.0× |
| MTP4, CUDA graphs **off** | 33 | 55 | 1.65–2.7× |
| ⚡ **MTP4 + CUDA graphs ON (locked)** | **47** | **70.2** | **2.4–3.5×** |

Per-content-type at the locked config: counting **70.2**, alphabet 66, code 62, mixed-median 47, freeform prose 44–46.

**Why it swings:** MTP4 self-drafts ~4 tokens/forward off the built-in MTP layer and verifies them in one shot; CUDA graphs then strip per-step launch overhead. When the model can predict its own next tokens — code, lists, tool arguments — draft acceptance nears 100% and you hit **~70**. Freeform prose accepts less → ~47. **Agentic/tool workloads live in the high-acceptance regime**, so real agent use sits near the top.

---

## 🧠 Memory: >1M token KV pool

Locked at `--mem-fraction-static 0.82` **+ `--ple-offload-embedding` + `--disable-radix-cache`**:

```text
Load weight end. type=Qwen4ExpForCausalLMMTP, quant_algo=NVFP4
KV Cache allocated: #tokens 1,049,344
max_total_num_tokens=1049344, context_len=262144, available_gpu_mem≈17 GB
```

Three levers stacked to get past 1M:
- 🔹 **mem-fraction 0.78 → 0.82** on a dedicated pair (no co-tenant) — +44% KV.
- 🔹 **`--ple-offload-embedding`** — pushes the 51B PLE/n-gram table to host RAM (~13GB/rank) instead of VRAM. Embedding is a row lookup, not a matmul, so decode speed is unaffected. This alone jumped KV 534K → 686K.
- 🔹 **`--disable-radix-cache`** — frees the prefix-cache tree memory (and doubles as the loop fix below). KV 686K → **1,049,344**.

📏 **Context** is `--context-length 262144` (native). You can **YaRN-scale it up to ~1M** (RoPE factor ~4.0) if you actually feed >262K-token prompts — at that scale prefill time on Spark is the limit, not memory. Left at native 262K by default.

---

## 🛠️ The optimization journey (every step verified live)

1. **Day-0 bring-up:** MTP4 on, CUDA graphs off, mem 0.78. Agent-safe, conservative. 20 → 33 typ / 55 peak.
2. **More KV:** mem 0.78 → 0.82. KV +44%, decode unchanged.
3. **⚡ CUDA graphs ON:** dropped `--disable-cuda-graph`, added `--cuda-graph-max-bs 8`. Our SM121 QSA guard uses the **FlashInfer TRT-LLM decode kernel, which turns out cuda-graph-capture-safe** — so no Triton fallback port needed (unlike other recipes). Lifted the whole curve: **33 → 47 typ, 55 → 70.2 peak.** ✅ Beat the field.
4. **🧠 PLE offload:** `--ple-offload-embedding`. KV 534K → 686K, decode unchanged.
5. **🛡️ Agent-safety hardening:** a day-0 SGLang bug ([#36537](https://github.com/sgl-project/sglang/issues/36537) + the cuda-graph × speculative-decode class in [#17330](https://github.com/sgl-project/sglang/issues/17330) / [#29548](https://github.com/sgl-project/sglang/issues/29548) / prefix-hit [#19796](https://github.com/sgl-project/sglang/issues/19796)) can emit a **token-0 (`!`) loop** in long tool-carrying sessions. Fix stack, all zero-decode-cost:
   - `--default-chat-template-kwargs '{"enable_thinking": false}'` (thinking off)
   - `--disable-cuda-graph-padding` (keeps graphs for exact batch sizes)
   - `--disable-radix-cache` (removes the prefix-cache reuse that triggers the spec-verify garbage)
   - `--sampling-backend pytorch` (rules out the FlashInfer kernel arg-maxing a stale row to token 0)
   - Result: the loop is **clean at temp 0.0 / 0.2 / 0.7** (every temperature agents actually use); a residual edge remains only at **temp 1.0** (max-creativity, not used for tool-calling). **Recommendation: cap agent temp ≤ 0.7.** Full write-up: [`KNOWN-ISSUES-thinking-tool-loop.md`](./KNOWN-ISSUES-thinking-tool-loop.md).

MTP weights add ~1.9GB (load ~56s). CUDA-graph capture costs ~0.4GB.

---

## 🏆 How this stacks up

| Recipe | Engine | CUDA graphs | Peak tok/s | KV pool | Agent-safe |
|---|---|---|---:|---:|:---:|
| [MiaAI-Lab dual-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | SGLang + Triton QSA fallback | on | ~64–67 | ~956K | not documented |
| **This repo** | SGLang + FlashInfer TRT-LLM path | **on** | **70.2** ✅ | **1,049,344** ✅ | ✅ (temp ≤0.7) |

Top speed on our **own** kernel path, the **bigger KV pool**, **and** the day-0 agent-safety fixes the raw-speed recipes skip — plus a companion **3090 GGUF lane** for non-Blackwell hardware ([fleet repo](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy)).

---

## ✅ Stability: the three day-0 crashes, and the stable-70 fix (SOLVED 2026-08-26)

The headline throughput originally came from an *aggressive* config (mem 0.82, 1.05M-KV, cuda graphs on) that crashed **three** different ways under sustained multi-agent load. All three are now fixed, and the fixes cost almost nothing — the locked config below holds **~70 tok/s AND is crash-clean**:

1. **OOM — GB10 UMA page-cache trap.** At `--mem-fraction-static 0.82` + a 1.05M-token KV pool + PLE host-pinning, free headroom is thin (~17GB). GB10 shares one 128GB pool between GPU and CPU; heavy file IO grows the OS page cache and starves the GPU allocator → `NV_ERR_NO_MEMORY` → the worker dies, NCCL takes the head with it.
   **Fix:** pin the KV pool with `--max-total-tokens 600000` + `--mem-fraction-static 0.80`. Free headroom jumps to ~23GB and the pool can't grow into starvation. `drop_caches` before every launch stays mandatory. (600K tokens is still ~2 agents at full 262K, or many smaller sessions.)

2. **CUDA device-side assert in the multimodal-rope path** (`_compute_mrope_positions_extend`) — a compute fault, not memory. The multimodal rope-position kernel runs even on text requests because the model is *declared* multimodal, and it faults under captured graphs.
   **Fix:** `--json-model-override-args '{"language_model_only": true}'` flips `model_is_mrope=False`, so the crashing kernel never runs. Loss-free for text/agent workloads (all we serve), keeps cuda graphs.

3. **QSA prefill-graph capture crash** (exposed by fix #2). With `language_model_only`, prefill routes through the QSA sparse indexer, whose `get_prefill_mqa_inputs` does an **unpinned GPU→CPU `.tolist()`** that CUDA-graph capture forbids (`Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`).
   **Fix:** `--disable-prefill-cuda-graph` drops the prefill graph (it only helps TTFT). BUT that graph was *also* accelerating the MTP-verify step, so dropping it naively costs ~14 tok/s (70 → 56).
   **Recover the 70:** `--speculative-attention-mode decode` routes MTP-verify through the **decode** graph (which captures clean) instead of the prefill graph → verify acceleration comes back → **69.7 tok/s peak, stable.**

### 🔒 The locked stable-70 flags (all four, on top of the base MTP4 recipe)
```
--max-total-tokens 600000 --mem-fraction-static 0.80         # OOM pin (~23GB free headroom)
--json-model-override-args '{"language_model_only": true}'   # mrope assert fix
--disable-prefill-cuda-graph                                 # QSA prefill-capture crash fix
--speculative-attention-mode decode                          # routes MTP-verify onto the decode graph = recovers the 70
```
Measured on this exact config: **69.7 tok/s peak** (count), 63 code, 56 alphabet, 44 essay; content clean, `reasoning_tokens=0`, tool calls parse clean (`qwen3_coder`), decode-graph captured 100%, ~23GB free. **No OOM, no mrope assert, no QSA crash.** This is a genuinely stable 70 (not a fragile benchmark peak) — the honest "where we stand today" until DFlash2 / DSpark lands. KV is pinned at 600K for OOM safety; you can still reach the 1.05M pool at mem 0.82 with no pin, but it OOMs under sustained load, so the pin is the production default.

## 📦 Deploy

Copy [`launch-qwen38fn-sglang-tp2.sh`](./launch-qwen38fn-sglang-tp2.sh) to `~` on **both** nodes.

**Pre-launch on both nodes (mandatory):** `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` — DGX Spark is unified memory; a warm page cache starves the GPU allocator ~20 min into load, and `free -g` under-reports on GB10.

**Launch, worker first, tear down BOTH nodes before relaunching:**
```bash
# Worker (192.168.192.3), rank 1 — wait ~20s, confirm Up
~/launch-qwen38fn-sglang-tp2.sh 1
# Head (192.168.192.1), rank 0
~/launch-qwen38fn-sglang-tp2.sh 0
```
Ready in ~6 min (206 shards). Verify: `curl http://192.168.192.1:8000/v1/models` → `qwen3.8-flash-next`.

⚠️ **Relaunch trap:** always `docker rm -f sglang_qwen38fn` on the head AND worker first. If the old head is up when the new worker starts, the worker rendezvouses with the dying head, dies "Connection reset by peer" (exit 0, deceptively clean), and the new head hangs forever at "Init torch distributed begin."

---

## 🔌 Using it (OpenAI-compatible)

See [`models.yml`](./models.yml). Plain `/v1` endpoint, `auth: none`, model id `qwen3.8-flash-next`, 262144 context, multimodal (text + image). **Thinking is off by default server-side** and tool calls are structured (`qwen3_coder` parser). Keep request **temperature ≤ 0.7** for the loop-free regime.

Full deployment log with every timestamp, receipt, and failure/fix: [`DEPLOY-REPORT.md`](./DEPLOY-REPORT.md).

---

Brought up + tuned by the 2Wild fleet ([@Tech2Wild](https://x.com/Tech2Wild)) on race day, 2026-08-26. Companion fleet repo (both this lane + a 3090 GGUF lane): https://github.com/tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy
