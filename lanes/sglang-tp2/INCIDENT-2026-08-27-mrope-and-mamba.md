# QWEN 3.8 FLASH NEXT — VISION-ON 70 TOK/S DEPLOY (FIX HANDOFF)

Date: 2026-08-27
Author: Knox (5080), on Tony's direct order
For: the next agent on the Spark lane. Companion to `QWEN38FN-SPARK-LANE-HANDOFF.md` (read that first for full history).

## What Tony ordered

Tear down GLM TP4, redeploy Qwen 3.8 Flash Next TP2 at the ~70 tok/s config **with vision back on** (he rejected the `language_model_only` text-only workaround), informed by fresh investigation of the NVIDIA forum thread and our own repos.

## Investigation results that shaped this deploy (3 sub-agents)

1. **NVIDIA forum thread (t/381228, all 92 posts):** no SGLang configs exist there at all; every community recipe is vLLM/llama.cpp, all eager-mode, all slower than us (best comparable: ~45 tok/s vLLM TP2 on our exact quant, post #62 oxbyte). Nobody serves this model with vision stably anywhere yet. Zero mentions of the token-0 `!!!!` loop. Post #78 is single-Spark NVMe PLE offload (interesting future lane, irrelevant to TP2). Upstream sglang PRs #36021 (mrope fused-path bypass) and #36497 (model support) both STILL unmerged; no new qwen4_exp image tag exists.
2. **Kai's repo (Mac mini `~/.openclaw/workspace/race-day-0826/qwen38fn-nvfp4-repo`):** the launcher deployed on the nodes was commit `2c99275` (text-only stability config). **HEAD `c96c6a3` "Vision is non-negotiable"** removes exactly three flags (`--json-model-override-args '{"language_model_only": true}'`, `--speculative-attention-mode decode`, `--disable-prefill-cuda-graph`) and documents the vision-on policy: graphs on = ~70 tok/s peak with a rare mrope assert (~1/90min) covered by auto-restart; graphs off = bulletproof ~55. HEAD was never deployed — GLM TP4 displaced Qwen first.
3. **Loop-fix extras from the repo we didn't have before:** residual token-0 loop exists at temperature 1.0 even with thinking off — **keep agent request temperature ≤ 0.7**. `--sampling-backend pytorch` and `--disable-radix-cache` are part of Kai's loop-fix stack; kept as-is.

## What was deployed (the fix)

**Config = Kai's HEAD (`c96c6a3`) + our mrope kernel patch + auto-restart.**

### 1. New image: `radixark/sglang-qwen38flashnext:sm121-qsa-mrope1`

Built on both Bluey and Asusi from `sm121-qsa` plus ONE guarded patch
(`~/Dockerfile.qwen38fn-sm121-mrope` on both nodes):

```python
# /sgl-workspace/sglang/python/sglang/kernels/ops/attention/rotary_triton.py:59
old:  t_mask = ~(h_mask | w_mask)
new:  t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)
```

Why: qwen4_exp is the first mrope model with partial rotary (rotary half-width 32
vs 128 padded kernel lanes). The unbounded temporal mask made the fused mrope
Triton kernel read out of bounds on every image call — the leading candidate for
the rare CUDA device-side assert that forced text-only mode. Bounding the mask
is provably a no-op for all full-rotary models (the neighboring `w_mask` at
line 65 was already bounded — that asymmetry was the tell). The build refuses to
apply if the source has drifted or is already patched.

### 2. Launcher changes vs the previous on-node version

`~/launch-qwen38fn-sglang-tp2.sh` on both nodes (worker `1` on Asusi first, wait 25s, head `0` on Bluey):

- IMAGE → `sm121-qsa-mrope1`
- `--restart no` → `--restart unless-stopped` (Kai's README recommends auto-restart to absorb a one-off assert; NCCL async error handling makes both ranks die together, then both restart and re-rendezvous)
- REMOVED `--json-model-override-args '{"language_model_only": true}'` → **vision ON**
- REMOVED `--speculative-attention-mode decode` and `--disable-prefill-cuda-graph` (per HEAD: decode+prefill graphs on = the 70 tok/s)
- Everything else unchanged from the stability config: MTP4 (NEXTN 3/4 + ReplaySSM), `--max-total-tokens 600000 --mem-fraction-static 0.80` (the GB10 OOM pin), `--cuda-graph-max-bs 8 --disable-cuda-graph-padding`, `--ple-offload-embedding`, `--disable-radix-cache`, `--sampling-backend pytorch`, `--tool-call-parser qwen3_coder`, `--default-chat-template-kwargs '{"enable_thinking": false}'`

### 3. GLM TP4 teardown (Tony's order)

`vllm_glm53` removed on all four nodes; final logs preserved as
`~/glm_tp4_final_<node>.log` on each node. GLM TP2 (Reddie+Spark4,
`launch-glm53-vllm-tp2.sh`) was NOT restored — Tony didn't order it; Reddie and
Spark4 are idle. The OMP route `glm53-nvfp4` (→ 100.113.138.96:8000) is DEAD
until someone relaunches GLM.

## Verification receipts (2026-08-27, ~02:35 UTC)

Deploy sequence clean: both-nodes teardown → drop_caches → Asusi NFS rw-check →
worker rank 1 (held 27s) → head rank 0 → SERVING.

1. **Text**: "Tensor parallelism is a technique that splits the computation of
   individual layers across multiple devices..." — `finish_reason: stop`,
   `reasoning_content: null`.
2. **Tools**: structured tool call `glob({"path": "C:/tmp/*.html"})`,
   `finish_reason: "tool_calls"`, `reasoning_tokens: 0` — thinking-off default
   active, parser active.
3. **VISION (the whole point)**: solid-red 64x64 probe → answer "Red",
   `image_tokens: 64` in usage. Full multimodal restored under CUDA graphs.
4. **Bench** (3× 200-token streamed, temp 0, cold): decode 44.3 / 49.7 / 33.8
   tok/s (median 44.3), TTFT 0.18-0.40s — squarely in the locked config's
   "47 typical" band; 70 is the warmed peak per the repo bench.
5. Image `sm121-qsa-mrope1` verified on both nodes: `rotary_triton.py:59` shows
   the bounded `t_mask`; the build's guard printed 1 occurrence patched.

## INCIDENT ADDENDUM (02:00-02:25 UTC): the wedge came back, and the real fix

The first vision-on deploy (pure repo-HEAD flags) wedged under OMP load within
minutes: all 6 slots degenerate to token 0, `accept rate: 0.00` — but with a NEW
tell: **`mamba num: 6, mamba usage: 1.00`**. Root cause found by comparing boot
receipts across every config of the day:

- **`--disable-radix-cache` collapses the mamba/SSM state pool.** With radix on,
  the extra_buffer strategy auto-sizes the pool to 97 states (~5.17GB). With
  radix off (Kai's loop-fix stack, kept in all later configs), the auto-sizer
  shrinks it to `max_running_requests` = **6** — one state per request, zero
  headroom for ReplaySSM speculative states (~4/request). Under 6 concurrent
  agents the pool exhausts, the linear-attention state corrupts, token-0 spam.
- This bomb was latent in Kai's own stability config — his 70 tok/s bench was
  single-stream and never soaked at 6 concurrent. Restoring his two removed
  flags did NOT fix it (verified: still 6). The fix is explicit:
  **`--max-mamba-cache-size 97`** (now in the launcher on both nodes).
- Also restored from `2c99275`: `--speculative-attention-mode decode` and
  `--disable-prefill-cuda-graph` (memory reclaim; repo HEAD's removal of them
  was an untested policy commit, not a validated config).
- Client-side hardening: OMP model card now has `reasoning: false` for this
  model — sending `reasoning_effort` can re-arm thinking past the server
  default and re-trigger the #36537 loop.
- **New verification rule: single-request smokes prove nothing about this bug.
  Always run the 6-way concurrent agentic load test** (script:
  scratchpad `load_test_qwen.py` pattern — 6 parallel tool-carrying requests at
  temp 0.7, grep responses for `!!!!`, then check decode logs for accept rate
  and mamba usage).

Post-fix receipts: boot line `max_mamba_cache_size: 97, ssm_state 5.17GB`;
6-way load test PASS (tool calls + coherent text, zero bang runs); under load
accept rate 0.36-0.45, mamba usage 0.04, CUDA graphs active, 100+ tok/s
aggregate.

## Soak-watch orders for the next agent

The mrope assert fired ~once per 90 minutes of load in the graphs-on config
WITHOUT our kernel patch. The patch is mechanically sound but unproven against
the production assert (a second candidate origin exists: the
`cos_sin_cache[positions]` gather in `mrope.py`). So:

1. **Watch the first 3+ hours of real traffic.** A crash shows up as the
   container restarting (`--restart unless-stopped` → check `docker ps` uptime
   resets, and `docker inspect sglang_qwen38fn --format '{{.RestartCount}}'`).
2. If the assert **never recurs** over days of vision traffic: the Triton OOB
   was the origin. File it upstream (sglang, referencing PR #36021 and the
   partial-rotary math) — it's an unreported bug and the fix is one line.
3. If it **recurs**: the origin is elsewhere. Fallback ladder, in order:
   a. Add `--disable-cuda-graph` to the launcher (vision stays, ~55 tok/s,
      Kai-validated bulletproof).
   b. Only if Tony demands 70 AND accepts no vision: re-add
      `--json-model-override-args '{"language_model_only": true}'` (last resort,
      Tony explicitly dislikes it).
4. Watch upstream: when sglang PR #36021 and #36497 merge, a real fixed image
   ends this whole patched-image era. Rebase then.
5. Wedge signature for the OTHER bug (`!!!!` loop): `accept len: 1.00, accept
   rate: 0.00` sustained + "state was deleted in TokenizerManager" spam →
   bounce. Should not happen with thinking off; if it does at temp ≤ 0.7,
   that's new information — capture logs BEFORE teardown.

## Client state after this deploy

- **OMP** (`C:\Users\tonyd\.omp\agent\models.yml`): `qwen38fn-nvfp4` flipped back to `input: [text, image]` after live vision verification. `glm53-nvfp4` marked DOWN.
- **dsh** (`C:\Users\tonyd\.dsh\settings.yaml`): `qwen38fn` route restored to `input: [ text, image ]`; still the dsh default.
- Temp discipline: agent requests ≤ 0.7 (residual loop regime at 1.0).
- The Mac mini `:8100` relay still points at dead DS4 (`~/.openclaw/workspace/ds4-lb-proxy/proxy.py`, backend Bluey:8888) — serving nothing. Kai's to fix or retire.

— Knox
