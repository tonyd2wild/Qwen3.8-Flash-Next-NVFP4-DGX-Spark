# Qwen3.8-Flash-Next NVFP4 — Vision Restore Proposal (mrope crash fix)

Date: 2026-08-27 (investigation night of 08-26)
Author: Knox (5080), from three parallel read-only investigations
Status: **PROPOSAL ONLY — nothing deployed.** For Tony + Kai to decide.

## What we're solving

Kai's stability relaunch (~22:43 UTC) set `language_model_only: true` to stop an
mrope crash on image inputs — trading vision away for a stable ~70 tok/s. Tony
wants vision back. This doc is the researched path to that.

## Findings (three independent agents, all read-only)

**1. The quant is innocent.** RadixArk's `config.json` is byte-identical to the
official `Qwen/Qwen3.8-Flash-Next` in every rope and vision key (verified against
official BF16, official FP8, and a second NVFP4 quant). The vision tower is
unquantized ("byte-identical to source" per the model card). No config edit can
fix this — and renaming `rope_type` to `"mrope"` would BREAK boot (SGLang's rope
factory has no such type). Do not touch the checkpoint.

**2. The boot warning is a red herring.** `Unrecognized keys in rope_parameters
... mrope_interleaved, mrope_section` comes from transformers' validator, which
WARNS but does not strip the keys (verified in source at v5.12.1 AND main, and
by a CPU repro inside a throwaway container: after loading, `model_is_mrope=True`
and all mrope keys intact). Config parsing, processor, and rope construction in
the day-0 image are all correct.

**3. The real defect: CUDA device-side assert in the multimodal-rope path with
CUDA graphs ON.** Image requests engage 2D `(3, N)` mrope position code that text
never touches. The assert is asynchronous and surfaces at the next host sync in
`_compute_mrope_positions_extend` — that frame is the surface point, not the
origin. Rare (~once per 90 min under vision load), which is why single probes
pass.

**4. A concrete latent bug was found by reading the kernel math** (candidate
origin, unproven on GPU): in `kernels/ops/attention/rotary_triton.py`, the fused
mrope Triton kernel pads offsets to `next_power_of_2(head_size)//2 = 128` lanes,
but this model's rotary cache row is only 32 wide. The temporal mask
`t_mask = ~(h_mask | w_mask)` is unbounded, so ~96 lanes read out of bounds on
every call. Every earlier mrope model (Qwen2.5/3-VL) had full-width rotary so the
masks were incidentally bounded — **qwen4_exp is the first mrope model with
`partial_rotary_factor < 1`, and it breaks that hidden assumption.** OOB reads
land ~2.5 rows past the current position; at the 262K context edge that's past
the cache end entirely.

**5. Upstream state:** nobody has filed the qwen4_exp image crash yet (we'd be
first — Kai's traceback + this analysis would make a strong report). The related
fused-mrope defect family is tracked in sglang #35345 / #35949 / #35482, with
approved-but-unmerged fixes PR #36021 (bypass fused path for multi-dim positions)
and PR #35485 (session-continuation slice guard). No fixed image tag exists;
nightlies DON'T contain qwen4_exp at all (support PR #36497 unmerged) — do not
switch tags. transformers v5.16.0 adds the real Qwen4Exp class but sglang pins
5.12.1; upgrading is a side-experiment at most.

## The options, ranked

### Option 1 — launcher-only, zero code risk (RECOMMENDED first step)
Drop `--json-model-override-args '{"language_model_only": true}'` and add
`--disable-cuda-graph`.

- Vision fully restored; no graph-captured mrope path, so the assert class is
  avoided entirely.
- **Already validated: this is the exact "bulletproof fallback" documented in
  Kai's own fleet repo README** (which at HEAD says "Vision is non-negotiable"
  and removes language_model_only — the repo and the deployed launcher have
  diverged; that divergence is itself worth reconciling).
- Cost: ~55 tok/s instead of ~70 (still >2.5x the original 20).

### Option 2 — one-line Triton mask guard (keeps graphs + vision + ~70 tok/s)
Dockerfile-layer patch in the exact style of the existing SM121 QSA fix
(`Dockerfile.qwen38fn-sm121`, with the `count(old) != 1` refusal guard):

```python
# kernels/ops/attention/rotary_triton.py, interleaved branch
old:  t_mask = ~(h_mask | w_mask)
new:  t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)
```

Provably a no-op for every full-rotary model; removes the OOB reads for
qwen4_exp. Risk: mechanically low, but **not yet proven to be THE production
assert** — deploy with graphs on and watch; if the assert recurs, the origin is
elsewhere (next suspect: the `cos_sin_cache[positions]` advanced-indexing gather,
which Option 3 covers).

### Option 3 — defensive position clamp (belt-and-braces only)
Clamp 2D mrope positions to `[0, cache_len-1]` before the cache gathers in
`mrope.py`. Converts any index-OOB assert into slightly-wrong rope at the extreme
context edge. Use only ON TOP of 1 or 2 — it hides real overflow bugs.

### Also worth doing regardless
- Cherry-pick upstream PR #36021 into the image when it merges (approved, CI
  re-running) — it's the upstream-blessed fix for the fused-path family.
- File the qwen4_exp crash upstream with Kai's traceback + the Triton OOB
  analysis. First report wins the fix.

## Suggested play

1. **Now:** Option 1 (launcher flags only) to get vision back stable — it's
   Kai's own repo-documented config.
2. **Next window:** build the Option 2 image, run it with graphs on under real
   vision load (90+ min soak, since the assert was ~1/90min). If it soaks clean,
   promote it and reclaim the ~70 tok/s. Add Option 3 if paranoid.
3. **Upstream:** file the issue; watch #36021/#36497 and the first image tag
   containing qwen4_exp on main.

## Fleet note (as of ~00:00 UTC 08-27)

During the investigation the Qwen container was torn down externally and a
`vllm_glm53` container appeared on Bluey — the fleet is being rearranged (Kai's
lane). This proposal assumes Qwen returns to Bluey+Asusi in roughly the 22:43
config; re-check the live launcher before applying anything.

## Receipts

- Config diffs: scratchpad `qcmp/` (official vs RadixArk vs FP8 vs Inferact).
- Crash-path file:line map inside `radixark/sglang-qwen38flashnext:sm121-qsa`:
  `mrope.py:137-165,246-274`, `rotary_triton.py:50-75,111-160`,
  `qsa_indexer.py:403-430`, `forward_batch_info.py:908-922,1241-1282`,
  factory mrope construction `factory.py:197-212`, transformers validator
  `modeling_rope_utils.py:1005-1028`.
- Kai's fleet repo: `~/.openclaw/workspace/race-day-0826/qwen38fn-nvfp4-repo`
  (Mac mini), commits `1e6bfe3` → `c96c6a3` document the assert and the
  graphs-off fallback.
- Upstream: sglang #35345 #35949 #35482 #36021 #35485 #36497, transformers
  #48337, v5.16.0/5.16.1 release notes.
