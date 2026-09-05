# Qwen3.8-Flash-Next NVFP4 on ONE DGX Spark (GB10) with vLLM main: prior art, pitfalls, credit map

Condensed from a research pass on 2026-09-05 (full agent report in the session transcript; raw source dumps were saved under the session scratchpad `research/gb10/`). Our own patch is `../patch/`; nothing below was copied into it. Where we later adopt a specific idea from one of these people, the README must name them.

## What is and is not in vLLM main nightly `8a728663` (2026-09-04)
- Model code: `vllm/models/qwen4_exp/nvidia/` (PR #53896, peakcrosser7, merged 08-31). The day-0 `vllm/vllm-openai:qwen38-flash-next` image is a pre-merge branch snapshot (`release/qwen38next`) with a different tree.
- ModelOpt MIXED_PRECISION (the nvidia checkpoint): PR #54882 (sychen52, merged 09-03) = the model card's minimum commit. `--quantization modelopt` resolves to `modelopt_mixed`; FP8 PLE shards + global scale load. RadixArk's `quant_algo: NVFP4` layout still fails on main (open PR #55334, stecasta, tested on the exact same nightly on an sm_121 host).
- NO PLE offload/mmap on main: no `VLLM_PLE_*` envs, no `v1/ple_offload/`. All table-relief PRs are OPEN: #53899 CPU offload (peakcrosser7), #54371 UVA offload (peakcrosser7), #54129 mmap (Trosfy), #54070 disk offload dir (jagat-primitive-org). Hence our own patch.
- Landed AFTER our nightly, we do not have it: PR #55375 (peakcrosser7, merged 09-05): fused PLE conv assumed contiguous state indices; with mamba align mode + MTP and more than one concurrent prefill, conv state lands in the wrong slot ("Theductductduct..." repetition). Mitigation until rebase: MTP off, or max-num-seqs 1 with MTP.
- torch.compile still applied to the NVIDIA class; PR #55272 (gau-nernst, OPEN) removes it because Inductor autotune duplicates the embedding op (~50 GB). Our placeholder table sidesteps that; we run eager anyway.
- QSA requires BF16 main KV cache (`nvidia/qsa.py`); FP8 KV needs the RFC #54426 (Nanetnounou) / PR #54846 (andreasgru) changes. x1.7-1.9 KV pool at a small decode/prefill cost.
- Top-k on GB10 takes the non-deterministic `persistent_topk` path (family 120); deterministic fix PR #55122 (jschmied, OPEN), ~1.5% per decode step.

## GB10 / sm_121 traps, with who found them
- CUDA stream memory ops (`cuStreamWriteValue32`/`WaitValue32`) unavailable on GB10 (`CAN_USE_STREAM_MEM_OPS=0`, MiaAI Lab measurement, single source). Only matters for the day-0 branch's offload semaphore; our reader uses plain copies. Probe locally before relying on it.
- Blockwise FP8 GEMM on sm_121: `VLLM_USE_DEEP_GEMM=0` (issue #54125, jschmied; DeepGEMM "unspecified launch failure" at profile run; CUTLASS block-scaled path works). Relevant to the MTP FP8 experts.
- FlashInfer CUTLASS NVFP4 MoE runs on sm_121 at TP1 (#54173, brainatworkharris). TP>1 intermediate-size padding error needs expert parallel (tsw2000). Non-deterministic fused finalize atomics: PR #54948 `VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0` (jschmied). Marlin remains the fallback if CUTLASS faults at JIT (Starkweather Digital, vlaicu, Conselara).
- `--no-enable-flashinfer-autotune` everywhere; FlashInfer <= 0.6.17 autotune cache "Invalid gemm2 profile id" (jschmied); we have 0.6.18.
- Env used by the working single-GB10 vLLM runs: `CUTE_DSL_ARCH=sm_121a`, `VLLM_USE_V2_MODEL_RUNNER=1`, `TORCH_CUDA_ARCH_LIST=12.1a`, `expandable_segments:True` (hmlee0315, eugr, vlaicu).
- Prefix caching + GDN: multi-turn growing prompts crash in `precopy_mamba_align_fused_kernel` (#54173, brainatworkharris; fix = PR #48375 variant + a mamba block-size divisor fix, blazux has the "two-line" version). Not verified in our nightly: keep prefix caching OFF until tested.
- FLA GDN shared-memory gate asks for 100 KiB, sm_121 reports 99 KiB, so all GDN layers silently run small tiles (Saren-Arterius via blazux; tsw2k confirmed on nightly 09-01). Perf item.
- Full torch.compile: Inductor int64-indexing assert on sm_121 (blazux); PIECEWISE graphs with the PLE lookup as a splitting op is the working mode; eager costs ~55% decode (Conselara, other model). Trosfy's later #54129 revision keeps FULL graphs by gathering in input prep.
- Driver: 590.x has a CUDA-graph deadlock on unified memory; GB10 reporters run 580.x (eugr, Conselara).
- Long-prefill stream hang above ~74-78K tokens at TP4+EP (#54629, tsw2k); TP1 runs did 92K needle and 400K prefill fine (blazux, MiaAI). Unverified at TP1 for us.

## Memory on a 128 GB GB10
- vLLM sees 119.6-121.6 GiB depending on driver. `is_integrated_gpu()` path: free memory is reported with reclaimable page cache; vLLM fills GPU side to GMU x MemTotal and keeps nothing for the host (MiaAI reading).
- Driver refuses allocations around MemFree ~3 GiB (`NV_ERR_NO_MEMORY` in `journalctl -k`); exhausting the pool hangs the box with no OOM kill. `drop_caches` before launch is mandatory (everyone).
- Non-KV overhead on UMA is bigger than on discrete (4.78 vs 1.25 GiB same model, changtimwu). Cold AOT compile is charged as peak activation memory (#54122).
- GMU that served on ONE Spark with this model: 0.75 (hmlee0315, 128K eager), 0.78 (Starkweather, MiaAI = MemTotal minus 26 GiB host reserve), 0.80 (blazux; 0.85 drifted into swap after a day, 0.875 OOM-killed on a 300K prefill with MTP), 0.90 only with CPU offload + 64 GiB swap at 8K ctx (jschmied).
- Budget for us: ~76 GiB non-PLE weights (+~5 GiB MTP head) inside GMU x ~121.6 GiB; at 0.78 ~95 GiB GPU side, ~14-18 GiB KV (bf16 ~25.45 KiB/token, so ~600-700K tokens); the other ~27 GiB must hold OS, container, vLLM host processes and the hot page cache of the 47.68 GiB table. Keep MemAvailable >= 10 GiB (MiaAI floor).

## Single-GB10 numbers, the bar (decode single stream unless noted)
- vLLM + #54129 mmap, eager, MTP2 (RadixArk ckpt): 21-23 tok/s, prompt 4,369 tok/s, GMU 0.75, 128K (hmlee0315, 09-03).
- vLLM day-0 + blazux patches, PIECEWISE, MTP2: ~26 (22 no MTP), prefill 2,400-2,900 warm, ~580K KV @0.80 (blazux); hybrid fp8 side layers ~31.
- vLLM CPU offload, no spec, 8K: 17.1 c=1, 87.5 agg @8, 266.8 agg @48 (jschmied); jschmied full stack (det. topk, stride fix, tile-union): 36.5 single, ~100-110 agg @16-32, prefill ~2,800.
- MiaAI single (own 99 GB ckpt, packed 27 GB table mmap, FP8 KV, MTP3, reduced-vocab drafting): 46.3 prose, 108 agg @4, prefill 2,265 @32K, KV 992,584 @0.78; BF16 KV variant 28.3.
- SGLang variants: 34-43 (hasso5703, azampatti), HashK-PLE ~35; Tony's own 2-Spark SGLang TP2: 47 median / 70 peak.
- llama.cpp UD-Q4_K_XL: 19-22 (styles01); UD-IQ1_S: 34.5 (kubesimplify).

## Credit map (specific ideas by person; cite if we adopt)
Trosfy (#54129: mmap gather in input prep, stable GPU buffers, FULL graphs; `_SERIAL` inline fast path; header-read of weight_scale) · MiaAI Lab (host-side handshake replacing stream mem ops; MADV_RANDOM; HOST_RESERVE_GIB budgeting; packed 4-bit table; AGPL, code not to be copied) · blazux + Saren-Arterius (prefix-cache block_size fix, FLA 99 KiB gate, num_warps pin, gather fast path with CPU dedup + async H2D, hybrid fp8 side layers) · jschmied (deterministic topk #55122, MoE finalize switch #54948, DeepGEMM report #54125, tile-union prefill #55430, M%4 pad, methodology) · Nanetnounou + andreasgru (FP8 KV on QSA #54426/#54846) · brainatworkharris (GDN prefix-cache root cause via cuda-gdb) · tsw2k (EP fix, long-prefill wall) · peakcrosser7 (#53896, #53899, #55375) · gau-nernst (#55272, QSA kernels) · jagat-primitive-org (#54070) · stecasta (#55334) · sychen52 (#54882) · dolf3131 (uniproc spawn diagnosis) · provsalt / raymond.goo (NVFP4-packed PLE) · hmlee0315 (first #54129 single-Spark report) · Daniel Han (`CUTE_DSL_ARCH=sm_121a` origin) · Flaviu Vlaicu, Conselara, eugr (Spark vLLM env hygiene).

## What our patch does differently (for the README)
- Positional reads (`preadv`) on a thread pool instead of mmap page faults (measured on Reddie: mmap capped at 10-20K rows/s cold because faults serialize on the process mmap_lock; preadv reached 131K rows/s at 64 threads).
- No packing or requantizing of the table: NVIDIA's FP8 rows are read byte-for-byte and the stock global-scale dequant runs unchanged.
- 1-row placeholder weight created at construction, shards dropped at load (after shape validation), so the model skeleton never allocates the 47.68 GiB.
- Gather inside the layer forward (eager first; piecewise graphs need the split-op registration; full graphs need the input-prep move, which Trosfy's PR pioneered and which we would credit if we follow that shape).
