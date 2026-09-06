# KV pool ledger, nvidia/Qwen3.8-Flash-Next-NVFP4 on one DGX Spark (vLLM nightly 8a728663, our PLE-on-disk patch)

Every boot, its config, and the `GPU KV cache size` line vLLM printed. Memory columns are `free -g` used/available right after boot.

| # | Node | gmu | max-model-len | KV dtype | graphs | MTP | KV pool (tokens) | KV GiB | Max concurrency at ctx | used / avail GB | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Reddie | 0.72 | 131,072 | bf16 | eager | 0 | 250,868 | 6.61 | 1.91x @128K | 94 / 28 | first serving boot |
| 2 | Spark4 | 0.78 | 262,144 | bf16 | eager | 0 | 715,213 | 17.48 | 2.73x @262K | 102 / 18 | prefix caching off |
| 3 | Spark4 | 0.78 | 262,144 | bf16 | piecewise | 0 | 581,8xx | ~14.2 | | 100 / 21 | torch.compile + graphs reserve ~130K tokens |
| 4 | Reddie | 0.78 | 262,144 | fp8_e4m3 | eager | 0 | 1,136,939 | 14.73 | 4.34x @262K | 102 / 18 | FP8 KV control; 1.59x the bf16 pool; quality clean |
| 5 | Spark4 | 0.80 | 262,144 | fp8_e4m3 | piecewise | 4 | FAILED at load | | | | MTP experts are FP8_BLOCK_SCALES; mixed ModelOpt config has no block-FP8 MoE method (upstream gap) |
| 6 | Reddie | 0.82 | 262,144 | fp8_e4m3 | eager | 0 | 1,605,263 | 20.82 | 6.12x @262K | 108 / 13 | ladder rung 2; 200K stress running |
| 7 | Spark4 | 0.80 | 262,144 | fp8_e4m3 | piecewise | 4 | 995,129 | 16.56 | 3.80x @262K | 105 / 16 | MTP4 (our two ModelOpt fixes); MTP head costs ~290K tokens vs no-MTP at the same gmu; acceptance 64%, mean accepted length 3.56 |
| 8 | Reddie | 0.85 | 262,144 | fp8_e4m3 | piecewise | 4 | 1,170,740 | ~19.5 | 4.47x @262K | 111 / 10 | ladder rung 3, full stack: serves, 176K stress passed (1,571 tok/s), but MemAvailable 10.7 GB after boot and 9.8 GB after the stress = at the floor. Works, tight; NOT the shipped default |

Stress: 200K-token prefill at row 4 (0.78, FP8 KV): needle answered correctly, TTFT 103.8 s (~1,900 tok/s prefill), MemAvailable stayed ~18.9 GB, no new driver errors.
Stress at row 6 (0.82, FP8 KV): 176,134-token prompt, needle correct, TTFT 106.1 s (1,660 tok/s prefill), no driver errors.
| 9 | Reddie | 0.80 | 262,144 | fp8_e4m3 | piecewise | 3 | 881,757 | ~14.7 | | | MTP3 twin for the A/B; smaller than Spark4's 995K at the same setting because Reddie carries the NFS server and more resident processes (node effect, not MTP3) |
| 10 | Reddie+Spark4 TP2 | 0.80 | 262,144 | fp8_e4m3 | piecewise | 4 | 5,874,061 | 52.15 | 22.41x @262K | 111 / 10 (head) | TP2 over the RoCE fabric, same stack; weights load 321 s; MemAvailable on the head ~10.5 GB, so 0.75 would be the comfortable TP2 setting |
| 11 | Reddie+Spark4+Asusi+Bluey TP4 + EP | 0.80 | 262,144 | fp8_e4m3 | piecewise | 4 | 9,088,133 | 72.24 | 34.67x @262K | 23.1 weights+non-torch / 72.2 KV per rank | TP4 needs `--enable-expert-parallel` (FlashInfer CUTLASS NVFP4 cannot pad the MoE intermediate size four ways; prior art tsw2000); Asusi reads its slice over NFS; weights 218 s + 65 s, init 135 s |
| 12 | Reddie+Spark4 TP2 | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 4 | 1,898,635 | 16.74 | 7.24x @262K | table resident (stock loader), 63 GiB weights/rank | first RAM boot that survived; slower than disk (28.4 median) |
| 13 | Reddie+Spark4 TP2 | 0.70 | 262,144 | fp8_e4m3 | piecewise (compile on) | 3 | 1,278,944 | 10.78 | 4.88x @262K | OUR resident slice (PLE_MODE=resident) | boots, 8-15 tok/s as implemented; experimental |
| 14 | Bluey+Asusi TP2 | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 4 | 1,832,462 | n/a | 6.99x @262K | table resident, chunk 4096 only (seqs 8) | 45.1 median |
| 15 | Bluey+Asusi TP2 **SPEED default** | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 3 | 1,970,051 | 16.62 | 7.52x @262K | table resident, MTP3, seqs 6, chunk 4096 | 53.7 median, 97.9 agg x6 |
| 16 | Bluey+Asusi TP2 CONTEXT+recipe | 0.80 | 262,144 | fp8_e4m3 | piecewise | 3 | 6,185,539 | n/a | 23.60x @262K | table on disk (ours), MTP3, seqs 6, chunk 4096 | booted; guard killed it at 9 GB free on the head; rerun at 0.75 pending |
| 17 | 4 Sparks TP4 **SPEED** + EP | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 3 | 6,246,558 | 48.31 | 23.83x @262K | table resident (11.9 GiB/rank), 32.63 GiB weights/rank | x1 median 29.9 (below TP4 CONTEXT 40.5): compile off costs more than the recipe gives at TP4+EP; aborted after C1 |
| 18 | 4 Sparks TP4 CONTEXT+recipe + EP | 0.75 | 262,144 | fp8_e4m3 | piecewise (compile on) | 3 | 8,595,716 | 66.37 | 32.79x @262K | table on disk, MTP3, seqs 6, chunk 4096 | x1 median 14.6, TTFT 800 ms: compile ON + chunk 4096 is the slow combination (same as row 13); aborted after C1. At 0.80 it carved 9,366,216 but the head rested at 4 GB free |
| 19 | Spark4 TP1 | 0.80 | 262,144 | fp8_e4m3 | eager (no graphs) | 3 | 1,077,703 | n/a | 4.11x @262K | table on disk, seqs 6, chunk 4096 | x1 median 35.2, prose 23.3 (shipped TP1: 32.5 / 21.7) |
| 20 | Reddie TP1 | 0.80 | 262,144 | fp8_e4m3 | eager (no graphs) | 4 | 1,038,395 | n/a | 3.96x @262K | table on disk, seqs 8, chunk 4096 | x1 median 34.1, prose 21.1 |
| 21 | Bluey+Asusi TP2 SPEED + MTP4 + index share | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 4 | 1,977,532 | n/a | 7.54x @262K | Chuck 208 knob 2 | x1 55.2 (+3%), prose 35.0 (−6%); not adopted |
| 22 | Bluey+Asusi TP2 SPEED + EP | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 3 | 2,089,208 | n/a | 7.97x @262K | Chuck 208 knob 1 | x1 51.8 (−4%); not adopted |
| 23 | Spark4 TP1 **STAGED** | 0.80 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 3 | 1,053,871 | n/a | 4.02x @262K | table on disk, rows staged in prepare_inputs (v3 patch), seqs 6, chunk 4096, capture 4..24 | x1 median 37.4, prose 25.2 (best single-Spark so far) |
| 24 | Reddie TP1 eager + DRAFT_VOCAB 65536 | 0.80 | 262,144 | fp8_e4m3 | eager | 3 | 1,080,351 | n/a | 4.12x @262K | our reduced-vocab draft (FR-Spec idea), MTP3, seqs 6, chunk 4096 | x1 median 37.7 vs 35.2 without; prose 27.0 vs 23.3 |
| 25 | Bluey+Asusi TP2 SPEED + DRAFT_VOCAB 65536 | 0.70 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 3 | 1,951,516 | n/a | 7.44x @262K | SPEED default + our reduced-vocab draft (TP-aware slice + all_reduce) | x1 median 53.2 vs 53.7 (flat), prose 40.8 vs 37.2 (+10%); x6 agg 105.0 vs 97.9 (+7%), TTFT under load +0.1 s; peak x1 68.1 vs 63.7; prefill 1,623/2,691/2,701 |
| 26 | Spark4 TP1 **STAGED + DRAFT_VOCAB 65536, MTP3** (new single-Spark default) | 0.80 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 3 | 1,027,392 | n/a | 3.92x @262K | table on disk, staged gather, reduced-vocab draft, seqs 6, chunk 4096, capture 4..24 | x1 median 43.9 (+35% vs shipped 32.5), prose 29.0; x2 33.4/43.9, x4 26.4/47.1, x6 21.7/68.8 |
| 27 | Reddie TP1 STAGED + DRAFT_VOCAB 65536, MTP4 | 0.80 | 262,144 | fp8_e4m3 | FULL_DECODE_ONLY, no compile | 4 | 1,033,305 | n/a | 3.94x @262K | same, MTP4, capture 5..30 | x1 median 43.6, prose 26.7; x2 32.5/44.0, x4 23.9/44.6, x6 21.5/64.1; prefill 1,280/1,755/1,766; ceiling not run (power cut 9:12 PM); behind MTP3 at every load, not adopted |
