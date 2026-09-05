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
