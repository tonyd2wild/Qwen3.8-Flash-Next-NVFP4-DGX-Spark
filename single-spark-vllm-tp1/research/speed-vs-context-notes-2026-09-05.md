# Draft: SPEED vs CONTEXT profiles (fill numbers after the resident-table lanes report)

## Pick a profile (for people and for agents reading this on their behalf)

Every lane in this repo runs the same official NVIDIA NVFP4 checkpoint with the same flags (MTP4, FP8 KV, CUDA graphs, gpu-memory-utilization 0.80, 262K context). The one choice that matters is where the 47.7 GB PLE n-gram table lives:

| Profile | Where the table lives | What you get | What it costs | Launch |
|---|---|---|---|---|
| CONTEXT | On disk, rows read per step (our patch) | Biggest KV pool: 995K (1 Spark), 5.87M (TP2), 9.09M (TP4) | Slower per reply: the lookup is a host round trip every step | `PLE_MODE=mmap` (default) |
| SPEED | In unified memory, each rank holds its slice | Faster per reply: (x1 median: TBD tok/s, prose TBD) | Smaller KV pool: (TP2: TBD tokens, TP4: TBD) | `PLE_MODE=none` (+ `GRAPHS=full` if it held) |

Rule of thumb for the agent doing the deploy: ask one question. "Do you need the longest possible context and the most parallel sessions, or the fastest replies?" Long context / many sessions = CONTEXT. Fast replies for a few users = SPEED. One Spark is CONTEXT only: the whole table does not fit next to the weights on a single box.

If the deployer has two Sparks and asks for both: SPEED on TP2 still leaves (TBD) tokens of KV, which is (TBD) full 262K contexts; say that number out loud before choosing.

## Notes to fill
- Lane B (Reddie+Spark4) PLE resident + piecewise: KV pool ___, boot ___ s, x1 median ___, prose ___, x6 ___.
- Lane A (Bluey+Asusi) PLE resident + FULL_AND_PIECEWISE: booted? ___ KV pool ___, x1 median ___.
- Winner → harness, chart (4 lanes: 1 Spark, TP2 context, TP2 speed, TP4 ...), README tables, tweet.

## Measured 2026-09-05 late afternoon (TP2, Reddie+Spark4 = lane B, Bluey+Asusi = lane A; all gmu 0.70 unless noted)

| Boot | Table | Compile | MTP | seqs | chunk | KV pool | x1 median | prose | TTFT | x4 per-stream / agg |
|---|---|---|---|---|---|---|---|---|---|---|
| Shipped CONTEXT TP2 (0.80) | disk (ours) | on, piecewise | 4 | 8 | default | 5,874,061 | 35.8 | 24.1 | 250 | 27.8 / 43.1 |
| RAM stock, no compile | resident (stock loader) | OFF, FULL_DECODE_ONLY | 4 | 8 | default | 1,898,635 | 28.4 | 18.6 | 380 | 18.9 / 34.0 |
| RAM stock, no compile, RECIPE | resident (stock loader) | OFF, FULL_DECODE_ONLY | 3 | 6 | 4096 | 1,970,051 | **53.7** | **37.2** | **180** | **37.4 / 68.1** |
| RAM ours + compile + RECIPE | resident (our patch slice) | on, piecewise | 3 | 6 | 4096 | (pending) | (pending) | | | |
| RAM stock, no compile, chunk 4096 only | resident (stock loader) | OFF | 4 | 8 | 4096 | (pending) | (pending) | | | |

Compile-off alone: −20% single stream, −19 to −25% aggregate, prefill halved. Recipe settings on top: +50% single stream vs shipped, +42 to +58% aggregate. Which of MTP3 / seqs 6 / chunk 4096 carries it: isolation boots pending.
