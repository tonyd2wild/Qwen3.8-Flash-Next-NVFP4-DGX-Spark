#!/usr/bin/env python3
"""bench_sweep.py <base_url> <served_model> <label> [--rounds 3] [--max-c 6] [--out results/sweep_<label>.json]

Full concurrency sweep c1..cN for one lane. Non-stream, temp 0, thinking OFF. Per concurrency level:
  decode aggregate tok/s   = sum(completion_tokens) / round wall          (median over rounds)
  decode per-stream tok/s  = each request's completion_tokens / its wall  (median over all requests)
  wall-to-wall latency (s) = each request's end-to-end wall               (median, p90)
  TTFT (s) at that concurrency = c concurrent ~1.5K-token-prompt / 8-token requests, median wall
  prefill tok/s at c1      = prompt_tokens / wall (median of the c1 TTFT requests)
Warm-up first (2x c1 + 1x cN). Isolate the lane before running; verify from the head's access log after.
"""
import sys, json, time, statistics, argparse, urllib.request, concurrent.futures

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("label")
ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--max-c", type=int, default=6)
ap.add_argument("--levels", default="", help="explicit concurrency ladder, e.g. 1,2,4,8,16,32,48 (overrides --max-c)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
URL = a.base.rstrip("/") + "/v1/chat/completions"
GEN = "List the numbers from 1 to 300 separated by commas. Output only the numbers, nothing else, no commentary."
LONG = ("Summarize the following in one sentence.\n\n" +
        ("The DGX Spark is a compact AI workstation built on the GB10 superchip with unified memory. " * 80))

def call(prompt, max_tokens):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t
    u = r.get("usage", {})
    return u.get("completion_tokens", 0), u.get("prompt_tokens", 0), dt

def concurrent_calls(c, prompt, max_tokens):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        res = [f.result() for f in [ex.submit(call, prompt, max_tokens) for _ in range(c)]]
    return res, time.time() - t0

def p90(xs):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]

print(f"[{a.label}] warm-up ...", flush=True)
for _ in range(2): call(GEN, 320)
concurrent_calls(a.max_c, GEN, 320)

rows = []
for c in ([int(x) for x in a.levels.split(',')] if a.levels else range(1, a.max_c + 1)):
    aggs, pers, walls = [], [], []
    for _ in range(a.rounds):
        res, wall = concurrent_calls(c, GEN, 320)
        aggs.append(sum(r[0] for r in res) / wall)
        pers += [r[0] / r[2] for r in res if r[2]]
        walls += [r[2] for r in res]
    pre, _ = concurrent_calls(c, LONG, 8)
    ttft = statistics.median([r[2] for r in pre])
    prefill = statistics.median([r[1] / r[2] for r in pre if r[2]]) if c == 1 else None
    row = {"c": c, "agg_tok_s": round(statistics.median(aggs), 1), "per_stream_tok_s": round(statistics.median(pers), 1),
           "w2w_med_s": round(statistics.median(walls), 2), "w2w_p90_s": round(p90(walls), 2),
           "ttft_med_s": round(ttft, 2), **({"prefill_tok_s": round(prefill)} if prefill else {})}
    rows.append(row)
    print(f"  c{c}: agg {row['agg_tok_s']:6.1f} tok/s | per-stream {row['per_stream_tok_s']:5.1f} | "
          f"w2w med {row['w2w_med_s']:5.2f}s p90 {row['w2w_p90_s']:5.2f}s | TTFT {row['ttft_med_s']:5.2f}s"
          + (f" | prefill {row['prefill_tok_s']} tok/s" if prefill else ""), flush=True)

out = {"label": a.label, "model": a.model, "base": a.base, "rounds": a.rounds, "gen_tokens": 300, "rows": rows,
       "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
path = a.out or f"results/sweep_{a.label.split()[0].lower()}.json"
with open(path, "w") as f: json.dump(out, f, indent=1)
print(f"JSON -> {path}")
