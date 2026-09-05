#!/usr/bin/env python3
"""itl_probe.py <base_url> <model> <concurrency> [--tokens 300]
Streams N concurrent prose requests, records per-token arrival gaps (inter-token latency) and TTFT,
prints p50/p90/p99 ITL across all tokens of all streams plus per-stream decode tok/s. Stdlib only."""
import json, sys, time, threading, urllib.request, statistics
base, model, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
ntok = int(sys.argv[sys.argv.index("--tokens")+1]) if "--tokens" in sys.argv else 300
PROMPTS = ["Explain what tensor parallelism is to a smart high-school student, in about 250 words.",
           "Write a 250-word product description for a compact AI workstation with 128 GB of unified memory.",
           "In about 250 words, argue for or against benchmarking language models with synthetic prompts.",
           "Write a 250-word scene in which an engineer discovers why a server room went silent at 3 AM.",
           "In about 250 words, explain why time to first token matters more than tokens per second for a chat assistant.",
           "Write a 250-word letter from a lighthouse keeper to the town council about the failing lamp.",
           "In about 250 words, describe how a memory-mapped lookup table works to a junior developer.",
           "Write a 250-word review of a fictional sneaker called the Meridian Runner."]
gaps, ttfts, rates, toks, steps = [], [], [], [], []; lock = threading.Lock()
def worker(i):
    p = PROMPTS[i % len(PROMPTS)]
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": ntok, "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); last = None; n = 0; first = None
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        j = json.loads(d)
        if j.get("usage"):
            with lock: toks.append(j["usage"]["completion_tokens"])
        for c in j.get("choices", []):
            if c.get("delta", {}).get("content"):
                now = time.time(); n += 1
                if first is None: first = now - t0
                elif last is not None:
                    with lock: gaps.append(now - last)
                last = now
    with lock:
        steps.append(n)
        if first is not None: ttfts.append(first)
        if n > 1 and last and first is not None: rates.append((n - 1) / (last - (t0 + first)))
ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
[t.start() for t in ths]; [t.join() for t in ths]
gaps.sort()
def pct(p): return gaps[min(len(gaps) - 1, int(p / 100 * len(gaps)))] * 1000
print(f"concurrency {conc} | streams {len(rates)} | tokens {len(gaps)+len(rates)} | TTFT med {statistics.median(ttfts)*1000:.0f} ms | per-stream decode med {statistics.median(rates):.1f} tok/s | aggregate ~{sum(rates):.1f} tok/s")
tps = (sum(toks) / sum(steps)) if toks and steps else 1.0
print(f"decode STEP latency (one SSE chunk = one MTP step, {tps:.2f} tokens per step on average): p50 {pct(50):.0f} ms | p90 {pct(90):.0f} ms | p99 {pct(99):.0f} ms | max {gaps[-1]*1000:.0f} ms")
print(f"effective per-token latency (step / tokens-per-step): p50 {pct(50)/tps:.0f} ms | p99 {pct(99)/tps:.0f} ms | per-stream tokens/s med ~{statistics.median(rates)*tps:.1f} | aggregate ~{sum(rates)*tps:.1f}")
