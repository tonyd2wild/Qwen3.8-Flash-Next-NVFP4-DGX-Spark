#!/usr/bin/env python3
"""ab_mtp.py <label_a> <base_a> <label_b> <base_b> [--model qwen3.8-flash-next]
Same prompt set on two endpoints, single stream, thinking off; prints decode tok/s per prompt and medians.
Use with the vLLM 'SpecDecoding metrics' log lines for acceptance."""
import json, sys, time, urllib.request, statistics
a_label, a_base, b_label, b_base = sys.argv[1:5]
model = sys.argv[sys.argv.index("--model")+1] if "--model" in sys.argv else "qwen3.8-flash-next"
PROMPTS = [
    ("prose", "Explain what tensor parallelism is to a smart high-school student, in 120 to 180 words.", 260),
    ("prose", "Write a product description of 80 to 120 words for a compact AI workstation with 128 GB of unified memory.", 200),
    ("narrative", "Write a 150-word scene in which an engineer discovers why a server room went silent at 3 AM.", 260),
    ("code", "Write a Python function that merges overlapping intervals. Code only.", 200),
    ("code", "Write a Python function roman_to_int(s) converting a Roman numeral string to an integer. Code only.", 220),
    ("json", "Return a JSON object describing three fictional books with title, author, year, and genre. JSON only.", 200),
    ("math", "A train leaves at 3:15 PM and arrives at 6:40 PM the same day. How many minutes is the trip? Explain in two sentences then give the number.", 120),
    ("count", "Count from 1 to 60 separated by commas, then say done.", 260),
]
def run(base, cat, prompt, n):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t; c = d["usage"]["completion_tokens"]
    return c, dt, c / dt
for base in (a_base, b_base):
    run(base, "warm", "Say hello.", 8)
rows = []
for cat, p, n in PROMPTS:
    ca, ta, ra = run(a_base, cat, p, n)
    cb, tb, rb = run(b_base, cat, p, n)
    rows.append((cat, ra, rb))
    print(f"{cat:10s} {a_label}: {ra:5.1f} tok/s ({ca} tok)   {b_label}: {rb:5.1f} tok/s ({cb} tok)   ratio {rb/ra:4.2f}", flush=True)
for cat in sorted(set(r[0] for r in rows)):
    xa = statistics.median([r[1] for r in rows if r[0] == cat]); xb = statistics.median([r[2] for r in rows if r[0] == cat])
    print(f"median {cat:10s} {a_label}: {xa:5.1f}   {b_label}: {xb:5.1f}   ratio {xb/xa:4.2f}")
print(f"overall median {a_label}: {statistics.median([r[1] for r in rows]):.1f}   {b_label}: {statistics.median([r[2] for r in rows]):.1f}")
