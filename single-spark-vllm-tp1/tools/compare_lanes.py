#!/usr/bin/env python3
"""compare_lanes.py <lane_a> <lane_b> [--prefill-a f] [--prefill-b f]
Per-metric comparison of two harness lanes (categories_<lane>_off_c{1,2,4,6}.json): single-stream per category,
per-stream and aggregate by load, TTFT, and cold prefill rates. Prints B vs A with signed % deltas for a tweet/README."""
import json, os, re, statistics, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(HERE, "results")
a, b = sys.argv[1], sys.argv[2]
pa = sys.argv[sys.argv.index("--prefill-a")+1] if "--prefill-a" in sys.argv else None
pb = sys.argv[sys.argv.index("--prefill-b")+1] if "--prefill-b" in sys.argv else None
CATS = ["prose", "coding", "reasoning", "json", "html", "narrative", "summary", "format"]
def load(lane, c):
    p = os.path.join(RES, f"categories_{lane}_off_c{c}.json"); return json.load(open(p)) if os.path.exists(p) else None
def med(d, cat, key="decode_tok_s"):
    v = [i[key] for i in d["items"] if (cat is None or i["category"] == cat) and i.get(key) is not None]; return statistics.median(v) if v else None
def pct(x, y): return f"{(y/x-1)*100:+.0f}%" if x and y else "n/a"
print(f"{'metric':34s} {a[:18]:>18s} {b[:18]:>18s}   delta")
for c in (1, 2, 4, 6):
    da, db = load(a, c), load(b, c)
    if not (da and db): continue
    if c == 1:
        for cat in CATS:
            x, y = med(da, cat), med(db, cat)
            if x and y: print(f"x1 {cat:31s} {x:15.1f} tok/s {y:15.1f} tok/s   {pct(x,y)}")
        x, y = med(da, None), med(db, None); print(f"x1 {'median all prompts':31s} {x:15.1f} tok/s {y:15.1f} tok/s   {pct(x,y)}")
    x, y = med(da, None), med(db, None); print(f"x{c} {'per stream (all prompts)':30s} {x:15.1f} tok/s {y:15.1f} tok/s   {pct(x,y)}")
    x, y = med(da, "prose"), med(db, "prose"); print(f"x{c} {'prose per stream':30s} {x:15.1f} tok/s {y:15.1f} tok/s   {pct(x,y)}")
    x, y = da["overall"].get("agg_tok_s_med") or (da["overall"]["decode_med_tok_s"] if c == 1 else None), db["overall"].get("agg_tok_s_med") or (db["overall"]["decode_med_tok_s"] if c == 1 else None)
    print(f"x{c} {'aggregate':30s} {x:15.1f} tok/s {y:15.1f} tok/s   {pct(x,y)}")
    x, y = da["overall"]["ttft_med_s"], db["overall"]["ttft_med_s"]; print(f"x{c} {'TTFT (lower is better)':30s} {x*1000:15.0f} ms    {y*1000:15.0f} ms      {pct(x,y)}")
def prefill(f):
    out = {}
    if f and os.path.exists(f):
        for line in open(f):
            m = re.search(r"prompt_tokens=(\d+) ttft=([\d.]+)s", line)
            if m: n, t = int(m.group(1)), float(m.group(2)); out[round(n/1000)] = n/t
    return out
fa, fb = prefill(pa), prefill(pb)
for k in sorted(set(fa) & set(fb)): print(f"prefill {k:>3d}K tokens{'':18s} {fa[k]:13,.0f} tok/s {fb[k]:13,.0f} tok/s   {pct(fa[k], fb[k])}")
