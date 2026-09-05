#!/usr/bin/env python3
"""summarize.py [lane ...]  -> results/summary.md (and prints it)
Reads results/categories_<lane>_off_c{1,4,16}.json written by bench_categories.py and prints one comparison table
per lane: prose and coding decode tok/s (C1 medians), overall C1 decode and TTFT medians, C4 and C16 aggregate
tok/s, and the auto-graded score. Lanes default to every lane found in results/."""
import json, os, re, sys, glob

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

def load(lane, c):
    p = os.path.join(RES, f"categories_{lane}_off_c{c}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def fmt(v, nd=1):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))

lanes = sys.argv[1:] or sorted({re.sub(r"^categories_(.+)_off_c\d+\.json$", r"\1", os.path.basename(p))
                                for p in glob.glob(os.path.join(RES, "categories_*_off_c1.json"))})
rows = []
for lane in lanes:
    c1, c4, c16 = load(lane, 1), load(lane, 4), load(lane, 16)
    if not c1:
        rows.append((lane, "no C1 file")); continue
    s = c1.get("summary", {}); o = c1.get("overall", {})
    rows.append({
        "lane": lane, "model": c1.get("model"),
        "prose": (s.get("prose") or {}).get("decode_med_tok_s"), "coding": (s.get("coding") or {}).get("decode_med_tok_s"),
        "c1_decode": o.get("decode_med_tok_s"), "c1_ttft": o.get("ttft_med_s"), "c1_score": o.get("auto_score"),
        "c4_agg": (c4 or {}).get("overall", {}).get("agg_tok_s_med"), "c4_score": (c4 or {}).get("overall", {}).get("auto_score"),
        "c16_agg": (c16 or {}).get("overall", {}).get("agg_tok_s_med"), "c16_ttft": (c16 or {}).get("overall", {}).get("ttft_med_s"),
        "c16_score": (c16 or {}).get("overall", {}).get("auto_score"),
        "errors": sum(1 for it in c1.get("items", []) if str(it.get("finish", "")).startswith("error")),
    })

lines = ["| lane | prose tok/s | code tok/s | C1 decode med | C1 TTFT med | C1 score | C4 aggregate | C16 aggregate | C16 TTFT | C16 score |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if isinstance(r, tuple):
        lines.append(f"| {r[0]} | {r[1]} | | | | | | | | |"); continue
    lines.append("| %s | %s | %s | %s | %s s | %s | %s | %s | %s s | %s |" % (
        r["lane"], fmt(r["prose"]), fmt(r["coding"]), fmt(r["c1_decode"]), fmt(r["c1_ttft"], 2), fmt(r["c1_score"], 3),
        fmt(r["c4_agg"]), fmt(r["c16_agg"]), fmt(r["c16_ttft"], 2), fmt(r["c16_score"], 3)))
    if r["errors"]:
        lines.append(f"| | | | | | | | | | {r['errors']} C1 prompts errored |")
per_cat = ["", "Per-category C1 decode tok/s (median) and auto score:", "",
           "| lane | " + " | ".join(["coding", "reasoning", "json", "html", "prose", "narrative", "summary", "format"]) + " |",
           "|---|" + "---|" * 8]
for lane in lanes:
    c1 = load(lane, 1)
    if not c1: continue
    s = c1.get("summary", {})
    per_cat.append("| %s | " % lane + " | ".join(
        "%s (%s)" % (fmt((s.get(c) or {}).get("decode_med_tok_s")), fmt((s.get(c) or {}).get("auto_score"), 2))
        for c in ["coding", "reasoning", "json", "html", "prose", "narrative", "summary", "format"]) + " |")
out = "\n".join(lines + per_cat) + "\n"
open(os.path.join(RES, "summary.md"), "w").write(out)
print(out)
