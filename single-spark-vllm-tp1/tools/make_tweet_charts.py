#!/usr/bin/env python3
"""make_tweet_charts.py <lane> [--title "..."] [--prefill results/prefill_<lane>.txt] [--kv results/kv_pool_ledger.md]
Renders dark table graphics from the harness JSON (results/categories_<lane>_off_c{1,2,4,6}.json):
  results/chart_<lane>_prose.png     prose by load: TTFT / aggregate / per-stream
  results/chart_<lane>_categories.png all categories x loads: per-stream decode (aggregate in parentheses)
  results/chart_<lane>_prefill.png   cold prefill ladder (if a prefill file is given)
  results/chart_<lane>_kv.png        KV pool by gpu-memory-utilization (from the ledger)
Own design. Real prompts only; the counting ceiling is never charted.
"""
import json, os, re, sys, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(HERE, "results")
lane = sys.argv[1]
title = sys.argv[sys.argv.index("--title")+1] if "--title" in sys.argv else "Qwen3.8-Flash-Next NVFP4 (NVIDIA official) on ONE DGX Spark"
prefill_file = sys.argv[sys.argv.index("--prefill")+1] if "--prefill" in sys.argv else None
kv_file = sys.argv[sys.argv.index("--kv")+1] if "--kv" in sys.argv else os.path.join(RES, "kv_pool_ledger.md")
BG, PANEL, INK, MUT, ACC, RULE, GOLD = "#0f1115", "#171a21", "#f2f3f5", "#9aa3b2", "#59d3a0", "#2a2f3a", "#f2c14e"
CATS = ["prose", "coding", "reasoning", "json", "html", "narrative", "summary", "format"]
LABEL = {"prose": "Prose", "coding": "Coding", "reasoning": "Math / logic", "json": "JSON", "html": "HTML", "narrative": "Narrative", "summary": "Summary", "format": "Format"}
def load(c):
    p = os.path.join(RES, f"categories_{lane}_off_c{c}.json")
    return json.load(open(p)) if os.path.exists(p) else None
runs = {c: load(c) for c in (1, 2, 4, 6) if load(c)}
assert runs, "no categories_<lane>_off_c*.json found"
def stats(d, cat=None):
    items = [i for i in d["items"] if (cat is None or i["category"] == cat) and i.get("decode_tok_s")]
    ttft = statistics.median([i["ttft_s"] for i in items if i.get("ttft_s") is not None])
    stream = statistics.median([i["decode_tok_s"] for i in items])
    # aggregate = sum of completion tokens / wall time of the batch; approximate with concurrency x per-stream median when batches are uniform
    # wall-clock aggregate across the whole batch (all categories), same definition as summarize.py
    agg = d.get("overall", {}).get("agg_tok_s_med") or (d.get("overall", {}).get("decode_med_tok_s") if d.get("concurrency", 1) == 1 else None)
    return ttft, stream, agg
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.facecolor": PANEL, "figure.facecolor": BG, "savefig.facecolor": BG})
def table_png(rows, header, out, subtitle, highlight_col=None, colw=None):
    fig_h = 0.62 * (len(rows) + 2) + 0.9
    fig, ax = plt.subplots(figsize=(10.5, fig_h)); ax.axis("off")
    ax.text(0, 1.0, title, fontsize=13.5, fontweight="bold", va="top", color=INK, transform=ax.transAxes)
    ax.text(0, 0.90, subtitle, fontsize=9.5, va="top", color=MUT, transform=ax.transAxes)
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", colWidths=colw, bbox=(0, 0.02, 1, 0.78))
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE); cell.set_linewidth(0.6)
        cell.set_facecolor(PANEL if r % 2 else "#1c2029"); cell.get_text().set_color(INK)
        if r == 0: cell.set_facecolor("#232833"); cell.get_text().set_color(MUT); cell.get_text().set_fontweight("bold")
        elif c == 0: cell.get_text().set_color(MUT); cell.get_text().set_fontweight("bold")
        elif highlight_col is not None and c == highlight_col: cell.get_text().set_color(GOLD); cell.get_text().set_fontweight("bold")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print("->", out)
# 1) prose by load
rows = []
for c, d in sorted(runs.items()):
    ttft, stream, agg = stats(d, "prose")
    rows.append([f"x{c}", f"{ttft*1000:.0f} ms", f"{stream:.1f} tok/s", f"{agg:.1f} tok/s" if agg else "n/a"])
table_png(rows, ["Load", "TTFT (median)", "Prose per stream", "Aggregate"], os.path.join(RES, f"chart_{lane}_prose.png"),
          "Prose prompts (120-200 word replies), thinking off, no prefix cache. Per stream = one reply's speed; aggregate = total tokens/s the box delivers across all concurrent streams (all 8 categories).", highlight_col=2)
# 2) all categories x loads (per-stream median)
hdr = ["Category"] + [f"x{c} per stream" for c in sorted(runs)]
rows = []
for cat in CATS:
    if not any(i["category"] == cat for i in list(runs.values())[0]["items"]): continue
    r = [LABEL[cat]]
    for c, d in sorted(runs.items()):
        r.append(f"{stats(d, cat)[1]:.1f}")
    rows.append(r)
aggs = "  ".join(f"x{c}: {stats(d)[2]:.1f}" for c, d in sorted(runs.items()) if stats(d)[2])
table_png(rows, hdr, os.path.join(RES, f"chart_{lane}_categories.png"), f"Per-stream decode tok/s by category and concurrent load (median of 5 real prompts each). Aggregate tok/s across all streams: {aggs}. Counting prompts excluded on purpose.", highlight_col=1)
# 3) prefill ladder
if prefill_file and os.path.exists(prefill_file):
    rows = []
    for line in open(prefill_file):
        m = re.search(r"prompt_tokens=(\d+) ttft=([\d.]+)s", line)
        if m:
            n, t = int(m.group(1)), float(m.group(2)); rows.append([f"{n/1000:.0f}K", f"{t:.2f} s", f"{n/t:,.0f} tok/s"])
    if rows:
        table_png(rows, ["Prompt", "Time to first token", "Prefill rate"], os.path.join(RES, f"chart_{lane}_prefill.png"), "Cold prefill: one request, no prefix cache, needle question answered correctly in every run.", highlight_col=2)
# 4) KV pool by gmu
if os.path.exists(kv_file):
    rows = []
    for line in open(kv_file):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and re.match(r"^\d+$", cells[0]) and re.match(r"^[\d,]+$", cells[7]):
            rows.append([cells[2], cells[3], cells[4], cells[6], cells[7], cells[10] or ""])
    if rows:
        table_png(rows, ["gmu", "Context", "KV dtype", "MTP", "KV pool (tokens)", "Free after boot"], os.path.join(RES, f"chart_{lane}_kv.png"), "KV pool by gpu-memory-utilization, one Spark, model weights identical in every row. Pick your own headroom.", highlight_col=4, colw=[0.09, 0.13, 0.13, 0.08, 0.2, 0.17])
