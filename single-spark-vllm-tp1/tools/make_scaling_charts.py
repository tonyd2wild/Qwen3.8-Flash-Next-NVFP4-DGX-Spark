#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tony DeAngelo (Tech2Wild / 2Wild); written by Kai
"""make_scaling_charts.py: one image, N lanes side by side (one Spark, TP2, TP4 ...), same stack, same harness.
usage: make_scaling_charts.py --lane <name>:<label>:<kv_tokens>:<prefill_txt> [--lane ...] [--out png] [--title "..."]
Percent labels are relative to the FIRST lane. Counting prompts excluded from every panel (footnote only)."""
import json, os, re, statistics, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
lanes = []
args = sys.argv[1:]
out, title = None, "Qwen 3.8 Flash Next NVFP4 (OFFICIAL NVIDIA quant): one DGX Spark vs two (TP2) vs four (TP4)"
i = 0
while i < len(args):
    if args[i] == "--lane":
        name, label, kv, pf = args[i+1].split(":"); lanes.append((name, label, int(kv), pf)); i += 2
    elif args[i] == "--out": out = args[i+1]; i += 2
    elif args[i] == "--title": title = args[i+1]; i += 2
    else: i += 1
assert lanes, "need --lane"
out = out or os.path.join(RES, "chart_scaling_" + "_vs_".join(l[0] for l in lanes) + ".png")
INK, MUT, GRID = "#1b1f24", "#5c6470", "#e3e6ea"; COLS = ["#2f6fb3", "#d9822b", "#2e8b57", "#8e44ad"]
def load(lane, c): return json.load(open(os.path.join(RES, f"categories_{lane}_off_c{c}.json")))
def med(d, cat=None, key="decode_tok_s"):
    v = [it[key] for it in d["items"] if (cat is None or it["category"] == cat) and it.get(key) is not None]; return statistics.median(v)
def agg(d): return d["overall"].get("agg_tok_s_med") or d["overall"]["decode_med_tok_s"]
def prefill(path):
    rows = []
    for line in open(path):
        m = re.search(r"prompt_tokens=(\d+).*prefill~(\d+)", line)
        if m: rows.append((int(m.group(1)), int(m.group(2))))
    return rows
def ceiling(lane):
    p = os.path.join(RES, f"sweep_{lane}.json")
    if not os.path.exists(p): return None
    return max(r["agg_tok_s"] for r in json.load(open(p))["rows"])
D = [{c: load(l[0], c) for c in (1, 2, 4, 6)} for l in lanes]
CATS = [("prose", "Prose"), ("coding", "Coding"), ("reasoning", "Math"), ("json", "JSON"), ("html", "HTML"), ("narrative", "Narrative"), ("summary", "Summary"), ("format", "Format")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(16, 11.4), dpi=150, facecolor="white")
fig.text(0.02, 0.975, title, fontsize=16.5, fontweight="bold", color=INK, va="top")
fig.text(0.02, 0.940, "Same stack on every lane: upstream vLLM + our PLE-on-disk patch, MTP4, FP8 KV, CUDA graphs, gpu-memory-utilization 0.80, 262K context. Same 40 real prompts,\nthinking off, no prefix cache, cold prefill. Multi-Spark lanes run over the ConnectX fabric; TP4 adds expert parallel. Counting prompts excluded from every panel. Percentages vs one Spark.", fontsize=9.4, color=MUT, va="top", linespacing=1.5)
gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, top=0.865, bottom=0.075, hspace=0.42, wspace=0.22, width_ratios=[1.15, 1])
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelcolor=INK); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
def bars(ax, labels, series, ylabel, ttl, fmt="{:.0f}"):
    n = len(series); w = 0.8 / n; x = range(len(labels)); top = max(max(s) for s in series)
    for j, s in enumerate(series):
        xs = [k - 0.4 + w*(j+0.5) for k in x]
        ax.bar(xs, s, w, color=COLS[j], label=lanes[j][1], zorder=3)
        for k, v in enumerate(s):
            if j == 0: ax.text(xs[k], v + top*0.015, fmt.format(v), ha="center", va="bottom", fontsize=6.4, color=INK)
            else:
                d = (v/series[0][k]-1)*100
                ax.text(xs[k], v + top*0.015, fmt.format(v) + f"\n{d:+.0f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=("#1e7a3c" if d >= 0 else "#b3261e"))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel(ylabel, color=MUT); ax.set_ylim(0, top*1.24)
    ax.set_title(ttl, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8); style(ax); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=len(series))
ax = fig.add_subplot(gs[0, 0])
bars(ax, [n for _, n in CATS] + ["Median\n(all 40)"], [[med(d[1], k) for k, _ in CATS] + [med(d[1])] for d in D], "tok/s per reply", "SINGLE STREAM (x1) by prompt category", "{:.1f}")
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.set_title("UNDER LOAD: TTFT, per-stream and aggregate throughput", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
rows = []
for c in (1, 2, 4, 6):
    t = [d[c]["overall"]["ttft_med_s"]*1000 for d in D]; s = [med(d[c]) for d in D]; g = [agg(d[c]) for d in D]
    rows.append([f"x{c}", " / ".join(f"{v:.0f}" for v in t), " / ".join(f"{v:.1f}" for v in s), " / ".join(f"{v:.1f}" for v in g),
                 " / ".join(f"{(v/t[0]-1)*100:+.0f}%" for v in t[1:]), " / ".join(f"{(v/s[0]-1)*100:+.0f}%" for v in s[1:]), " / ".join(f"{(v/g[0]-1)*100:+.0f}%" for v in g[1:])])
order = " / ".join(l[1] for l in lanes); chg = " / ".join(l[1] for l in lanes[1:])
hdr = ["Streams", "TTFT ms", "Per stream tok/s", "Aggregate tok/s", "TTFT chg", "Per stream chg", "Aggregate chg"]
t = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center", colWidths=[0.075, 0.14, 0.19, 0.19, 0.12, 0.145, 0.14], bbox=[0, 0.22, 1, 0.72])
t.auto_set_font_size(False); t.set_fontsize(7.9)
for (r, cc), cell in t.get_celld().items():
    cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
    if r == 0: cell.set_facecolor("#f1f3f5"); cell.set_text_props(fontweight="bold", color=INK, fontsize=7.6)
    elif cc >= 4: cell.set_text_props(fontweight="bold", color=INK)
ax.text(0, 0.17, f"Values listed as {order}. Change columns vs one Spark, listed as {chg}.\nPer stream = median over the 40 prompts at that concurrency. Aggregate = harness wall clock,\nall streams summed. Lower TTFT is better. Quality scores identical on every lane.", fontsize=8.4, color=MUT, va="top", transform=ax.transAxes, linespacing=1.5)
ax = fig.add_subplot(gs[1, 0])
P = [prefill(l[3]) for l in lanes]; n = min(len(p) for p in P)
bars(ax, [f"{P[0][i][0]/1000:.0f}K prompt" for i in range(n)], [[p[i][1] for i in range(n)] for p in P], "prefill tok/s (cold, no prefix cache)", "PREFILL LADDER, cold, needle answered correctly at every rung")
ax = fig.add_subplot(gs[1, 1])
kv = [l[2] for l in lanes]; ax.bar(range(len(kv)), [v/1e6 for v in kv], 0.55, color=COLS[:len(kv)], zorder=3)
for i, v in enumerate(kv): ax.text(i, v/1e6 + max(kv)/1e6*0.02, f"{v:,}" + ("" if i else " tokens") + (f"\n{v/kv[0]:.1f}x" if i else ""), ha="center", fontsize=9.5, fontweight="bold", color=INK)
ax.set_xticks(range(len(kv))); ax.set_xticklabels([l[1] for l in lanes]); ax.set_ylabel("KV pool, millions of tokens", color=MUT); ax.set_ylim(0, max(kv)/1e6*1.28)
ax.set_title("KV POOL at gmu 0.80, FP8 KV, MTP4, 262K context", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8); style(ax)
ax.text(0.03, 0.95, "Each rank holds a slice of the weights,\nso the freed memory on every box becomes KV.\nFull 262K contexts in flight: " + " / ".join(str(v//262144) for v in kv) + ".", fontsize=8.6, color=MUT, va="top", transform=ax.transAxes, linespacing=1.5)
ce = [ceiling(l[0]) for l in lanes]
foot = "Footnote, not a headline: synthetic counting prompts (count to 100) peak at x6 aggregate " + " / ".join(f"{c:.0f}" if c else "n/a" for c in ce) + " tok/s. Left out of every panel above on purpose.\nPrefill is measured from cold TTFT on a single request; the 176K rung is the 200K stress prompt."
fig.text(0.02, 0.012, foot, fontsize=8.4, color=MUT, va="bottom", linespacing=1.5)
fig.savefig(out, facecolor="white"); print("wrote", out)
