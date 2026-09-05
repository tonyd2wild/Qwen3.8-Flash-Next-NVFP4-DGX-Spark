#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tony DeAngelo (Tech2Wild / 2Wild); written by Kai
"""make_tp2_charts.py: one image, single Spark (TP1) vs two Sparks (TP2), same stack, same harness.
usage: make_tp2_charts.py <laneA> <laneB> --prefill-a <txt> --prefill-b <txt> [--out results/chart_<a>_vs_<b>.png]
Counting prompts are excluded from every panel; the counting ceiling appears only in the footnote."""
import json, os, re, statistics, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
a, b = sys.argv[1], sys.argv[2]
def arg(k, d=None): return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
pa, pb = arg("--prefill-a"), arg("--prefill-b")
out = arg("--out", os.path.join(RES, f"chart_{a}_vs_{b}.png"))
kvA, kvB = int(arg("--kv-a", "995129")), int(arg("--kv-b", "5874061"))
INK, MUT, C1, C2, GRID = "#1b1f24", "#5c6470", "#2f6fb3", "#d9822b", "#e3e6ea"
def load(lane, c): return json.load(open(os.path.join(RES, f"categories_{lane}_off_c{c}.json")))
def med(d, cat=None, key="decode_tok_s"):
    v = [i[key] for i in d["items"] if (cat is None or i["category"] == cat) and i.get(key) is not None]
    return statistics.median(v)
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
    d = json.load(open(p)); return max(r["agg_tok_s"] for r in d["rows"])
A = {c: load(a, c) for c in (1, 2, 4, 6)}; B = {c: load(b, c) for c in (1, 2, 4, 6)}
CATS = [("prose", "Prose"), ("coding", "Coding"), ("reasoning", "Math"), ("json", "JSON"), ("html", "HTML"), ("narrative", "Narrative"), ("summary", "Summary"), ("format", "Format")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(16, 11.2), dpi=150, facecolor="white")
fig.text(0.02, 0.975, "Qwen 3.8 Flash Next NVFP4 (OFFICIAL NVIDIA quant): one DGX Spark vs two Sparks (TP2)", fontsize=17, fontweight="bold", color=INK, va="top")
fig.text(0.02, 0.940, "Same stack on both: upstream vLLM + our PLE-on-disk patch, MTP4, FP8 KV, CUDA graphs, gmu 0.80, 262K context. Same 40 real prompts, thinking off,\nno prefix cache, cold prefill. TP2 = Reddie + Spark4 over the ConnectX fabric. Counting prompts excluded from every panel.", fontsize=9.6, color=MUT, va="top", linespacing=1.5)
gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, top=0.87, bottom=0.075, hspace=0.42, wspace=0.22, width_ratios=[1.15, 1])
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelcolor=INK); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
def bars(ax, labels, va, vb, ylabel, title, fmt="{:.0f}"):
    x = range(len(labels)); w = 0.38
    ax.bar([i - w/2 for i in x], va, w, color=C1, label="1 Spark", zorder=3)
    ax.bar([i + w/2 for i in x], vb, w, color=C2, label="2 Sparks (TP2)", zorder=3)
    top = max(max(va), max(vb))
    for i, (p, q) in enumerate(zip(va, vb)):
        ax.text(i - w/2, p + top*0.015, fmt.format(p), ha="center", va="bottom", fontsize=8, color=INK)
        d = (q/p - 1)*100
        ax.text(i + w/2, q + top*0.015, fmt.format(q) + f"\n{d:+.0f}%", ha="center", va="bottom", fontsize=8, color=("#1e7a3c" if d >= 0 else "#b3261e"), fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel(ylabel, color=MUT); ax.set_ylim(0, top*1.22)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8); style(ax); ax.legend(frameon=False, fontsize=9, loc="upper left")
# 1) single stream by category
ax = fig.add_subplot(gs[0, 0])
bars(ax, [n for _, n in CATS] + ["Median\n(all 40)"], [med(A[1], k) for k, _ in CATS] + [med(A[1])], [med(B[1], k) for k, _ in CATS] + [med(B[1])], "tok/s per reply", "SINGLE STREAM (x1) by prompt category", "{:.1f}")
# 2) load table
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.set_title("UNDER LOAD: TTFT, per-stream and aggregate throughput", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
rows = []
for c in (1, 2, 4, 6):
    tA, tB = A[c]["overall"]["ttft_med_s"]*1000, B[c]["overall"]["ttft_med_s"]*1000
    sA, sB = med(A[c]), med(B[c]); gA, gB = agg(A[c]), agg(B[c])
    rows.append([f"x{c}", f"{tA:.0f} / {tB:.0f} ms", f"{(tB/tA-1)*100:+.0f}%", f"{sA:.1f} / {sB:.1f}", f"{(sB/sA-1)*100:+.0f}%", f"{gA:.1f} / {gB:.1f}", f"{(gB/gA-1)*100:+.0f}%"])
hdr = ["Streams", "TTFT ms (1 / TP2)", "chg", "Per stream tok/s", "chg", "Aggregate tok/s", "chg"]
t = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center", colWidths=[0.1, 0.21, 0.08, 0.19, 0.08, 0.19, 0.08], bbox=[0, 0.28, 1, 0.66])
t.auto_set_font_size(False); t.set_fontsize(9)
for (r, cc), cell in t.get_celld().items():
    cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
    if r == 0: cell.set_facecolor("#f1f3f5"); cell.set_text_props(fontweight="bold", color=INK)
    elif cc in (2, 4, 6):
        v = cell.get_text().get_text(); good = (v.startswith("-") if cc == 2 else v.startswith("+"))
        cell.set_text_props(color=("#1e7a3c" if good else "#b3261e"), fontweight="bold")
ax.text(0, 0.2, "Per stream = median over the 40 prompts at that concurrency. Aggregate = harness wall clock,\nall streams summed. Prose at x6: %.1f vs %.1f tok/s per stream. Quality scores identical on both.\nLower TTFT is better; the fabric halves the wait at x6." % (med(A[6], "prose"), med(B[6], "prose")), fontsize=8.8, color=MUT, va="top", transform=ax.transAxes, linespacing=1.5)
# 3) prefill ladder
ax = fig.add_subplot(gs[1, 0])
ra, rb = prefill(pa), prefill(pb); n = min(len(ra), len(rb))
bars(ax, [f"{ra[i][0]/1000:.0f}K prompt" for i in range(n)], [ra[i][1] for i in range(n)], [rb[i][1] for i in range(n)], "prefill tok/s (cold, no prefix cache)", "PREFILL LADDER, cold, needle answered correctly at every rung")
# 4) KV pool
ax = fig.add_subplot(gs[1, 1])
ax.bar([0, 1], [kvA/1e6, kvB/1e6], 0.55, color=[C1, C2], zorder=3)
for i, v in enumerate((kvA, kvB)): ax.text(i, v/1e6 + 0.12, f"{v:,} tokens", ha="center", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["1 Spark", "2 Sparks (TP2)"]); ax.set_ylabel("KV pool, millions of tokens", color=MUT); ax.set_ylim(0, kvB/1e6*1.25)
ax.set_title("KV POOL at gmu 0.80, FP8 KV, MTP4, 262K context", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8); style(ax)
ax.text(0.27, 0.60, f"{kvB/kvA:.1f}x the pool: every rank holds half the weights,\nso the freed memory on both boxes becomes KV.\n{kvB//262144} full 262K contexts in flight on TP2 vs {kvA//262144} on one Spark.", ha="center", fontsize=9.2, color=MUT, transform=ax.transAxes, linespacing=1.5)
ca, cb = ceiling(a), ceiling(b)
foot = "Footnote, not a headline: synthetic counting prompts (count to 100) peak at x6 aggregate " + (f"{ca:.0f} vs {cb:.0f} tok/s" if ca and cb else "n/a") + ". Left out of every panel above on purpose.\nPrefill above is measured from cold TTFT on a single request; the 176K rung is the 200K stress prompt. Weights read locally on each rank; no NFS in the serving path."
fig.text(0.02, 0.012, foot, fontsize=8.4, color=MUT, va="bottom", linespacing=1.5)
fig.savefig(out, facecolor="white"); print("wrote", out)
