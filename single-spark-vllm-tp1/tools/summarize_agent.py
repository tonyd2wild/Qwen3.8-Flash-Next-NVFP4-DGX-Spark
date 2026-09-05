#!/usr/bin/env python3
"""summarize_agent.py [lane ...] -> prints a table of agent-loop results per lane (results/agent_loop_<lane>_<tag>.json)."""
import json, os, glob, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
lanes = sys.argv[1:] or sorted({re.sub(r"^agent_loop_(.+)_(short|doc8k)\.json$", r"\1", os.path.basename(p))
                                for p in glob.glob(os.path.join(RES, "agent_loop_*_short.json"))})

def load(lane, tag):
    p = os.path.join(RES, f"agent_loop_{lane}_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d: return d[k]
    return None

print("| lane | short: TTFT med | p90 | first | last | decode | doc8k: TTFT med | p90 | first | last | decode |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for lane in lanes:
    row = [lane]
    for tag in ("short", "doc8k"):
        d = load(lane, tag)
        if not d:
            row += ["n/a"] * 5; continue
        med, p90, first, last, dec = (d.get(k) for k in ("ttft_med_s", "ttft_p90_s", "ttft_first_s", "ttft_last_s", "decode_med_tok_s"))
        row += [f"{med:.2f} s" if med else "n/a", f"{p90:.2f} s" if p90 else "n/a", f"{first:.2f} s" if first else "n/a",
                f"{last:.2f} s" if last else "n/a", f"{dec:.1f}" if isinstance(dec, (int, float)) else "n/a"]
    print("| " + " | ".join(row) + " |")
