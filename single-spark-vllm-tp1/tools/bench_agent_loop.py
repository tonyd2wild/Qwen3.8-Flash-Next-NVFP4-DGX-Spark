#!/usr/bin/env python3
"""bench_agent_loop.py <base_url> <served_model> <lane> [--turns 20] [--doc-tokens 0] [--tag short]

Simulates an agent conversation the way agents actually run: every turn re-sends the ENTIRE growing message history
(system prompt, every prior user turn and every prior assistant reply, verbatim) plus one new user instruction, and the
assistant's real reply is appended for the next turn. With --doc-tokens N the first user turn carries an N-token
document, so every later turn re-sends that document too (the long-context agent case). Streaming, so each turn reports
time to first token, decode tok/s, wall, and the prompt size the server counted. Thinking off, temperature 0,
max 200 tokens per reply. Writes results/agent_loop_<lane>_<tag>.json.
"""
import sys, json, time, argparse, urllib.request, statistics
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("lane")
ap.add_argument("--turns", type=int, default=20); ap.add_argument("--doc-tokens", type=int, default=0); ap.add_argument("--tag", default="short")
a = ap.parse_args(); URL = a.base.rstrip("/") + "/v1/chat/completions"
SYSTEM = ("You are an infrastructure engineer's assistant working through a deployment plan for a two-node GPU cluster. "
          "Keep every reply under 150 words, concrete, and consistent with everything said earlier in this conversation. "
          "Refer back to prior decisions by name when relevant.")
DOC_PARA = ("Node A hosts the head process and exports the model directory over NFS; node B mounts it read-only at boot. "
            "The fabric is a single RoCE rail per node; the second rail is unpopulated. Weights are 91 GiB per node after "
            "sharding, leaving roughly 20 GiB for KV cache and page cache combined. The watchdog restarts a container when "
            "free memory drops under 3 GiB. Clocks must be verified under load after every reboot. ")
TURNS = ["Summarize the constraints we are working with in three bullets.", "Propose a boot order for the two nodes and say why.",
         "What is the single biggest risk in that boot order? Name it and give a mitigation.", "Draft a one-line health check we can poll.",
         "We just saw the worker die with a memory error during warm-up. What do you check first, second, third?",
         "Write the exact shell command to drop page caches on a node.", "How would you prove the lane is isolated from other clients before a benchmark?",
         "List the metrics we should record per request, one per line.", "Which of those metrics is most sensitive to prefix caching, and why?",
         "Give me a fresh-prompt benchmark rule in one sentence I can put in a README.", "Now write a 5-step runbook for a full restart of both nodes.",
         "Which step in that runbook is most often skipped, in your experience, and what breaks when it is?", "Rewrite step 3 to be idempotent.",
         "What log line tells us warm-up is finished? Suggest a grep.", "We want to add a second lane on two more nodes. What changes in the plan?",
         "Name two things that must never be shared between the two lanes.", "Summarize every decision we have made so far as a numbered list.",
         "Which decision would you revisit first if the KV pool turned out to be half the size we assumed?", "Draft the commit message for this plan.",
         "Close out: three sentences on what we learned about measuring this system honestly."]
def doc(n_tokens): return ("Reference document for this session:\n\n" + DOC_PARA * max(1, int(n_tokens * 4.0 / len(DOC_PARA))) + "\n\n") if n_tokens else ""
def stream(messages):
    body = json.dumps({"model": a.model, "messages": messages, "temperature": 0, "max_tokens": 200, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; out = []; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if t_first is None: t_first = time.time()
                    out.append(c)
    t_end = time.time(); ct = usage.get("completion_tokens", 0)
    return {"ttft_s": round((t_first or t_end) - t0, 2), "wall_s": round(t_end - t0, 2), "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": ct,
            "decode_tok_s": (round((ct - 1) / (t_end - t_first), 1) if (t_first and ct > 1 and t_end > t_first) else None)}, "".join(out)
messages = [{"role": "system", "content": SYSTEM}]; turns = []; t_all = time.time()
for i, u in enumerate(TURNS[:a.turns]):
    content = (doc(a.doc_tokens) if i == 0 else "") + u
    messages.append({"role": "user", "content": content})
    m, reply = stream(messages); messages.append({"role": "assistant", "content": reply})
    m["turn"] = i + 1; turns.append(m)
    print(f"  [{a.lane}/{a.tag}] turn {i+1:2}: ctx {m['prompt_tokens']:>6} tok  TTFT {m['ttft_s']:5.2f}s  decode {m['decode_tok_s'] or 0:5.1f} tok/s  wall {m['wall_s']:5.1f}s", flush=True)
total = round(time.time() - t_all, 1)
out = {"lane": a.lane, "model": a.model, "tag": a.tag, "turns": len(turns), "doc_tokens": a.doc_tokens, "total_s": total,
       "ttft_med_s": round(statistics.median(t["ttft_s"] for t in turns), 2), "ttft_p90_s": round(sorted(t["ttft_s"] for t in turns)[int(0.9 * (len(turns) - 1))], 2),
       "ttft_first_s": turns[0]["ttft_s"], "ttft_last_s": turns[-1]["ttft_s"], "ctx_last": turns[-1]["prompt_tokens"],
       "decode_med_tok_s": round(statistics.median(t["decode_tok_s"] for t in turns if t["decode_tok_s"]), 1), "per_turn": turns, "ts": time.strftime("%Y-%m-%d %H:%M")}
json.dump(out, open(f"results/agent_loop_{a.lane}_{a.tag}.json", "w"), indent=1)
print(f"[{a.lane}/{a.tag}] {len(turns)} turns, final ctx {out['ctx_last']} tok: TTFT median {out['ttft_med_s']}s p90 {out['ttft_p90_s']}s (first {out['ttft_first_s']}s, last {out['ttft_last_s']}s), decode {out['decode_med_tok_s']} tok/s, total {total}s")
