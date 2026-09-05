#!/usr/bin/env python3
"""stress_prefill.py <base_url> <model> <prompt_tokens> [--repeat N]
Long-prefill stress for the gmu ladder: builds a ~prompt_tokens prompt, asks a needle question,
reports TTFT/total time, tokens, finish reason, and whether the answer is right. Stdlib only."""
import json, sys, time, urllib.request
base, model, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
rep = int(sys.argv[sys.argv.index("--repeat")+1]) if "--repeat" in sys.argv else 1
filler = "The mountain road curves past the old sawmill, where the river bends toward the eastern valley and the pines lean into the wind. "
words_per_tok = 0.75
needle = "The secret code word is PELICAN-7."
body_words = int(ntok * words_per_tok)
chunk = filler.split()
text = []
while len(text) < body_words:
    text.extend(chunk)
mid = len(text) // 2
prompt = " ".join(text[:mid]) + " " + needle + " " + " ".join(text[mid:]) + "\n\nWhat is the secret code word mentioned above? Answer with the code word only."
for i in range(rep):
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); ttft = None; out = []; usage = None; finish = None
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        j = json.loads(d)
        if j.get("usage"): usage = j["usage"]
        for c in j.get("choices", []):
            delta = c.get("delta", {}).get("content")
            if delta:
                if ttft is None: ttft = time.time() - t0
                out.append(delta)
            if c.get("finish_reason"): finish = c["finish_reason"]
    total = time.time() - t0
    ans = "".join(out).strip()
    ok = "PELICAN-7" in ans.upper()
    ptoks = (usage or {}).get("prompt_tokens")
    print(f"run {i+1}: prompt_tokens={ptoks} ttft={ttft:.1f}s total={total:.1f}s prefill~{(ptoks or 0)/(ttft or 1):.0f} tok/s finish={finish} answer={ans[:40]!r} {'OK' if ok else 'WRONG'}", flush=True)
