import concurrent.futures
import json
import time
import urllib.request

URL = "http://100.92.77.51:8000/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files matching a pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
]

PROMPTS = [
    "Summarize the pros and cons of tensor parallelism vs pipeline parallelism in about 300 words. Do not call tools for this.",
    "Explain how speculative decoding with a built-in MTP layer works, about 300 words. No tools needed.",
    "Use the glob tool to find all python files under /src, then explain what you would do next.",
    "Write a 250-word explanation of NVFP4 quantization tradeoffs. No tools needed.",
    "Use the read_file tool to read /etc/hostname, then describe what you'd check next on a GPU server.",
    "Describe the InfiniBand vs RoCE tradeoffs for a 2-node tensor-parallel deployment in 300 words. No tools.",
]


def one(i):
    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "tools": TOOLS,
        "max_tokens": 400,
        "temperature": 0.7,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    dur = time.perf_counter() - start
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "") + ""
    bangs = content.count("!!!!")
    usage = data.get("usage", {})
    return {
        "req": i,
        "secs": round(dur, 1),
        "finish": data["choices"][0].get("finish_reason"),
        "tool_call": bool(msg.get("tool_calls")),
        "completion_tokens": usage.get("completion_tokens"),
        "bang_runs": bangs,
        "content_head": content[:80].replace("\n", " "),
    }


results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(one, i) for i in range(6)]
    for f in concurrent.futures.as_completed(futs):
        try:
            r = f.result()
        except Exception as e:
            r = {"error": str(e)}
        results.append(r)
        print(json.dumps(r), flush=True)

bad = [r for r in results if r.get("bang_runs", 0) > 0 or "error" in r]
print("VERDICT:", "FAIL - degenerate output or errors" if bad else "PASS - all 6 concurrent streams clean")
