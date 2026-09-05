#!/usr/bin/env python3
"""bench_categories.py <base_url> <served_model> <lane> [--thinking off|on] [--concurrency N] [--only cat,cat] [--max-tokens N]

Real-prompt benchmark: 40 prompts, 8 categories x 5 (coding, reasoning, json, html, prose, narrative, summary, format),
identical for every lane. Streaming, so each prompt yields TTFT (first content token), decode tok/s
((completion_tokens-1)/(t_end-t_first)) and wall. Quality is auto-graded wherever there is a checkable answer:
  coding    hidden tests exec'd against the model's code (subprocess, 10 s timeout) -> tests passed / total
  reasoning final ANSWER: line vs expected
  json      parsed JSON compared to expected (numbers numeric, strings exact)
  html      structure checks (required tags / ids / attributes / counts)
  format    rule checks (counts, ranges, exact strings)
  prose / narrative / summary  constraint checks only (word ranges, required words, bullet counts); ranking is done
            separately by judge_pairwise.py (blind, both orders).
Writes results/categories_<lane>_<thinking>_c<N>.json. With --concurrency N the 40 prompts run in batches of N
(different categories in the same batch) and the file also carries per-batch aggregate tok/s.
"""
import sys, json, time, re, argparse, urllib.request, subprocess, tempfile, os, statistics, concurrent.futures
from html.parser import HTMLParser
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("lane")
ap.add_argument("--thinking", default="off", choices=["off", "on"]); ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--only", default=""); ap.add_argument("--max-tokens", type=int, default=0)
ap.add_argument("--tag", default="", help="suffix for the output file, e.g. run2 (so repeats do not overwrite run 1)")
a = ap.parse_args(); URL = a.base.rstrip("/") + "/v1/chat/completions"
MAXTOK = a.max_tokens or (3000 if a.thinking == "on" else 900)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def doc(path, cap=5500):
    p = os.path.join(REPO, path)
    return open(p, encoding="utf-8").read()[:cap] if os.path.exists(p) else "(document missing)"

# ------------------------------------------------------------------ prompts
P = []  # (id, category, prompt, grader-spec)
# coding: (tests appended to the model's code; exec'd)
P += [("code1", "coding", "Write a Python function merge_intervals(intervals) that merges overlapping [start, end] intervals and returns the merged list sorted by start. Return only one ```python code block.",
       {"tests": "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,6],[1,2]])==[[1,2],[5,6]]"}),
      ("code2", "coding", "Write a Python function parse_duration(s) that converts strings like '1h30m15s', '45s', '2h', '3m', '1h1s' into total seconds (int). Units may be omitted but always appear in h, m, s order. Return only one ```python code block.",
       {"tests": "assert parse_duration('1h30m15s')==5415\nassert parse_duration('45s')==45\nassert parse_duration('2h')==7200\nassert parse_duration('3m')==180\nassert parse_duration('1h1s')==3601"}),
      ("code3", "coding", "Write a Python function top_k_frequent(words, k) returning the k most frequent words, most frequent first; ties broken alphabetically. Return only one ```python code block.",
       {"tests": "assert top_k_frequent(['b','a','c','a','b','a'],2)==['a','b']\nassert top_k_frequent(['x','y'],1)==['x']\nassert top_k_frequent(['z','z','y','y','x'],2)==['y','z']\nassert top_k_frequent([],3)==[]"}),
      ("code4", "coding", "This binary search has a bug that can make it loop forever. Return the corrected function only, as one ```python code block, same name and signature.\n\ndef binary_search(a, target):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] == target:\n            return mid\n        if a[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return -1",
       {"tests": "assert binary_search([1,3,5,7,9],7)==3\nassert binary_search([1,3,5,7,9],2)==-1\nassert binary_search([],1)==-1\nassert binary_search([1],1)==0\nassert binary_search([1,2],2)==1\nassert binary_search([1,3,5,7,9],9)==4"}),
      ("code5", "coding", "Write a Python function roman_to_int(s) converting a Roman numeral string to an integer. Return only one ```python code block.",
       {"tests": "assert roman_to_int('XIV')==14\nassert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('IX')==9\nassert roman_to_int('LVIII')==58\nassert roman_to_int('III')==3"})]
SUF = "\n\nGive your final answer on the last line exactly as: ANSWER: <answer>"
P += [("reason1", "reasoning", "Ana is twice as old as Ben. In 6 years Ana will be 1.5 times Ben's age. How old is Ben now?" + SUF, {"answer": "6"}),
      ("reason2", "reasoning", "A fair coin is flipped 4 times. What is the probability of exactly 2 heads? Give a fraction in lowest terms." + SUF, {"answer": "3/8", "alt": ["0.375"]}),
      ("reason3", "reasoning", "Five runners finish a race. Dee finishes before Eve but after Cal. Bo finishes last. Al finishes before Cal. Who finishes third?" + SUF, {"answer": "Dee"}),
      ("reason4", "reasoning", "Two trains 300 km apart travel toward each other at 70 km/h and 80 km/h. A bird flying at 120 km/h goes back and forth between them until they meet. How many km does the bird fly?" + SUF, {"answer": "240"}),
      ("reason5", "reasoning", "What is the remainder when 7^100 is divided by 5?" + SUF, {"answer": "1"})]
JS = "\n\nOutput only the JSON, no prose, no code fences."
P += [("json1", "json", "Extract into a JSON object with keys invoice_id (integer), customer (string), issued (ISO date string), amount (number), due_days (integer):\n\n\"Invoice 4471 was issued to Marta Ruiz on 2026-03-14 for $1,250.50, due in 30 days.\"" + JS,
       {"json": {"invoice_id": 4471, "customer": "Marta Ruiz", "issued": "2026-03-14", "amount": 1250.5, "due_days": 30}}),
      ("json2", "json", "Convert this CSV to a JSON array of objects with numeric qty and price:\n\nname,qty,price\napple,3,0.5\npear,2,0.75" + JS,
       {"json": [{"name": "apple", "qty": 3, "price": 0.5}, {"name": "pear", "qty": 2, "price": 0.75}]}),
      ("json3", "json", "Fix this invalid JSON and output only the corrected JSON:\n\n{'user': 'sam', 'tags': ['a', 'b',], 'active': True}" + JS,
       {"json": {"user": "sam", "tags": ["a", "b"], "active": True}}),
      ("json4", "json", "Return a JSON object mapping each city to the arithmetic mean of its temperatures:\n\n{\"Oslo\": [2, 4, 6], \"Cairo\": [30, 32], \"Lima\": [18, 19, 20, 21]}" + JS,
       {"json": {"Oslo": 4, "Cairo": 31, "Lima": 19.5}}),
      ("json5", "json", "Produce a JSON object with exactly these keys and values: id = integer 7, name = string \"Widget\", price = number 19.99, in_stock = boolean true, tags = array of strings [\"tools\", \"home\"]." + JS,
       {"json": {"id": 7, "name": "Widget", "price": 19.99, "in_stock": True, "tags": ["tools", "home"]}})]
HS = "\n\nOutput only the HTML."
P += [("html1", "html", "Write a complete HTML5 page: a <header> containing an <h1> that reads Spark Bench and a <nav> with three links (Home, Results, Method), then a <main> containing a <section id=\"summary\"> with one paragraph." + HS,
       {"html": {"doctype": True, "tags": {"header": 1, "h1": 1, "nav": 1, "main": 1, "section": 1, "p": 1}, "min_tags": {"a": 3}, "ids": ["summary"], "text": ["Spark Bench", "Home", "Results", "Method"]}}),
      ("html2", "html", "Write an HTML <table> with a <thead> row of Lane, Decode tok/s, TTFT s and a <tbody> with two rows: NVFP4, 64.0, 1.54 and EXL3, 61.5, 0.52." + HS,
       {"html": {"tags": {"table": 1, "thead": 1, "tbody": 1}, "min_tags": {"th": 3, "td": 6}, "text": ["NVFP4", "64.0", "1.54", "EXL3", "61.5", "0.52"]}}),
      ("html3", "html", "Write an HTML <form> with: a required email input with id=\"email\", a <select id=\"lane\"> with the options NVFP4 and EXL3, a <textarea id=\"notes\">, and a submit <button>." + HS,
       {"html": {"tags": {"form": 1, "select": 1, "textarea": 1, "button": 1}, "min_tags": {"option": 2}, "ids": ["email", "lane", "notes"], "attrs": [("input", "type", "email"), ("input", "required", None)]}}),
      ("html4", "html", "Write an accessible site navigation: a skip link <a href=\"#main\" class=\"skip\">Skip to content</a>, then <nav aria-label=\"Main\"> containing a <ul> of four <li> items each with an <a>." + HS,
       {"html": {"tags": {"nav": 1, "ul": 1}, "min_tags": {"li": 4, "a": 5}, "attrs": [("nav", "aria-label", "Main"), ("a", "class", "skip"), ("a", "href", "#main")]}}),
      ("html5", "html", "Write an HTML card component with an inline <style> block: a <div class=\"card\"> containing an <img> with an alt attribute, an <h2>, a <p>, and an <a class=\"btn\">." + HS,
       {"html": {"tags": {"style": 1, "img": 1, "h2": 1, "p": 1}, "min_tags": {"div": 1, "a": 1}, "attrs": [("div", "class", "card"), ("a", "class", "btn"), ("img", "alt", None)]}})]
P += [("prose1", "prose", "Explain what tensor parallelism is to a smart high-school student, in 120 to 180 words.", {"words": (120, 180)}),
      ("prose2", "prose", "Write a product description of 80 to 120 words for a compact AI workstation with 128 GB of unified memory. Do not use the words revolutionary, game-changing, or unleash.", {"words": (80, 120), "banned": ["revolutionary", "game-changing", "unleash"]}),
      ("prose3", "prose", "Write a professional email under 150 words politely declining a meeting request and proposing two specific alternative times next week.", {"words": (30, 150)}),
      ("prose4", "prose", "In 150 to 200 words, argue for or against benchmarking language models with synthetic prompts. Take one side clearly.", {"words": (150, 200)}),
      ("prose5", "prose", "In 100 to 150 words, explain in plain language why time to first token matters more than tokens per second for a chat assistant. Use exactly one concrete analogy.", {"words": (100, 150)})]
P += [("story1", "narrative", "Write a 150 to 250 word short story in first person that includes the words lantern, ledger and thunder, contains at least two lines of dialogue, and ends with a twist.", {"words": (150, 250), "must": ["lantern", "ledger", "thunder"], "quotes": 2}),
      ("story2", "narrative", "Write a 120 to 200 word fable with a talking fox. State the moral in the final sentence, starting with the word Moral:", {"words": (120, 200), "must": ["fox", "Moral:"]}),
      ("story3", "narrative", "Write a 200 to 300 word scene: two engineers arguing at 2 a.m. about a benchmark result. Mostly dialogue; no narration longer than one sentence at a time.", {"words": (200, 300), "quotes": 6}),
      ("story4", "narrative", "Write a 100 to 150 word micro-story told entirely in second person, present tense, set in a data center during a power outage.", {"words": (100, 150), "must": ["you"]}),
      ("story5", "narrative", "Write a 150 to 250 word story for children aged 6 to 8 about a robot learning to wait. Exactly three paragraphs.", {"words": (150, 250), "paragraphs": 3})]
P += [("sum1", "summary", "Summarize the following notes in exactly 3 bullet points, each one sentence.\n\n---\n" + doc("bench-docs/NOTES.md"), {"bullets": 3}),
      ("sum2", "summary", "Summarize the following report in one paragraph of at most 80 words.\n\n---\n" + doc("bench-docs/REPORT.md"), {"words": (20, 80), "paragraphs": 1}),
      ("sum3", "summary", "In exactly two sentences, summarize what the following text says about quality.\n\n---\n" + doc("bench-docs/x_article_paste.txt", 7000), {"sentences": 2}),
      ("sum4", "summary", "Summarize the following README in 3 bullets labeled What, How, Credits.\n\n---\n" + doc("bench-docs/README.md"), {"bullets": 3, "must": ["What", "How", "Credits"]}),
      ("sum5", "summary", "Turn the following into a single tweet of at most 280 characters. Output only the tweet.\n\n---\n" + doc("bench-docs/tweet.md", 3000), {"chars": 280})]
P += [("fmt1", "format", "List exactly 7 prime numbers greater than 50, one per line, ascending. Output nothing else.", {"primes": 7}),
      ("fmt2", "format", "Output a markdown table with the columns Lane | Context | KV pool and two rows: NVFP4 with 262,144 and 295,230; EXL3 with 1,048,576 and 1,396,551. Output nothing else.", {"table_rows": 2, "text": ["262,144", "295,230", "1,048,576", "1,396,551"]}),
      ("fmt3", "format", "Reply with the sentence 'The quick brown fox jumps over the lazy dog' with the words in reverse order, all lowercase, no punctuation. Output only the result.", {"exact": "dog lazy the over jumps fox brown quick the"}),
      ("fmt4", "format", "In at most 20 words: why should a unified-memory machine drop its page cache before loading a large model?", {"words": (1, 20)}),
      ("fmt5", "format", "Write a haiku about a GPU fan: three lines, 5-7-5 syllables. Output only the three lines.", {"lines": 3})]
if a.only: P = [p for p in P if p[1] in a.only.split(",")]

# ------------------------------------------------------------------ graders
def words(t): return len(re.findall(r"\b[\w'’-]+\b", t))
def strip_fences(t):
    m = re.search(r"```(?:json|html|python)?\s*(.*?)```", t, re.S); return m.group(1) if m else t
def first_json(t):
    t = strip_fences(t).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener); j = t.rfind(closer)
        if i != -1 and j > i:
            try: return json.loads(t[i:j + 1])
            except Exception: pass
    return None
def jeq(x, y):
    if isinstance(x, bool) or isinstance(y, bool): return x is y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)): return abs(x - y) < 1e-6
    if isinstance(x, dict) and isinstance(y, dict): return x.keys() == y.keys() and all(jeq(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list): return len(x) == len(y) and all(jeq(p, q) for p, q in zip(x, y))
    return isinstance(x, str) and isinstance(y, str) and x.strip() == y.strip()
class HP(HTMLParser):
    def __init__(s): super().__init__(); s.tags = {}; s.attrs = []; s.stack = []; s.bad = 0; s.text = []
    VOID = {"img", "input", "br", "hr", "meta", "link", "source"}
    def handle_starttag(s, tag, attrs):
        s.tags[tag] = s.tags.get(tag, 0) + 1; s.attrs.append((tag, dict(attrs)))
        if tag not in s.VOID: s.stack.append(tag)
    def handle_endtag(s, tag):
        if tag in s.VOID: return
        if s.stack and s.stack[-1] == tag: s.stack.pop()
        elif tag in s.stack:
            while s.stack and s.stack[-1] != tag: s.stack.pop(); s.bad += 1
            s.stack.pop()
        else: s.bad += 1
    def handle_data(s, d): s.text.append(d)
def grade_html(spec, out):
    h = HP(); raw = strip_fences(out); h.feed(raw); checks = []
    if spec.get("doctype"): checks.append(("doctype", "<!doctype html>" in raw.lower()))
    for t, n in spec.get("tags", {}).items(): checks.append((f"{t}x{n}", h.tags.get(t, 0) == n))
    for t, n in spec.get("min_tags", {}).items(): checks.append((f"{t}>={n}", h.tags.get(t, 0) >= n))
    ids = {at.get("id") for _, at in h.attrs}
    for i in spec.get("ids", []): checks.append((f"id={i}", i in ids))
    for tag, k, v in spec.get("attrs", []):
        checks.append((f"{tag}[{k}{'=' + v if v else ''}]", any(t == tag and k in at and (v is None or at.get(k) == v) for t, at in h.attrs)))
    txt = " ".join(h.text)
    for s in spec.get("text", []): checks.append((f"text:{s}", s in txt))
    checks.append(("balanced", h.bad == 0 and not h.stack or all(t in ("html", "body", "head") for t in h.stack)))
    return sum(ok for _, ok in checks) / len(checks), [c for c, ok in checks if not ok]
def grade_code(spec, out):
    code = strip_fences(out)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + "\n".join(f"try:\n    {t}\n    print('PASS')\nexcept Exception as e:\n    print('FAIL', type(e).__name__)" for t in spec["tests"].splitlines())); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10); lines = r.stdout.splitlines()
        passed = sum(l == "PASS" for l in lines); total = len(spec["tests"].splitlines())
        return passed / total, ([] if passed == total else [f"{passed}/{total} tests" + (" · " + r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else "")])
    except subprocess.TimeoutExpired: return 0.0, ["timeout (infinite loop?)"]
    finally: os.unlink(path)
def is_prime(n): return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
def grade(cat, spec, out):
    """-> (score 0..1 or None when only a judge can rank it, list of failed checks)"""
    fails = []
    if cat == "coding": return grade_code(spec, out)
    if cat == "reasoning":
        m = re.findall(r"ANSWER:\s*(.+)", out); got = (m[-1] if m else out.strip().splitlines()[-1] if out.strip() else "").strip().strip("*` .")
        got = re.sub(r"\s*(km|kilometers?|degrees?|°|years?( old)?|dollars?|\$)\s*$", "", got, flags=re.I).strip().strip("*` .")  # units are not the answer
        ok = got.replace(" ", "").lower() == spec["answer"].lower() or got in spec.get("alt", [])
        try: ok = ok or abs(float(got) - float(spec["answer"])) < 1e-6
        except ValueError: pass
        return (1.0 if ok else 0.0), ([] if ok else [f"got {got[:30]!r}"])
    if cat == "json":
        got = first_json(out); ok = got is not None and jeq(got, spec["json"]); return (1.0 if ok else 0.0), ([] if ok else ["json mismatch" if got is not None else "no parsable json"])
    if cat == "html": return grade_html(spec["html"], out)
    # constraint checks (prose / narrative / summary / format)
    checks = []
    if "words" in spec: lo, hi = spec["words"]; w = words(out); checks.append((f"words {w} in {lo}-{hi}", lo <= w <= hi))
    if "chars" in spec: checks.append((f"chars {len(out.strip())}<={spec['chars']}", len(out.strip()) <= spec["chars"]))
    for m_ in spec.get("must", []): checks.append((f"has {m_}", m_.lower() in out.lower()))
    for b in spec.get("banned", []): checks.append((f"no {b}", b.lower() not in out.lower()))
    if "quotes" in spec: checks.append((f"dialogue>={spec['quotes']}", len(re.findall(r"[\"“][^\"”]{2,}[\"”]", out)) >= spec["quotes"]))
    if "paragraphs" in spec: checks.append((f"paragraphs={spec['paragraphs']}", len([p for p in re.split(r"\n\s*\n", out.strip()) if p.strip()]) == spec["paragraphs"]))
    if "bullets" in spec: checks.append((f"bullets={spec['bullets']}", len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", out, re.M)) == spec["bullets"]))
    if "sentences" in spec: checks.append((f"sentences={spec['sentences']}", len(re.findall(r"[^.!?]+[.!?]", out.strip())) == spec["sentences"]))
    if "lines" in spec: checks.append((f"lines={spec['lines']}", len([l for l in out.strip().splitlines() if l.strip()]) == spec["lines"]))
    if "exact" in spec: checks.append(("exact", re.sub(r"\s+", " ", out.strip().strip("\"'`.").lower()) == spec["exact"]))
    if "primes" in spec:
        nums = [int(x) for x in re.findall(r"\b\d+\b", out)]
        checks.append((f"7 primes>50 ascending", len(nums) == spec["primes"] and all(is_prime(n) and n > 50 for n in nums) and nums == sorted(nums)))
    if "table_rows" in spec:
        rows = [l for l in out.splitlines() if l.strip().startswith("|") and not re.match(r"^\s*\|[\s:-]+\|", l)]
        checks.append((f"table rows={spec['table_rows']}+header", len(rows) == spec["table_rows"] + 1))
    for s in spec.get("text", []): checks.append((f"text:{s}", s in out))
    if not checks: return None, []
    return sum(ok for _, ok in checks) / len(checks), [c for c, ok in checks if not ok]

# ------------------------------------------------------------------ streaming call
def call(prompt):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": MAXTOK,
                       "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": a.thinking == "on", "thinking": a.thinking == "on"}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; txt = []; rc = 0; usage = {}; fin = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            j = json.loads(data)
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {}); c = d.get("content"); r_ = d.get("reasoning") or d.get("reasoning_content")
                if r_: rc += len(r_)
                if r_ and t_first is None: t_first = time.time()   # first visible token, reasoning counts
                if c:
                    if t_first is None: t_first = time.time()
                    txt.append(c)
                if ch.get("finish_reason"): fin = ch["finish_reason"]
    t_end = time.time(); out = "".join(txt); ct = usage.get("completion_tokens", 0)
    dec = (ct - 1) / (t_end - t_first) if (t_first and t_end > t_first and ct > 1) else None
    return {"ttft_s": round((t_first or t_end) - t0, 2), "wall_s": round(t_end - t0, 2), "completion_tokens": ct, "prompt_tokens": usage.get("prompt_tokens", 0),
            "decode_tok_s": round(dec, 1) if dec else None, "reasoning_chars": rc, "finish": fin, "output": out}

def run_one(item):
    pid, cat, prompt, spec = item
    try:
        r = call(prompt)
    except Exception as exc:  # an HTTP 4xx/5xx or a dropped stream must not abort the other 39 prompts
        r = {"ttft_s": None, "wall_s": None, "completion_tokens": 0, "prompt_tokens": 0, "decode_tok_s": None,
             "reasoning_chars": 0, "finish": "error: %s" % str(exc)[:120], "output": ""}
    score, fails = grade(cat, spec, r["output"])
    r.update({"id": pid, "category": cat, "prompt": prompt, "score": score, "fails": fails})
    print(f"  [{a.lane}/{a.thinking}/c{a.concurrency}] {pid:8} {cat:10} score={('%.2f' % score) if score is not None else ' n/a'}  ttft {(r['ttft_s'] if r['ttft_s'] is not None else -1):5.2f}s  decode {r['decode_tok_s'] or 0:5.1f} tok/s  {r['completion_tokens']:4} tok  {r['finish']}" + (f"  fails: {'; '.join(fails)[:70]}" if fails else ""), flush=True)
    return r

results, batches = [], []
if a.lane.endswith(".json") and os.path.exists(a.lane):
    # --- regrade mode: bench_categories.py x x results/categories_<lane>_<mode>_c<N>.json  (re-scores stored outputs, no requests)
    old = json.load(open(a.lane)); a.lane, a.thinking, a.concurrency = old["lane"], old["thinking"], old["concurrency"]; SPEC = {p[0]: (p[1], p[3]) for p in P}
    for r in old["items"]:
        cat, spec = SPEC[r["id"]]; sc, fl = grade(cat, spec, r["output"]); r["score"], r["fails"] = sc, fl; results.append(r)
    batches = old.get("batches", []); print(f"regraded {len(results)} items from {sys.argv[3]}")
elif a.concurrency <= 1:
    for it in P: results.append(run_one(it))
else:
    # interleave categories so every batch mixes prompt types
    order = sorted(P, key=lambda p: (P.index(p) % 5, P.index(p)))
    for i in range(0, len(order), a.concurrency):
        batch = order[i:i + a.concurrency]; t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex: rs = list(ex.map(run_one, batch))
        wall = time.time() - t0; toks = sum(r["completion_tokens"] for r in rs)
        batches.append({"ids": [r["id"] for r in rs], "wall_s": round(wall, 2), "agg_tok_s": round(toks / wall, 1), "ttft_med_s": round(statistics.median(r["ttft_s"] for r in rs), 2)}); results += rs
        print(f"    batch {i // a.concurrency + 1}: agg {toks / wall:.1f} tok/s, wall {wall:.1f}s", flush=True)
by_cat = {}
for r in results:
    b = by_cat.setdefault(r["category"], {"n": 0, "scored": 0, "score_sum": 0.0, "ttft": [], "decode": [], "tokens": []})
    b["n"] += 1; b["ttft"].append(r["ttft_s"]); b["tokens"].append(r["completion_tokens"])
    if r["decode_tok_s"]: b["decode"].append(r["decode_tok_s"])
    if r["score"] is not None: b["scored"] += 1; b["score_sum"] += r["score"]
summary = {c: {"n": b["n"], "auto_score": (round(b["score_sum"] / b["scored"], 3) if b["scored"] else None), "ttft_med_s": round(statistics.median(b["ttft"]), 2),
               "decode_med_tok_s": (round(statistics.median(b["decode"]), 1) if b["decode"] else None), "tokens_med": int(statistics.median(b["tokens"]))} for c, b in by_cat.items()}
out = {"lane": a.lane, "model": a.model, "thinking": a.thinking, "concurrency": a.concurrency, "n": len(results), "summary": summary, "batches": batches,
       "overall": {"auto_score": round(sum(r["score"] for r in results if r["score"] is not None) / max(1, sum(r["score"] is not None for r in results)), 3),
                   "ttft_med_s": round(statistics.median(r["ttft_s"] for r in results), 2), "decode_med_tok_s": round(statistics.median(r["decode_tok_s"] for r in results if r["decode_tok_s"]), 1),
                   "agg_tok_s_med": (round(statistics.median(b["agg_tok_s"] for b in batches), 1) if batches else None)},
       "items": results, "ts": time.strftime("%Y-%m-%d %H:%M")}
path = f"results/categories_{a.lane}_{a.thinking}_c{a.concurrency}{('_' + a.tag) if a.tag else ''}.json"; json.dump(out, open(path, "w"), indent=1)
print(f"[{a.lane}/{a.thinking}/c{a.concurrency}] done: auto {out['overall']['auto_score']}, ttft med {out['overall']['ttft_med_s']}s, decode med {out['overall']['decode_med_tok_s']} tok/s" + (f", mixed agg med {out['overall']['agg_tok_s_med']} tok/s" if batches else "") + f" -> {path}")
for c, s in summary.items(): print(f"    {c:10} auto={s['auto_score']}  ttft={s['ttft_med_s']}s  decode={s['decode_med_tok_s']}  tokens={s['tokens_med']}")
