# Qwen3.8-Flash-Next "!!!!" Output Bug — Diagnosis + Fix

Date: 2026-08-26
Endpoint: `http://100.92.77.51:8000/v1` (Bluey+Asusi TP2, SGLang, NVFP4)
Status: **FIX DEPLOYED — verification receipts at the bottom**

## The symptom

Mid-session, agentic requests (OMP) degrade into an endless stream of `!!!!!!!!` until `max_tokens`. Two incidents within ~1 hour, each wedging multiple concurrent sessions at once. Fresh simple requests kept working alongside the broken ones.

## Root cause

**Not** hardware, NaN, thermal, or the SM121 kernel patch. It is a day-0 SGLang bug, reported upstream the same day this model was released:

> [sgl-project/sglang#36537](https://github.com/sgl-project/sglang/issues/36537) — "Qwen3.8-Flash-Next thinking + qwen3_coder tool parser loops on token ID 0"

When **thinking mode + OpenAI `tools` + `--tool-call-parser qwen3_coder`** interact, the model emits token ID 0 in a deterministic loop. Token 0 in Qwen's vocab decodes as `!`. The upstream token trace matches ours exactly.

Why it looked like corruption: every OMP session sends tools, and the model thinks by default, so all sessions eventually tripped it, flatlining the speculative accept rate to 0.00 and filling all 6 request slots with zombie generations (clients had disconnected; the server kept generating to max_tokens).

Why it never happened this morning: Codex's first deploy was **missing** `--tool-call-parser qwen3_coder` — so instead of this loop, raw `<tool_call>` XML leaked into content (the original OMP complaint). Adding the parser fixed the XML leak and exposed this second day-0 bug. Two faces of the same immature stack.

## Reproduction notes (honest ones)

Minimal single-turn probes did **not** trigger the loop, even with thinking + tools + parser combined, with `reasoning_effort` set, or with tool results in history. The exact trigger needs more session state than a small probe carries (long OMP-shaped sessions reproduce it reliably; twice in one evening). Thinking-off requests were verified clean in every test. This matches the upstream issue's workaround guidance.

## The fix

Server-side default added to the launcher on BOTH nodes (`~/launch-qwen38fn-sglang-tp2.sh`):

```
--default-chat-template-kwargs '{"enable_thinking": false}'
```

- Tool calling stays structured (`qwen3_coder` parser kept — verified returning real `tool_calls` with correct JSON arguments).
- No hidden reasoning burn per turn → agent turns are faster.
- A request CAN still opt back into thinking with `"chat_template_kwargs": {"enable_thinking": true}` — **do not do this in sessions that carry tools** until upstream fixes #36537.

## Ops traps learned tonight (add to the repo README)

1. **Relaunch = tear down BOTH nodes first.** `docker rm -f sglang_qwen38fn` on Bluey AND Asusi before launching anything. If the old head is still up when the new worker starts, the worker rendezvouses with the dying head, dies with "Connection reset by peer" (exit 0, deceptively clean), and the new head hangs forever at "Init torch distributed begin."
2. **Capture `docker logs` BEFORE tearing down** — `docker rm -f` destroys the evidence.
3. Wedge signature to watch for: `Decode batch ... accept len: 1.00, accept rate: 0.00` sustained across all running requests + `Received output ... but the state was deleted in TokenizerManager` spam = sessions looping on token 0 with disconnected clients. Bounce the server.
4. Launch sequence (unchanged otherwise): drop_caches both nodes → Asusi NFS rw-check → `~/launch-qwen38fn-sglang-tp2.sh 1` on Asusi → wait 25s, confirm Up → `~/launch-qwen38fn-sglang-tp2.sh 0` on Bluey → ~7 min to SERVING.

## Client-side notes

- **OMP** (`~/.omp/agent/models.yml`, entry `qwen38fn-nvfp4`): no change needed; server default now protects it. Model card still advertises thinking levels — leave them unused for now.
- **dsh** (`~/.dsh/settings.yaml`, provider `qwen38fn`): route added 2026-08-26 with text+image verified. Same caveat documented inline.
- `max_tokens` includes hidden reasoning tokens IF thinking is re-enabled; with thinking off this footgun is gone too.

## Verification receipts

Fix relaunch completed 2026-08-26 ~18:32 UTC (clean both-nodes-down-first sequence). Receipts from the live server:

Server args confirm both flags active together:

```text
tool_call_parser='qwen3_coder'
default_chat_template_kwargs={'enable_thinking': False}
```

Tools request in OMP's exact shape — NO opt-out in the request body — returns a structured tool call with zero reasoning tokens:

```json
{"message":{"role":"assistant","content":"","reasoning_content":null,
 "tool_calls":[{"type":"function","function":{"name":"glob",
 "arguments":"{\"path\": \"C:/tmp/*.html\"}"}}]},
 "finish_reason":"tool_calls",
 "usage":{"prompt_tokens":295,"completion_tokens":29,"reasoning_tokens":0}}
```

Plain-text request also clean, `reasoning_tokens: 0`:

```text
"Tensor parallelism is a technique that splits individual layers of a
neural network across multiple devices..." — finish_reason: stop
```

Endpoint: `http://100.92.77.51:8000/v1`, model `qwen3.8-flash-next`, 262K context, MTP4 config unchanged.
