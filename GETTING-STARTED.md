# Getting Started: from zero to a working endpoint

The lane READMEs document the engine; this document is the road to it. It covers everything between "I cloned this repo" and "my editor is talking to the model" — checkpoint download, patch placement, launch, health checks, and client configuration. Written from the steps a fresh machine actually needs, in order.

All commands run **on the Spark** unless marked otherwise. `~` is your home directory on the Spark.

## 0. Prerequisites

- DGX Spark (GB10) with Docker and the NVIDIA container runtime (`docker run --gpus all` works).
- ~260 GB free: the checkpoint is 124 GB on disk and the default HF cache will hold it once (see the hardlink step below to avoid a second copy).
- `git`, `python3`, and `huggingface_hub` (`pip3 install -U huggingface_hub`).
- This repo cloned on the Spark:

```bash
git clone https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark.git
```

## 1. Download the checkpoint

The launcher does not download anything. It hard-fails with `MODEL MISSING` unless the checkpoint sits at `/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia/config.json` (line 17 of the launcher). The checkpoint is `nvidia/Qwen3.8-Flash-Next-NVFP4` (124 GB: weights + the 47.68 GiB PLE table).

```bash
mkdir -p /var/tmp/models
python3 - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/Qwen3.8-Flash-Next-NVFP4", max_workers=4)
print("SNAPSHOT", p)
open("/var/tmp/model-snapshot-path.txt", "w").write(p)
EOF
```

- If the repo asks for acceptance on the HF page, accept it there first and export `HF_TOKEN` in the shell running the download. **Do not commit a token.**
- The download is resumable: re-run the same snippet and `snapshot_download` picks up the missing shards. If a resume stalls at 0 B/s, clear the per-file inactivity markers first — they block the resume:

  ```bash
  find ~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots -name .cache -type d -exec rm -rf {} + 2>/dev/null
  ```

- Re-run until the snippet prints `SNAPSHOT` and `config.json` exists in the snapshot.

### Stage it at the path the launcher expects

```bash
SNAP=$(cat /var/tmp/model-snapshot-path.txt)
rm -f /var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia
ln -L "$SNAP" /var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia   # -L = hardlink through symlinks
ls /var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia/config.json && echo SYMLINK_OK
```

`ln -L` hardlinks the blob files instead of copying, so staging costs ~0 bytes when the HF cache and `/var/tmp` share a filesystem (they do on a stock Spark). A plain `cp -r` would mean a second 124 GB.

## 2. Place the patch files

The launcher mounts individual files from `PATCH_DIR` (default `$HOME/patches/qwen4exp-ple-mmap`) into the container — it does not read them out of the repo checkout. Copy them into place:

```bash
mkdir -p ~/patches/qwen4exp-ple-mmap
cp -r single-spark-vllm-tp1/patch/. ~/patches/qwen4exp-ple-mmap/
ls ~/patches/qwen4exp-ple-mmap/upstream-overlays/   # sanity: modelopt.py etc. present
```

Or point the launcher at the checkout instead: `PATCH_DIR=$PWD/single-spark-vllm-tp1/patch bash launch/qwen38fn-nvidia-tp1.sh`.

Files the default (`PLE_MODE=staged`, `OVERLAYS=1`) launch path mounts:

| File | Mounted as |
|---|---|
| `ple_layer.py` | `vllm/models/qwen4_exp/nvidia/ple_layer.py` |
| `ple_mmap.py` | `vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py` |
| `model_state.py` | `vllm/models/qwen4_exp/nvidia/model_state.py` |
| `mtp_draft_vocab.py` | `vllm/models/qwen4_exp/nvidia/mtp.py` (when `DRAFT_VOCAB` set) |
| `upstream-overlays/{ops_ple,ops_qsa,qsa,platforms_interface,modelopt}.py` | upstream overlay targets |

A missing file under `PATCH_DIR` shows up as a Docker `-v` source error at `docker run`, not a Python traceback.

## 3. Launch

```bash
bash single-spark-vllm-tp1/launch/qwen38fn-nvidia-tp1.sh
```

The first run pulls the pinned nightly image (~20 GB) — allow extra time. Weights load in **~9-10 minutes** (the loader still streams the 47.68 GiB PLE file through RAM). Watch it:

```bash
docker logs -f vllm_qwen38fn        # "Uvicorn running" / "ready" at the end
curl -s http://localhost:8000/v1/models
```

## 4. Health checks

The model id is **exact-match**: `qwen3.8-flash-next` (all lowercase). `Qwen3.8-Flash-Next`, `Qwen3.8-Flash-Next-NVFP4`, etc. all return `404 model does not exist`.

```bash
# models + context window (max_model_len: 262144)
curl -s http://localhost:8000/v1/models

# basic chat
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"say OK"}],"max_tokens":10}'

# tool calling (agents need this; parser is qwen3_xml + auto tool choice)
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
       "tool_choice":"auto","max_tokens":200}' | grep -o '"finish_reason":"tool_calls"'
```

Vision (base64 data-URI images) and SSE streaming also work; thinking is **off by default** (`enable_thinking` per request) and the reasoning parser strips any reasoning into `message.reasoning` either way.

## 5. Use it from a client (Cline / VS Code)

The server is plain OpenAI-compatible, so any client works. From a different machine, tunnel the port:

```bash
ssh -N -L 8000:localhost:8000 <user>@<your-spark>      # keep this running
```

To survive reboots/disconnects, put the forward in `~/.ssh/config` (`Host spark` … `LocalForward 8000 localhost:8000`) and supervise the ssh with autossh or a launchd/`systemd` unit that restarts it on failure.

Cline → ⚙️ (API Configuration):

| Field | Value |
|---|---|
| API Provider | `OpenAI Compatible` |
| Base URL | `http://localhost:8000/v1` |
| API Key | any string (the server runs without `--api-key`) |
| Model ID | `qwen3.8-flash-next` |
| Context Window | `262144` |
| Max Output Tokens | `8192` |
| Image Support / Tool (Computer Use) | both enabled — tested working |

Hit **Verify**, then confirm with a trivial prompt. Same values apply to other OpenAI-compatible clients (Continue, Roo Code, Open WebUI, LangChain's `OpenAI(base_url=...)`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `MODEL MISSING at /var/tmp/models/...` and exit 3 | Checkpoint not staged — step 1, in particular the `ln -L` step |
| Docker `-v` source error at run | `PATCH_DIR` files missing — step 2 |
| `404 The model ... does not exist` | Wrong model id — use `qwen3.8-flash-next` exactly |
| Download resumes at 0 B/s | Stale `.cache` inactivity dirs — delete them (step 1) |
| `FileNotFoundError` crash loop over safetensors shards | Incomplete/staged snapshot (partial download or a symlink into a half-finished cache). Re-run `snapshot_download`, then re-do the `ln -L` staging |
| Nothing answers on port 8000 for ~10 min | Normal — weights load in ~9-10 min. `docker logs -f vllm_qwen38fn` |
| Client connects, then dies after a while | The ssh tunnel dropped — supervise it (step 5) |
| OOM / swap on long runs | Keep `MemAvailable` ≥ ~10 GB; shipped default is `GMU=0.80` (see the KV pool ledger in the lane README) |

