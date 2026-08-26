# Qwen3.8-Flash-Next-NVFP4 TP2 Deployment Report

Deployment date: 2026-08-26  
Endpoint: `http://100.92.77.51:8000/v1`  
Model: `qwen3.8-flash-next`  
Final status: **SERVING on Bluey + Asusi as TP2 with MTP4**

Operator directive after deployment: **leave Qwen up; do not restore DeepSeek/DS4**.

## 1. Timestamps

All timestamps below include UTC and America/New_York (EDT, UTC-04:00).

| Event | UTC | EDT |
|---|---:|---:|
| Download verified complete on Bluey | 2026-08-26 13:49:04 | 2026-08-26 09:49:04 |
| DS4 TP4 teardown completed | 2026-08-26 14:22:49 | 2026-08-26 10:22:49 |
| First launch: Asusi worker rank 1 | 2026-08-26 14:24:13 | 2026-08-26 10:24:13 |
| First launch: Bluey head rank 0 | 2026-08-26 14:24:49 | 2026-08-26 10:24:49 |
| Final launch: Asusi worker rank 1 | 2026-08-26 14:38:33 | 2026-08-26 10:38:33 |
| Final launch: Bluey head rank 0 | 2026-08-26 14:39:13 | 2026-08-26 10:39:13 |
| SERVING / Uvicorn ready | 2026-08-26 14:45:31 | 2026-08-26 10:45:31 |
| MTP4 restart: Asusi worker rank 1 | 2026-08-26 14:56:55 | 2026-08-26 10:56:55 |
| MTP4 restart: Bluey head rank 0 | 2026-08-26 14:57:19 | 2026-08-26 10:57:19 |
| MTP4 SERVING / Uvicorn ready | 2026-08-26 15:04:42 | 2026-08-26 11:04:42 |

Download verification receipts:

```text
DOWNLOAD-COMPLETE
126G /var/tmp/models/qwen3.8-flash-next-nvfp4
config.json present
420 repository files (839 raw paths including cache/metadata files)
```

Asusi read/write verification on the NFS mount succeeded before launch.

## 2. Architecture probe and selected image

The required pre-teardown vLLM probe was run first. Neither candidate vLLM image registered the checkpoint's architecture, `Qwen4ExpForConditionalGeneration`, so no teardown occurred until this was reported and Tony explicitly authorized SGLang.

Probe output (the two vLLM images produced the same Qwen registry list):

```text
==vllm/vllm-openai:qwen38-arm64-cu130
qwen archs: ['Qwen3NextForCausalLM', 'Qwen2ForCausalLM', 'Qwen2MoeForCausalLM',
'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM', 'Qwen3_5ForCausalLM',
'Qwen3_5MoeForCausalLM', 'Qwen2Model', 'VoyageQwen3BidirectionalEmbedModel',
'Qwen2VLForConditionalGeneration', 'ColQwen3', 'OpsColQwen3Model', 'ColQwen3_5',
'Qwen3VLNemotronEmbedModel', 'Qwen2ForRewardModel', 'Qwen2ForProcessRewardModel',
'Qwen3ASRForcedAlignerForTokenClassification', 'Qwen2_5_VLForConditionalGeneration',
'Qwen2AudioForConditionalGeneration', 'Qwen2_5OmniModel',
'Qwen2_5OmniForConditionalGeneration', 'Qwen3OmniMoeForConditionalGeneration',
'Qwen3ASRForConditionalGeneration', 'Qwen3ASRRealtimeGeneration',
'Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration',
'Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForConditionalGeneration',
'Qwen3DSparkModel', 'Eagle3Qwen2_5vlForCausalLM', 'Eagle3Qwen3vlForCausalLM',
'Eagle3Qwen3ForCausalLM', 'PeagleQwen3ForCausalLM', 'Qwen3NextMTP',
'Qwen3_5MTP', 'Qwen3_5MoeMTP']
model declares: ['Qwen4ExpForConditionalGeneration']

==eugr/spark-vllm:nightly-20260815
qwen archs: [same list as above; Qwen4ExpForConditionalGeneration absent]
model declares: ['Qwen4ExpForConditionalGeneration']
```

The checkpoint README and qualification notes require SGLang's native `qwen4_exp` model and NVFP4 PLE loader. Tony approved SGLang. The official day-zero image was selected:

```text
lmsysorg/sglang:qwen38flashnext
base digest: sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
sglang: 0.0.0.dev1+gd91c3682b
transformers: 5.12.1
qwen4_exp native: True
```

The final deployed image is a minimal derivative:

```text
radixark/sglang-qwen38flashnext:sm121-qsa
Bluey build manifest: sha256:ebf937d186bc58efc04606a5c83a21b2059587fd8e83bd11eb2ac851b25a28dc
```

It changes one SGLang architecture guard so the bundled FlashInfer TRT-LLM sparse-decode kernel, which passed a direct representative-shape probe on Bluey's SM121 GPU, is selected on SM121 as well as SM100. The model, checkpoint, quantization loader, and serve arguments are unchanged.

Representative kernel probe receipt:

```text
GPU capability: (12, 1)
model shape: 24 query heads, 2 KV heads, head_dim 256
TRT-LLM batch decode: PASS torch.Size([1, 24, 256]) torch.bfloat16
```

## 3. Exact final launch commands

Pre-launch on both Bluey and Asusi:

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

The exact launcher installed as `~/launch-qwen38fn-sglang-tp2.sh` on both nodes is below. It contains the complete Docker command and environment verbatim:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-Flash-Next-NVFP4 on Bluey + Asusi using the official day-0 SGLang image.
# Run worker first: this script with rank 1 on Asusi, then rank 0 on Bluey.
NODE_RANK="${1:?usage: launch-qwen38fn-sglang-tp2.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || {
  echo "rank must be 0 or 1" >&2
  exit 2
}

IMAGE="radixark/sglang-qwen38flashnext:sm121-qsa"
NAME="sglang_qwen38fn"
MODEL_HOST_PATH="/var/tmp/models/qwen3.8-flash-next-nvfp4"
MODEL_PATH="/models/qwen3.8-flash-next-nvfp4"
NCCL_HOST_PATH="/var/tmp/models/hub/nccl-2.30.4"
CACHE_HOST_PATH="/var/tmp/qwen38fn-sglang-cache"
HEAD_IP="192.168.192.1"
INIT_PORT="29511"
PORT="8000"

test -f "$MODEL_HOST_PATH/config.json"
test -f "$NCCL_HOST_PATH/libnccl.so.2"
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --memory 110g --memory-swap 110g \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$NCCL_HOST_PATH:/opt/nccl:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -e LD_PRELOAD=/opt/nccl/libnccl.so.2 \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_CACHE=/cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
  sglang serve \
    --model-path "$MODEL_PATH" \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size 2 \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --dist-init-addr "$HEAD_IP:$INIT_PORT" \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --mamba-track-interval 64 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-linear-replayssm-spec \
    --chunked-prefill-size 4096 \
    --max-running-requests 6 \
    --context-length 262144 \
    --mem-fraction-static 0.78 \
    --allow-auto-truncate \
    --reasoning-parser auto \
    --trust-remote-code \
    --disable-cuda-graph

echo "launched $NAME rank=$NODE_RANK"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
```

Exact node invocations, worker first:

```bash
# Asusi (192.168.192.3), rank 1
~/launch-qwen38fn-sglang-tp2.sh 1

# Waited about 20 seconds and confirmed the worker remained Up.

# Bluey (192.168.192.1), rank 0/head
~/launch-qwen38fn-sglang-tp2.sh 0
```

## 4. Failures and fixes

1. **Both required vLLM architecture probes failed.** Neither registry contained `Qwen4ExpForConditionalGeneration`. Per the order, execution stopped before teardown and the raw result was reported. Tony then authorized SGLang. Fix: use the checkpoint's required official SGLang day-zero image.

2. **Direct image acquisition on Asusi was slow/unreliable.** Fix: pull/build on Bluey and transfer with `docker save ... | ssh ... docker load` over the 192.168.192.x fabric.

3. **First SGLang boot loaded all 206 shards but died in final warmup.** The FA4 CUTE fallback in `qwen_sparse_attn_backend.py` failed on SM121 with:

   ```text
   MLIRError: Operation creation failed:
   expects `coord` and shape of view are weakly congruent
   Received sigquit from a child process.
   ```

   Root cause: the image's QSA sparse-decode resolver restricted the bundled TRT-LLM path to `is_sm100_supported()` and fell back to FA4 CUTE on SM121. Fix: directly probe the same FlashInfer TRT-LLM decode function on Bluey's SM121 GPU with the model's real head shape, then build a one-line guard extension accepting `is_sm120_supported()`. The second launch passed warmup and served requests.

4. **Two local PowerShell SSH commands hit quoting/parser errors while collecting source evidence.** No remote state changed. Fix: split the compound probes into simpler single-quoted SSH commands.

No CUDA OOM, NCCL failure, watchdog reset, thermal throttle, NFS error, or unexplained second death occurred.

## 5. Engine receipts

```text
server_args: nnodes=2, node_rank=0/1, tp_size=2
dist_init_addr='192.168.192.1:29511'
[TP0] sglang is using nccl==2.30.4
[TP0] Init torch distributed ends. elapsed=8.03 s
[TP1] Init torch distributed ends
```

```text
context_length=262144
max_running_requests=6
mem_fraction_static=0.78
quantization='modelopt_fp4'
fp4_gemm_runner_backend='flashinfer_cutlass'
```

```text
[TP0] Load weight end. elapsed=282.07 s,
type=Qwen4ExpForConditionalGeneration, quant=modelopt_fp4,
quant_algo=NVFP4, avail mem=35.23 GB, mem usage=76.11 GB.
```

```text
[TP0] Mamba Cache is allocated. max_mamba_cache_size: 97,
conv_state size: 0.10GB, ssm_state size: 5.17GB
[TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 484032,
K size: 2.77 GB, V size: 2.77 GB
[TP0] Memory pool end. avail mem=24.45 GB
[TP0] max_total_num_tokens=484032, chunked_prefill_size=4096,
max_prefill_tokens=16384, max_running_requests=6,
context_len=262144, available_gpu_mem=19.86 GB
```

```text
[2026-08-26 14:45:31] Application startup complete.
[2026-08-26 14:45:31] Uvicorn running on http://0.0.0.0:8000
```

`GET /v1/models` receipt:

```json
{"object":"list","data":[{"id":"qwen3.8-flash-next","object":"model","owned_by":"sglang","root":"qwen3.8-flash-next","max_model_len":262144}]}
```

## 6. Smoke and benchmark

Exact smoke curls used (Windows `curl.exe` syntax; payload contents shown inline):

```powershell
curl.exe -sS --max-time 180 -H "Content-Type: application/json" --data-binary "@qwen38fn-spec-smoke.json" http://100.92.77.51:8000/v1/chat/completions
```

Equivalent portable curl:

```bash
curl http://100.92.77.51:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello and name yourself."}],"max_tokens":40}'
```

Smoke response receipt:

```text
content: "Hello! My name is Qwen."
finish_reason: stop
prompt_tokens: 58
completion_tokens: 31 (including 21 reasoning tokens)
```

Benchmark request: OpenAI chat-completions streaming, `max_tokens=200`, temperature 0, with usage included. TTFT is measured from request start to the first non-empty content or reasoning delta. Decode rate is `(completion_tokens - 1) / (end - first_token)`.

| Run | TTFT | Total | Completion tokens | Decode tok/s |
|---:|---:|---:|---:|---:|
| 1 | 0.3576 s | 10.0299 s | 200 | 20.574 |
| 2 | 0.6572 s | 10.5860 s | 200 | 20.043 |
| 3 | 1.1779 s | 11.4202 s | 200 | 19.429 |

Middle/median result: **0.6572 s TTFT, 20.043 decode tok/s**.

### MTP4 restart validation

The checkpoint contains one built-in MTP layer (`mtp_num_hidden_layers: 1`). It was relaunched with the image's Qwen3.8 MTP4 recipe: NEXTN/EAGLE, three speculative steps, top-k 1, four draft tokens, and ReplaySSM speculation. No separate draft checkpoint is required.

```text
speculative_algorithm='EAGLE'
speculative_draft_model_path='/models/qwen3.8-flash-next-nvfp4'
speculative_num_steps=3
speculative_eagle_topk=1
speculative_num_draft_tokens=4
speculative_draft_model_quantization='modelopt_fp4'
enable_linear_replayssm_spec=True
```

```text
Load weight end. elapsed=55.60 s, type=Qwen4ExpForCausalLMMTP,
quant=modelopt_fp4, quant_algo=NVFP4, mem usage=1.90 GB.
QSA MTP index sharing enabled: draft decode steps reuse the
draft-extend selection for layers [0]
```

MTP4 scheduler receipts under real generation traffic:

```text
mamba num: 4, accept len: 2.67, accept rate: 0.56
mamba num: 4, accept len: 2.20, accept rate: 0.40
mamba num: 4, accept len: 2.15, accept rate: 0.38
mamba num: 4, accept len: 2.08, accept rate: 0.36
mamba num: 4, accept len: 2.62, accept rate: 0.54
```

Three post-MTP4 200-token runs:

| Run | TTFT | Total | Completion tokens | Decode tok/s |
|---:|---:|---:|---:|---:|
| 1 | 0.4848 s | 8.0695 s | 200 | 26.237 |
| 2 | 0.6945 s | 6.7645 s | 200 | 32.784 |
| 3 | 0.1823 s | 5.7762 s | 200 | 35.574 |

Post-MTP4 median decode: **32.784 tok/s**, versus **20.043 tok/s** before MTP4 (about **63.6% faster** on this three-run prompt test).

## 7. Final fleet state

Audited at 2026-08-26 14:47:30 UTC / 10:47:30 EDT.

| Node | Qwen TP2 | H3 `comfy-h3-x` | DS4 TP4 | GPU temp |
|---|---|---|---|---:|
| Bluey | `sglang_qwen38fn` UP, rank 0/head, :8000 | DOWN (`Exited (137)`) | DOWN/removed | 47 C |
| Reddie | n/a | UP (20 hours) | DOWN/removed | 43 C |
| Asusi | `sglang_qwen38fn` UP, rank 1/worker | DOWN (`Exited (137)`) | DOWN/removed | 47 C |
| Spark4 | n/a | UP (20 hours) | DOWN/removed | 44 C |

Endpoint UP: `http://100.92.77.51:8000/v1`  
Model UP: `qwen3.8-flash-next`  
DS4 TP4: DOWN on all four nodes  
H3: UP only on Reddie and Spark4; DOWN on Bluey and Asusi

## 8. Recommended README changes

1. Document that the two current vLLM images do not support `Qwen4ExpForConditionalGeneration`; use SGLang until vLLM adds the architecture and ModelOpt NVFP4 PLE loader.
2. Pin the official base image digest and publish the minimal SM121 QSA derivative or upstream its architecture-guard fix. Do not silently use the FA4 CUTE fallback on DGX Spark/SM121.
3. Include the exact worker-first TP2 sequence and the fabric NCCL variables from the launcher above.
4. Keep `drop_caches` on both nodes as a mandatory pre-launch step and validate the Asusi NFS mount read/write before every start.
5. Set startup expectations: about 6 minutes end-to-end, 206 shards, about 76 GB model memory per rank, 484,032 KV tokens, and about 24.45 GB free immediately after memory-pool allocation.
6. Explain that `max_tokens` includes hidden reasoning tokens for this model; very small values can truncate visible content.
7. Record the current day-zero warnings as non-fatal: deprecated mamba/CUDA-graph flags, unknown multimodal RoPE keys, and missing `torchcodec` for audio. Image input remains enabled; audio is not qualified.
8. Mark CUDA graphs disabled for this first stable deployment. Re-enable and benchmark only as a separate controlled change.

## Restore note

No restore was performed. Tony explicitly directed that Qwen remain up and that DeepSeek/DS4 not be restored. Only after a future explicit restore order, follow the companion handoff worker-first for DS4 TP4, then start `comfy-h3-x` everywhere and have Kai repoint the relay.

## OMP model registration

Added and validated in `C:\Users\tonyd\.omp\agent\models.yml`:

```yaml
qwen38fn-nvfp4:
  baseUrl: http://100.92.77.51:8000/v1
  api: openai-completions
  auth: none
  models:
    - id: qwen3.8-flash-next
      name: Qwen 3.8 Flash Next NVFP4
      input: [text, image]
      reasoning: true
      contextWindow: 262144
      maxTokens: 32768
```

OMP validation receipt:

```text
qwen38fn-nvfp4 (1)
model: qwen3.8-flash-next
context: 262K
max-out: 33K
thinking: minimal,low,medium,high
images: yes
```
