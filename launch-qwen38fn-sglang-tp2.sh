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
