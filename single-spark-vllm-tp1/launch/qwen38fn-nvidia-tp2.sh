#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 (nvidia/ModelOpt) on TWO DGX Sparks, TP2 over the RoCE fabric, vLLM. Kai / Tech2Wild 2026-09-05.
# DEFAULT = SPEED profile (measured 2026-09-05): n-gram table in unified memory (stock loader), CUDA graphs for decode with
# torch.compile off, MTP3, 6 seqs, 4096-token prefill chunks, FP8 KV, gmu 0.70, 262K. 53.7 tok/s median single stream, 97.9 agg at x6,
# KV pool 1.97M. CONTEXT profile (table on disk via our patch, 5.87M pool, 35.8 median): PLE_MODE=mmap GRAPHS=piecewise MTP=4 SEQS=8 GMU=0.80.
# Usage: qwen38fn-nvidia-tp2.sh <0|1>   (run rank 1 = worker FIRST, then rank 0 = head)
# Env knobs: IMAGE, PLE_MODE (mmap = table on disk [CONTEXT] | resident = our patch keeps each rank's slice in memory [SPEED] | none = stock loader, needs GRAPHS=nocompile),
#   GRAPHS (eager|piecewise|full|nocompile|default), LANE (B = Reddie head + Spark4 [default] | A = Bluey head + Asusi), KV_DTYPE, OVERLAYS,
#   PATCH_DIR, GMU, MAXLEN, SEQS, MTP, PORT, MPORT, TOOL_PARSER (qwen3_xml|qwen3_coder), EXTRA
set -euo pipefail
NODE_RANK="${1:?usage: qwen38fn-nvidia-tp2.sh <0|1>}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
NAME="${NAME:-vllm_qwen38fn}"
MODEL_HOST="${MODEL_HOST:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia}"
PATCH_DIR="${PATCH_DIR:-$HOME/patches/qwen4exp-ple-mmap}"
PLE_MODE="${PLE_MODE:-none}"       # SPEED default (2026-09-05 evening): table in unified memory. CONTEXT: PLE_MODE=mmap
GMU="${GMU:-0.70}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-6}"; MTP="${MTP:-3}"; PORT="${PORT:-8000}"
CHUNK="${CHUNK:-4096}"              # --max-num-batched-tokens; the single biggest speed lever measured on GB10 (see README)
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; GRAPHS="${GRAPHS:-nocompile}"; OVERLAYS="${OVERLAYS:-1}"
LANE="${LANE:-B}"
case "$LANE" in
  B) HEAD_IP="192.168.192.2"; WORKER_IP="192.168.192.4"; MPORT="${MPORT:-29531}" ;;  # Reddie head, Spark4 worker
  A) HEAD_IP="192.168.192.1"; WORKER_IP="192.168.192.3"; MPORT="${MPORT:-29532}" ;;  # Bluey head, Asusi worker
  *) echo "LANE must be A or B" >&2; exit 2 ;;
esac
case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP"; HEADLESS="" ;;               # head, serves :PORT
  1) HOST_IP="$WORKER_IP"; HEADLESS="--headless" ;;   # worker
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac
CACHE_HOST="/var/tmp/qwen38fn-vllm-cache"; mkdir -p "$CACHE_HOST"
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST (each rank needs a readable copy: local NVMe, or an NFS mount when PLE_MODE=none)" >&2; exit 3; }
VP=/usr/local/lib/python3.12/dist-packages/vllm
PLE_ENV=(); PLE_MOUNT=()
if [ "$PLE_MODE" = "mmap" ] || [ "$PLE_MODE" = "resident" ]; then
  PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}")
  # resident: each rank keeps its slice of the FP8 table as a plain GPU tensor behind our gather op
  # (compile stays on, no Inductor copy of the table). TP2: 23.8 GiB per rank, TP4: 11.9 GiB.
  [ "$PLE_MODE" = "resident" ] && PLE_ENV+=(-e QWEN4EXP_PLE_RESIDENT=1)
  PLE_MOUNT=(-v "$PATCH_DIR/ple_layer.py:$VP/models/qwen4_exp/nvidia/ple_layer.py:ro"
             -v "$PATCH_DIR/ple_mmap.py:$VP/models/qwen4_exp/nvidia/ops/ple_mmap.py:ro")
fi
KV_ARGS=(); if [ "$KV_DTYPE" != "auto" ]; then KV_ARGS=(--kv-cache-dtype "$KV_DTYPE"); fi
OVERLAY_MOUNT=()
if [ "$OVERLAYS" = "1" ]; then
  OVERLAY_MOUNT=(-v "$PATCH_DIR/upstream-overlays/ops_ple.py:$VP/models/qwen4_exp/nvidia/ops/ple.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/ops_qsa.py:$VP/models/qwen4_exp/nvidia/ops/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/qsa.py:$VP/models/qwen4_exp/nvidia/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/platforms_interface.py:$VP/platforms/interface.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/modelopt.py:$VP/model_executor/layers/quantization/modelopt.py:ro")
fi
GRAPH_ARGS=(); GRAPH_MOUNT=()
case "$GRAPHS" in
  eager)     GRAPH_ARGS=(--enforce-eager) ;;
  piecewise) GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
             GRAPH_MOUNT=(-v "$PATCH_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
  full)      GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
             GRAPH_MOUNT=(-v "$PATCH_DIR/compilation.py:$VP/config/compilation.py:ro") ;;
  # nocompile: CUDA graphs for decode WITHOUT torch.compile. Needed when the PLE table is resident (PLE_MODE=none):
  # Inductor autotune duplicates the n-gram table during compile (~24 GiB per rank at TP2, ~50 GiB at TP1), which
  # starved and rebooted two Sparks on 2026-09-05. gau-nernst's open vLLM PR #55272 removes compile for that reason.
  nocompile) GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  default)   ;;
  *) echo "GRAPHS must be eager|piecewise|full|nocompile|default" >&2; exit 2 ;;
esac
# MTP_INDEX_SHARE=1 adds index_share_for_mtp_iteration=true (QSA indexer top-k reused across MTP draft steps; knob named by Chuck 208 @CK2084, 2026-09-05)
SPEC=(); if [ "$MTP" != "0" ]; then
  if [ "${MTP_INDEX_SHARE:-0}" = "1" ]; then SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP,\"index_share_for_mtp_iteration\":true}");
  else SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi
fi
# ASYNC_SCHED=1 adds --async-scheduling (also from Chuck 208's list)
ASYNC_ARGS=(); [ "${ASYNC_SCHED:-0}" = "1" ] && ASYNC_ARGS=(--async-scheduling)
docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache" \
  -e VLLM_HOST_IP="$HOST_IP" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "${PLE_ENV[@]}" "${PLE_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${GRAPH_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --quantization modelopt --tensor-parallel-size 2 \
    --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" --max-num-batched-tokens "$CHUNK" \
    --no-enable-flashinfer-autotune ${PREFIX_CACHE_ARG:---no-enable-prefix-caching} \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_xml}" \
    --default-chat-template-kwargs "{\"enable_thinking\": false}" \
    "${SPEC[@]}" "${ASYNC_ARGS[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS ${EXTRA:-}
echo "launched $NAME lane=$LANE rank=$NODE_RANK host=$HOST_IP tp=2 ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE mtp=$MTP gmu=$GMU maxlen=$MAXLEN"
sleep 3; docker ps --format "{{.Names}} {{.Status}}" | grep "$NAME" || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
