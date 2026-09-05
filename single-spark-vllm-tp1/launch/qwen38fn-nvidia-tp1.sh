#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 (nvidia/ModelOpt) on ONE DGX Spark, TP1, vLLM. Kai 2026-09-04.
# Defaults = the shipped production stack (2026-09-05): PLE table on disk, MTP4, FP8 KV, piecewise CUDA graphs, gmu 0.80, 262K.
# Env knobs: IMAGE, PLE_MODE (mmap|offload|none), GRAPHS (eager|piecewise|default), KV_DTYPE (auto|fp8_e4m3|nvfp4), OVERLAYS (1|0), PATCH_DIR, GMU, MAXLEN, SEQS, MTP (0|N), PORT, EXTRA, PREFIX_CACHE_ARG (default off: GDN prefix-cache crash vLLM #54173 by brainatworkharris, unverified here)
set -euo pipefail
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
NAME="${NAME:-vllm_qwen38fn}"
MODEL_HOST="/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia"
PATCH_DIR="${PATCH_DIR:-$HOME/patches/qwen4exp-ple-mmap}"
PLE_MODE="${PLE_MODE:-mmap}"
GMU="${GMU:-0.80}"; MAXLEN="${MAXLEN:-262144}"; SEQS="${SEQS:-8}"; MTP="${MTP:-4}"; PORT="${PORT:-8000}"
CHUNK="${CHUNK:-}"                   # --max-num-batched-tokens; 4096 under test for the single-Spark default (2026-09-05)
CACHE_HOST="/var/tmp/qwen38fn-vllm-cache"; mkdir -p "$CACHE_HOST"
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
PLE_ENV=(); case "$PLE_MODE" in
  mmap)    PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}"
           -v $PATCH_DIR/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py:ro
           -v $PATCH_DIR/ple_mmap.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py:ro) ;;
  staged)  # table on disk, rows gathered in the model state before each forward/graph replay (decode CUDA graphs with GRAPHS=nocompile)
           PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_STAGED=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}"
           -v $PATCH_DIR/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py:ro
           -v $PATCH_DIR/ple_mmap.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py:ro
           -v $PATCH_DIR/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py:ro) ;;
  offload) PLE_ENV=(-e VLLM_PLE_CPU_OFFLOAD=1) ;;
  none)    ;;
  *) echo "PLE_MODE must be mmap|staged|offload|none" >&2; exit 2 ;;
esac
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"; KV_ARGS=(); if [ "$KV_DTYPE" != "auto" ]; then KV_ARGS=(--kv-cache-dtype "$KV_DTYPE"); fi
# DRAFT_VOCAB=65536: reduced-vocabulary drafting for the MTP head (our overlay of vLLM's mtp.py; idea FR-Spec, shown on this model by MiaAI-Lab)
DRAFT_ENV=(); DRAFT_MOUNT=()
if [ -n "${DRAFT_VOCAB:-}" ]; then
  DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB="$DRAFT_VOCAB")
  DRAFT_MOUNT=(-v "$PATCH_DIR/mtp_draft_vocab.py:$VP/models/qwen4_exp/nvidia/mtp.py:ro")
fi
# Upstream overlays (credited, unmodified upstream vLLM code applied onto the nightly's files):
#   PR #55375 peakcrosser7 (merged 09-05): fused PLE conv state-index stride fix (MTP + concurrent prefills)
#   PR #54846 andreasgru (open): fp8_e4m3 / nvfp4 KV cache on the QSA path
#   modelopt.py: OUR two fixes (Kai, 2026-09-05): (1) draft-local MTP index candidates so mtp.layers.48 finds ModelOpt's
#     mtp.layers.0 entry; (2) FP8_BLOCK_SCALES routed experts -> Fp8MoEMethod (128x128 block scales). The same gaps were
#     fixed independently the same day by sfxnz (MIT) and MiaAI-Lab (AGPL); no shared code.
OVERLAYS="${OVERLAYS:-1}"; OVERLAY_MOUNT=()
if [ "$OVERLAYS" = "1" ]; then
  VP=/usr/local/lib/python3.12/dist-packages/vllm
  OVERLAY_MOUNT=(-v "$PATCH_DIR/upstream-overlays/ops_ple.py:$VP/models/qwen4_exp/nvidia/ops/ple.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/ops_qsa.py:$VP/models/qwen4_exp/nvidia/ops/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/qsa.py:$VP/models/qwen4_exp/nvidia/qsa.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/platforms_interface.py:$VP/platforms/interface.py:ro"
                 -v "$PATCH_DIR/upstream-overlays/modelopt.py:$VP/model_executor/layers/quantization/modelopt.py:ro")
fi
GRAPHS="${GRAPHS:-piecewise}"; GRAPH_ARGS=(); GRAPH_MOUNT=()
case "$GRAPHS" in
  eager)     GRAPH_ARGS=(--enforce-eager) ;;
  piecewise) GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
             GRAPH_MOUNT=(-v "$PATCH_DIR/compilation.py:/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py:ro") ;;
  # nocompile: CUDA graphs for decode without torch.compile (Inductor duplicates the PLE table at compile with a resident
  # table; with the table on disk this is simply the eager path + decode graphs). CAPTURE_SIZES="4,8,12,16" pins the
  # graph widths to (1+K)*seqs (MiaAI-Lab's README names this as the precondition for max-num-seqs > 4).
  nocompile) if [ -n "${CAPTURE_SIZES:-}" ]; then GRAPH_ARGS=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}");
             else GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'); fi ;;
  default)   ;;
  *) echo "GRAPHS must be eager|piecewise|nocompile|default" >&2; exit 2 ;;
esac
SPEC=(); if [ "$MTP" != "0" ]; then SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi
docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "${PLE_ENV[@]}" "${DRAFT_ENV[@]}" "${DRAFT_MOUNT[@]}" "${GRAPH_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    /models/qwen38fn --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --quantization modelopt --tensor-parallel-size 1 \
    --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" ${CHUNK:+--max-num-batched-tokens $CHUNK} \
    --no-enable-flashinfer-autotune ${PREFIX_CACHE_ARG:---no-enable-prefix-caching} \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs "{\"enable_thinking\": false}" \
    "${SPEC[@]}" "${GRAPH_ARGS[@]}" "${KV_ARGS[@]}" ${EXTRA:-}
echo "launched $NAME image=$IMAGE ple=$PLE_MODE graphs=$GRAPHS kv=$KV_DTYPE overlays=$OVERLAYS gmu=$GMU maxlen=$MAXLEN seqs=$SEQS mtp=$MTP"
sleep 3; docker ps --format "{{.Names}} {{.Status}}" | grep "$NAME" || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -5; exit 1; }
