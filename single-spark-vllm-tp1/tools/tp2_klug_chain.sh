#!/usr/bin/env bash
# After the async knob: (a) bf16 KV on SPEED (TJ Klug: FP8 KV cost prose via MTP acceptance), (b) NCCL 8 channels on SPEED
cd ~/.openclaw/workspace/qwen38fn-tp1-single-spark
J='ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=25 -o StrictHostKeyChecking=no'
boot() { ssh -i ~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o ConnectTimeout=20 tonyspark3@100.90.25.78 "J=\"$J\"; docker rm -f vllm_qwen38fn >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; \$J tonyspark1@10.0.0.6 'docker rm -f vllm_qwen38fn >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1'; echo cleared; sleep 5; env LANE=A $1 MODEL_HOST=/mnt/bluey-models/Qwen3.8-Flash-Next-NVFP4-nvidia ~/qwen38fn-nvidia-tp2.sh 1 2>&1 | tail -1; \$J tonyspark1@10.0.0.6 \"LANE=A $1 ~/qwen38fn-nvidia-tp2.sh 0 2>&1 | tail -1\"" 2>&1 | grep -v "^Warning"; echo "RELAUNCHED $2 $(date +%H:%M:%S)"; sleep 120; until curl -s -m 5 -o /dev/null -w "%{http_code}" http://100.92.77.51:8000/v1/models 2>/dev/null | grep -q 200; do sleep 5; done; echo "ENDPOINT UP $2 $(date +%H:%M:%S)"; sleep 15; tools/bench_lane.sh http://100.92.77.51:8000 qwen3.8-flash-next "$2"; echo "HARNESS DONE $2 $(date +%H:%M:%S)"; }
until grep -q "TP2 ASYNC HARNESS DONE" results/tp2_async_chain.log 2>/dev/null; do sleep 15; done
echo "ASYNC DONE $(date +%H:%M:%S)"
boot "GMU=0.70 KV_DTYPE=auto" qwen38fn_tp2_speed_bf16kv_070
boot "GMU=0.70 NCCL_CHANNELS=8" qwen38fn_tp2_speed_nccl8_070
echo "TP2 KLUG KNOBS ALL DONE $(date +%H:%M:%S)"
