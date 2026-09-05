#!/usr/bin/env bash
# Chuck knob 3: --async-scheduling on the TP2 SPEED default (Bluey+Asusi), after the MTP4 index-share harness
cd ~/.openclaw/workspace/qwen38fn-tp1-single-spark
until grep -q "TP2 MTP4 IXS HARNESS DONE" results/bench_tp2_speed_mtp4_ixs_070.log 2>/dev/null; do sleep 15; done
echo "IXS HARNESS DONE $(date +%H:%M:%S)"
ssh -i ~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o ConnectTimeout=20 tonyspark3@100.90.25.78 'J="ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=25 -o StrictHostKeyChecking=no"; docker rm -f vllm_qwen38fn >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; $J tonyspark1@10.0.0.6 "docker rm -f vllm_qwen38fn >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1"; echo cleared; sleep 5; LANE=A GMU=0.70 ASYNC_SCHED=1 MODEL_HOST=/mnt/bluey-models/Qwen3.8-Flash-Next-NVFP4-nvidia ~/qwen38fn-nvidia-tp2.sh 1 2>&1 | tail -1; $J tonyspark1@10.0.0.6 "LANE=A GMU=0.70 ASYNC_SCHED=1 ~/qwen38fn-nvidia-tp2.sh 0 2>&1 | tail -1"' 2>&1 | grep -v "^Warning"
echo "RELAUNCHED ASYNC $(date +%H:%M:%S)"; sleep 120
until curl -s -m 5 -o /dev/null -w "%{http_code}" http://100.92.77.51:8000/v1/models 2>/dev/null | grep -q 200; do sleep 5; done
echo "ENDPOINT UP $(date +%H:%M:%S)"; sleep 15
tools/bench_lane.sh http://100.92.77.51:8000 qwen3.8-flash-next qwen38fn_tp2_speed_async_070
echo "TP2 ASYNC HARNESS DONE $(date +%H:%M:%S)"
