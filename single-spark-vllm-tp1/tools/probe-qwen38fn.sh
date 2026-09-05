#!/usr/bin/env bash
# Probe the Qwen3.8-Flash-Next TP1 container on Reddie via the Asusi jump. Prints: status line, then key log lines.
MODE="${1:-brief}"
ssh -i /Users/clawdbot/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o ConnectTimeout=10 tonyspark3@100.90.25.78 \
  "ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=8 -o StrictHostKeyChecking=no tonyspark2@10.0.0.9 'bash -s' <<'REMOTE'
ST=\$(docker inspect -f '{{.State.Status}}' vllm_qwen38fn 2>/dev/null || echo missing)
echo "STATUS=\$ST"
if [ "$MODE" = "full" ]; then
  docker logs vllm_qwen38fn 2>&1 | grep -vE 'deprecated|register_opaque' | grep -B2 -A14 -E 'Traceback|Error' | tail -50 | cut -c1-240
  echo '---- tail'; docker logs vllm_qwen38fn 2>&1 | tail -8 | cut -c1-240
else
  docker logs vllm_qwen38fn 2>&1 | grep -E 'PLEMmapTable|checkpoint shards: 100|Loading weights took|GPU KV cache size|Maximum concurrency|Application startup|Traceback|OutOfMemory|NV_ERR|CUDA error|RuntimeError|ValueError' | grep -vE 'deprecated|register_opaque' | tail -4 | cut -c1-220
  docker logs vllm_qwen38fn 2>&1 | grep -oE 'checkpoint shards: +[0-9]+% Completed \| [0-9]+/[0-9]+' | tail -1
fi
REMOTE" 2>&1 | grep -v '^Warning'
