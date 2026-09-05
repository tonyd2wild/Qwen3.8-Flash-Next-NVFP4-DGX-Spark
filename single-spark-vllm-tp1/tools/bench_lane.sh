#!/usr/bin/env bash
# Full lane benchmark for the single-Spark Qwen3.8-Flash-Next: real-prompt categories at C1/C2/C4/C6,
# cold long-prefill ladder, then the count-to-100 ceiling (peak row only, never a headline).
# Usage: tools/bench_lane.sh <base_url> <served_model> <lane>
set -uo pipefail
BASE="${1:?base}"; MODEL="${2:?model}"; LANE="${3:?lane}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"; mkdir -p results
PY=/opt/homebrew/bin/python3.11
echo "== $LANE start $(date '+%H:%M:%S')"
for C in 1 2 4 6; do
  echo "== $LANE C$C $(date '+%H:%M:%S')"
  "$PY" tools/bench_categories.py "$BASE" "$MODEL" "$LANE" --thinking off --concurrency "$C" 2>&1 | tail -2
done
echo "== $LANE cold prefill ladder $(date '+%H:%M:%S')"
for N in 8000 32000 128000; do
  echo "-- prefill $N"; python3 tools/stress_prefill.py "$BASE/v1" "$MODEL" "$N" 2>&1 | tail -1
done
echo "== $LANE counting ceiling (peak row only) $(date '+%H:%M:%S')"
"$PY" tools/bench_sweep.py "$BASE" "$MODEL" "$LANE" 2>&1 | tail -3
echo "== $LANE done $(date '+%H:%M:%S')"
