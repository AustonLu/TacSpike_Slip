#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter10_random_score_cache_ensemble}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

RUN_ID=iter10_core_snn_seed123_random100k
mkdir -p "$LOG_ROOT/$RUN_ID"
nohup bash scripts/train/run_iter10_random_score_cache_ensemble.sh "$RUN_ID" 0 \
  > "$LOG_ROOT/$RUN_ID/run.out" 2>&1 < /dev/null &
echo "$RUN_ID $!"
