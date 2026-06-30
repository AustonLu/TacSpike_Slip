#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter01_postprocess_ensemble}"
RUNNER="$PROJECT_DIR/scripts/train/run_iter01_postprocess_ensemble.sh"

start_run() {
  local mode="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT"
  if [[ -f "$LOG_ROOT/$mode.pid" ]] && kill -0 "$(cat "$LOG_ROOT/$mode.pid")" 2>/dev/null; then
    echo "$mode already running as pid $(cat "$LOG_ROOT/$mode.pid")"
    return 0
  fi
  nohup bash "$RUNNER" "$mode" "$gpu_id" >"$LOG_ROOT/$mode.out" 2>&1 </dev/null &
  echo "$!" >"$LOG_ROOT/$mode.pid"
  echo "$mode pid $(cat "$LOG_ROOT/$mode.pid") gpu $gpu_id"
}

start_run smoothing_distill 0
start_run smoothing_ignore50 1
start_run ensemble2_random 2
start_run ensemble2_balanced 3
start_run ensemble3_random 4
start_run ensemble3_balanced 5
