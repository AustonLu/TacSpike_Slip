#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter02_training_objectives}"
RUNNER="$PROJECT_DIR/scripts/train/run_iter02_training_objectives.sh"

start_run() {
  local run_id="$1"
  local gpu_id="$2"
  local out="$LOG_ROOT/$run_id"
  mkdir -p "$out"
  if [[ -f "$out/pid.txt" ]] && kill -0 "$(cat "$out/pid.txt")" 2>/dev/null; then
    echo "$run_id already running as pid $(cat "$out/pid.txt")"
    return 0
  fi
  nohup bash "$RUNNER" "$run_id" "$gpu_id" >"$out/train.out" 2>&1 </dev/null &
  echo "$!" >"$out/pid.txt"
  echo "$run_id pid $(cat "$out/pid.txt") gpu $gpu_id"
}

start_run iter02_random_wide2_v1 0
start_run iter02_random_wide2_distill_v1 1
start_run iter02_random_wide2_focal_v1 2
start_run iter02_balanced_wide2_focal_quick_v1 3
start_run iter02_balanced_wide2_margin_quick_v1 4
start_run iter02_balanced_wide2_focal_distill_quick_v1 5
