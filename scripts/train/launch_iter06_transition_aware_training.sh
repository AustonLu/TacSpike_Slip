#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training}"
RUNNER="$PROJECT_DIR/scripts/train/run_iter06_transition_aware_training.sh"

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

MODE="${1:-full_grid}"

case "$MODE" in
  full_grid)
    start_run iter06_time_channel_random_ignore50_v1 0
    start_run iter06_time_channel_random_ignore100_v1 1
    start_run iter06_time_channel_random_ignore150_v1 2
    start_run iter06_time_channel_random_ignore100_smooth03_v1 3
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    echo "Usage: $0 [full_grid]" >&2
    exit 2
    ;;
esac
