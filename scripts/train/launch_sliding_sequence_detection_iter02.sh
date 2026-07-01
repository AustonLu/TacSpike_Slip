#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter02}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT/$run_id"
  nohup bash scripts/train/run_sliding_sequence_detection_iter02.sh "$run_id" "$gpu_id" \
    > "$LOG_ROOT/$run_id/run.out" 2>&1 < /dev/null &
  echo "$run_id $!"
}

case "${1:-train_sweep}" in
  train_sweep)
    launch_one ctx100_w32_h256_v1 0
    launch_one ctx200_w32_h256_v1 1
    launch_one ctx300_w32_h256_v1 2
    launch_one ctx400_w32_h256_v1 3
    launch_one ctx500_w32_h256_v1 4
    ;;
  eval_sweep)
    launch_one eval_ctx100_w32_h256_v1 0
    launch_one eval_ctx200_w32_h256_v1 1
    launch_one eval_ctx300_w32_h256_v1 2
    launch_one eval_ctx400_w32_h256_v1 3
    launch_one eval_ctx500_w32_h256_v1 4
    ;;
  retrain)
    launch_one retrain_ctx300_w48_h384_v1 0
    launch_one retrain_ctx400_w48_h384_v1 1
    ;;
  eval_retrain)
    launch_one eval_retrain_ctx300_w48_h384_v1 0
    launch_one eval_retrain_ctx400_w48_h384_v1 1
    ;;
  *)
    echo "Usage: $0 [train_sweep|eval_sweep|retrain|eval_retrain]" >&2
    exit 2
    ;;
esac
