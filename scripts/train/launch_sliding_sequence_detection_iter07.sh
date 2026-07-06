#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter07}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT/$run_id"
  nohup bash scripts/train/run_sliding_sequence_detection_iter07.sh "$run_id" "$gpu_id" \
    > "$LOG_ROOT/$run_id/run.out" 2>&1 < /dev/null &
  echo "$run_id $!"
}

case "${1:-train_core}" in
  train_core)
    launch_one multiscale_l384_sqrt 0
    launch_one multitau_l384_ignore30 1
    launch_one multiscale_l384_mean 2
    ;;
  train_followup)
    launch_one multiscale_l512_sqrt 3
    launch_one multiscale_multitau_l384 4
    ;;
  eval_core)
    launch_one eval_multiscale_l384_sqrt 0
    launch_one eval_multitau_l384_ignore30 1
    launch_one eval_multiscale_l384_mean 2
    ;;
  eval_followup)
    launch_one eval_multiscale_l512_sqrt 3
    launch_one eval_multiscale_multitau_l384 4
    ;;
  adapter)
    launch_one adapter_iter04_ctx400 5
    launch_one adapter_iter04_seqft 6
    launch_one adapter_iter04_best5 7
    ;;
  adapter_iter07)
    launch_one adapter_iter07_multiscale_l384_sqrt 5
    ;;
  *)
    echo "Usage: $0 [train_core|train_followup|eval_core|eval_followup|adapter|adapter_iter07]" >&2
    exit 2
    ;;
esac
