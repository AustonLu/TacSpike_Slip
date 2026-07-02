#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter06}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT/$run_id"
  nohup bash scripts/train/run_sliding_sequence_detection_iter06.sh "$run_id" "$gpu_id" \
    > "$LOG_ROOT/$run_id/run.out" 2>&1 < /dev/null &
  echo "$run_id $!"
}

case "${1:-probe}" in
  build_cache)
    launch_one build_cache 0
    ;;
  probe)
    launch_one sanity_probe_l128 0
    ;;
  eval_probe)
    launch_one eval_sanity_probe_l128 0
    ;;
  train_main)
    launch_one stream_l384_wide 0
    launch_one stream_l256_wide 1
    launch_one stream_l512_wide 2
    launch_one stream_l384_ignore30 3
    ;;
  train_large)
    launch_one stream_l384_large 4
    ;;
  eval_main)
    launch_one eval_stream_l384_wide 0
    launch_one eval_stream_l256_wide 1
    launch_one eval_stream_l512_wide 2
    launch_one eval_stream_l384_ignore30 3
    ;;
  eval_large)
    launch_one eval_stream_l384_large 4
    ;;
  *)
    echo "Usage: $0 [build_cache|probe|eval_probe|train_main|train_large|eval_main|eval_large]" >&2
    exit 2
    ;;
esac
