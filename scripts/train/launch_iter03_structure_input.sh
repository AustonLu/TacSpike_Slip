#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter03_structure_input}"
RUNNER="$PROJECT_DIR/scripts/train/run_iter03_structure_input.sh"

start_run() {
  local run_id="$1"
  local gpu_id="$2"
  local out="$LOG_ROOT/$run_id"
  mkdir -p "$out"
  if [[ -f "$out/pid.txt" ]] && kill -0 "$(cat "$out/pid.txt")" 2>/dev/null; then
    echo "$run_id already running as pid $(cat "$out/pid.txt")"
    return 0
  fi
  nohup "$RUNNER" "$run_id" "$gpu_id" >"$out/train.out" 2>&1 </dev/null &
  echo "$!" >"$out/pid.txt"
  echo "$run_id pid $(cat "$out/pid.txt") gpu $gpu_id"
}

MODE="${1:-quick}"

case "$MODE" in
  quick)
    start_run iter03_time_channel_scnn_quick_v1 0
    start_run iter03_time_channel_scnn_distill_quick_v1 1
    start_run iter03_temporal_conv_scnn_quick_v1 2
    start_run iter03_temporal_conv_scnn_distill_quick_v1 3
    start_run iter03_wide3_scnn_quick_v1 4
    ;;
  stability_quick)
    start_run iter03_time_channel_scnn_thr1_quick_v1 5
    start_run iter03_temporal_conv_scnn_thr1_quick_v1 6
    ;;
  full_time_channel)
    start_run iter03_time_channel_scnn_full_v1 0
    ;;
  full_time_channel_thr1)
    start_run iter03_time_channel_scnn_thr1_full_v1 3
    ;;
  full_temporal_conv)
    start_run iter03_temporal_conv_scnn_full_v1 1
    ;;
  full_temporal_conv_thr1)
    start_run iter03_temporal_conv_scnn_thr1_full_v1 4
    ;;
  full_wide3)
    start_run iter03_wide3_scnn_full_v1 2
    ;;
  full_all)
    start_run iter03_time_channel_scnn_full_v1 0
    start_run iter03_temporal_conv_scnn_full_v1 1
    start_run iter03_wide3_scnn_full_v1 2
    ;;
  full_thr1_candidates)
    start_run iter03_time_channel_scnn_thr1_full_v1 3
    start_run iter03_temporal_conv_scnn_thr1_full_v1 4
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    echo "Usage: $0 [quick|full_time_channel|full_temporal_conv|full_wide3|full_all]" >&2
    exit 2
    ;;
esac
