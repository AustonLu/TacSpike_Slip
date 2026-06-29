#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_temporal_streaming}"
RUNNER="$PROJECT_DIR/scripts/train/run_stage2_temporal_streaming_exploration.sh"

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

start_run ctx300_frame_cnn_v1 0
start_run ctx500_frame_cnn_v1 1
start_run ctx300_lite_scnn_v1 2
start_run stream_lite_t256_last_v1 3
