#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_snn_90pct}"
RUNNER="$PROJECT_DIR/scripts/train/run_stage2_snn_90pct_exploration.sh"

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

MODE="${1:-tb100_quick}"

case "$MODE" in
  tb500_quick)
    start_run ctx500_lite_scnn_quick_v1 0
    start_run ctx500_lite_scnn_tail_quick_v1 1
    start_run ctx500_wide_scnn_tail_quick_v1 2
    start_run ctx500_wide_scnn_distill_quick_v1 3
    ;;
  tb100_quick)
    start_run ctx500_tb100_lite_scnn_quick_v1 0
    start_run ctx500_tb100_wide_scnn_quick_v1 1
    start_run ctx500_tb100_wide2_scnn_quick_v1 2
    start_run ctx500_tb100_wide2_scnn_smooth_quick_v1 3
    start_run ctx500_tb100_wide2_scnn_ignore50_quick_v1 4
    start_run ctx500_tb100_frame_cnn_teacher_v1 5
    ;;
  distill_quick)
    start_run ctx500_tb100_wide2_scnn_distill_quick_v1 0
    ;;
  full_best)
    start_run ctx500_tb100_wide2_scnn_v1 0
    ;;
  full_candidates)
    start_run ctx500_tb100_wide2_scnn_ignore50_v1 1
    start_run ctx500_tb100_wide2_scnn_distill_v1 2
    ;;
  deep_quick)
    start_run ctx500_tb100_deep_scnn_quick_v1 3
    start_run ctx500_tb100_deep_scnn_ignore50_quick_v1 4
    start_run ctx500_tb100_deep_scnn_distill_quick_v1 5
    ;;
  deep_full)
    start_run ctx500_tb100_deep_scnn_distill_v1 3
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    echo "Usage: $0 [tb500_quick|tb100_quick|distill_quick|full_best|full_candidates|deep_quick|deep_full]" >&2
    exit 2
    ;;
esac
