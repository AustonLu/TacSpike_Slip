#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter05}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT/$run_id"
  nohup bash scripts/train/run_sliding_sequence_detection_iter05.sh "$run_id" "$gpu_id" \
    > "$LOG_ROOT/$run_id/run.out" 2>&1 < /dev/null &
  echo "$run_id $!"
}

case "${1:-train}" in
  train)
    launch_one ft_ctx400_seg512_no_smooth 0
    launch_one ft_ctx400_seg512_low_smooth 1
    launch_one ft_ctx400_seg1024_no_smooth 2
    launch_one ft_ctx400_seg1024_ignore50_no_smooth 3
    ;;
  eval_best)
    launch_one eval_ft_ctx400_seg512_no_smooth_best 0
    launch_one eval_ft_ctx400_seg512_low_smooth_best 1
    launch_one eval_ft_ctx400_seg1024_no_smooth_best 2
    launch_one eval_ft_ctx400_seg1024_ignore50_no_smooth_best 3
    ;;
  probe)
    launch_one probe_seg512_no_smooth 4
    launch_one probe_seg1024_no_smooth 5
    ;;
  eval_probe)
    launch_one eval_probe_seg512_no_smooth_best 4
    launch_one eval_probe_seg1024_no_smooth_best 5
    ;;
  eval_512_epochs)
    launch_one eval_ft_ctx400_seg512_no_smooth_epoch_001 0
    launch_one eval_ft_ctx400_seg512_no_smooth_epoch_002 1
    launch_one eval_ft_ctx400_seg512_no_smooth_epoch_003 2
    launch_one eval_ft_ctx400_seg512_no_smooth_epoch_004 3
    ;;
  eval_512_low_epochs)
    launch_one eval_ft_ctx400_seg512_low_smooth_epoch_001 0
    launch_one eval_ft_ctx400_seg512_low_smooth_epoch_002 1
    launch_one eval_ft_ctx400_seg512_low_smooth_epoch_003 2
    launch_one eval_ft_ctx400_seg512_low_smooth_epoch_004 3
    ;;
  eval_1024_epochs)
    launch_one eval_ft_ctx400_seg1024_no_smooth_epoch_001 0
    launch_one eval_ft_ctx400_seg1024_no_smooth_epoch_002 1
    launch_one eval_ft_ctx400_seg1024_no_smooth_epoch_003 2
    launch_one eval_ft_ctx400_seg1024_no_smooth_epoch_004 3
    ;;
  *)
    echo "Usage: $0 [train|probe|eval_probe|eval_best|eval_512_epochs|eval_512_low_epochs|eval_1024_epochs]" >&2
    exit 2
    ;;
esac
