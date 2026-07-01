#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter03}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT/$run_id"
  nohup bash scripts/train/run_sliding_sequence_detection_iter03.sh "$run_id" "$gpu_id" \
    > "$LOG_ROOT/$run_id/run.out" 2>&1 < /dev/null &
  echo "$run_id $!"
}

case "${1:-train_stream}" in
  train_stream)
    launch_one stream_t400_all_ignore25_smooth_v1 0
    launch_one stream_t400_tail200_ignore25_smooth_v1 1
    launch_one stream_t512_tail256_ignore50_smooth_v1 2
    ;;
  eval_stream)
    launch_one eval_stream_t400_all_ignore25_smooth_v1 0
    launch_one eval_stream_t400_tail200_ignore25_smooth_v1 1
    launch_one eval_stream_t512_tail256_ignore50_smooth_v1 2
    ;;
  train_sequence)
    launch_one seq_ctx400_s32_transition_mix_smooth_v1 3
    launch_one seq_ctx400_s64_transition_mix_smooth_v1 4
    launch_one seq_ctx400_s32_end_bal_ignore4_v1 5
    ;;
  eval_sequence)
    launch_one eval_seq_ctx400_s32_transition_mix_smooth_v1 3
    launch_one eval_seq_ctx400_s64_transition_mix_smooth_v1 4
    launch_one eval_seq_ctx400_s32_end_bal_ignore4_v1 5
    ;;
  train_sequence_ft)
    launch_one seq_ft_ctx400_s32_transition_mix_lr1e4_v1 3
    launch_one seq_ft_ctx400_s32_random_lr5e5_v1 4
    launch_one seq_ft_ctx400_s32_tail16_ignore4_lr1e4_v1 5
    ;;
  eval_sequence_ft)
    launch_one eval_seq_ft_ctx400_s32_transition_mix_lr1e4_v1 3
    launch_one eval_seq_ft_ctx400_s32_random_lr5e5_v1 4
    launch_one eval_seq_ft_ctx400_s32_tail16_ignore4_lr1e4_v1 5
    ;;
  *)
    echo "Usage: $0 [train_stream|eval_stream|train_sequence|eval_sequence|train_sequence_ft|eval_sequence_ft]" >&2
    exit 2
    ;;
esac
