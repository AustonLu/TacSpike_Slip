#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter04}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  mkdir -p "$LOG_ROOT/$run_id"
  nohup bash scripts/train/run_sliding_sequence_detection_iter04.sh "$run_id" "$gpu_id" \
    > "$LOG_ROOT/$run_id/run.out" 2>&1 < /dev/null &
  echo "$run_id $!"
}

case "${1:-all}" in
  cache)
    launch_one cache_ctx400_train 0
    launch_one cache_ctx400_val 1
    launch_one cache_seqft_train 2
    launch_one cache_seqft_val 3
    ;;
  audit)
    launch_one audit_val_labels 0
    launch_one postprocess_ctx400_seqft 1
    ;;
  subset_cache)
    launch_one subset_ctx400_train64 0
    launch_one subset_seqft_train64 1
    launch_one subset_ctx400_val16 2
    launch_one subset_seqft_val16 3
    ;;
  subset_cache_fast)
    launch_one subset_ctx400_train16_fast 0
    launch_one subset_seqft_train16_fast 1
    launch_one subset_ctx400_val16_fast 2
    launch_one subset_seqft_val16_fast 3
    ;;
  best5_cache_fast)
    launch_one subset_best5_train16_fast 0
    launch_one subset_best5_val16_fast 1
    ;;
  best5_cache_full_val)
    launch_one subset_best5_val16_full 1
    ;;
  state_head)
    launch_one state_head_ctx400_seqft 0
    launch_one state_head_ctx400_seqft_small 1
    ;;
  state_head_subset)
    launch_one state_head_subset_ctx400_seqft 0
    ;;
  state_head_subset_fast)
    launch_one state_head_subset_fast_ctx400_seqft 0
    ;;
  state_head_subset_fast_regularized)
    launch_one state_head_subset_fast_regularized 1
    ;;
  state_head_subset_fast_ctx400_seqft_causal_v2)
    launch_one state_head_subset_fast_ctx400_seqft_causal_v2 0
    ;;
  state_head_subset_fast_all3)
    launch_one state_head_subset_fast_all3 2
    ;;
  state_head_subset_fast_all3_v2)
    launch_one state_head_subset_fast_all3_v2 2
    ;;
  state_head_subset_fast_all3_causal_v2)
    launch_one state_head_subset_fast_all3_causal_v2 2
    ;;
  finetune)
    launch_one seq_ft_ctx400_state_loss_v2 2
    ;;
  eval_finetune)
    launch_one eval_seq_ft_ctx400_state_loss_v2 2
    ;;
  all)
    "$0" cache
    ;;
  *)
    echo "Usage: $0 [cache|audit|subset_cache|subset_cache_fast|best5_cache_fast|best5_cache_full_val|state_head|state_head_subset|state_head_subset_fast|state_head_subset_fast_regularized|state_head_subset_fast_ctx400_seqft_causal_v2|state_head_subset_fast_all3|state_head_subset_fast_all3_v2|state_head_subset_fast_all3_causal_v2|finetune|eval_finetune|all]" >&2
    exit 2
    ;;
esac
