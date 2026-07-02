#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter05}"
ITER03_LOG_ROOT="${ITER03_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter03}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts/train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

BASE_CKPT="$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt"

train_state_ft() {
  local epochs="$1"
  local segment_windows="$2"
  local train_segments="$3"
  local val_segments="$4"
  local lr="$5"
  local smoothness_weight="$6"
  local flip_penalty_weight="$7"
  local transition_ignore_steps="$8"
  local best_metric="$9"
  exec "$PYTHON_BIN" scripts/train/train_sequence_scnn.py \
    --model time_channel_scnn \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --cache-dir "$LOG_ROOT/cache" \
    --epochs "$epochs" \
    --train-segments-per-epoch "$train_segments" \
    --val-segments "$val_segments" \
    --segment-windows "$segment_windows" \
    --sequence-stride 1 \
    --batch-size 1 \
    --num-workers 8 \
    --sampling transition_mix \
    --class-weight none \
    --context-ms 400 \
    --time-bins 100 \
    --threshold 1.0 \
    --model-width 32 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --smoothness-weight "$smoothness_weight" \
    --flip-penalty-weight "$flip_penalty_weight" \
    --transition-ignore-steps "$transition_ignore_steps" \
    --scheduler cosine \
    --best-metric "$best_metric" \
    --init-checkpoint "$BASE_CKPT" \
    --lr "$lr" \
    --target-accuracy 0.95 \
    --save-epoch-checkpoints \
    --amp
}

train_state_ft_workers() {
  local epochs="$1"
  local segment_windows="$2"
  local train_segments="$3"
  local val_segments="$4"
  local lr="$5"
  local smoothness_weight="$6"
  local flip_penalty_weight="$7"
  local transition_ignore_steps="$8"
  local best_metric="$9"
  local num_workers="${10}"
  exec "$PYTHON_BIN" scripts/train/train_sequence_scnn.py \
    --model time_channel_scnn \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --cache-dir "$LOG_ROOT/cache" \
    --epochs "$epochs" \
    --train-segments-per-epoch "$train_segments" \
    --val-segments "$val_segments" \
    --segment-windows "$segment_windows" \
    --sequence-stride 1 \
    --batch-size 1 \
    --num-workers "$num_workers" \
    --sampling transition_mix \
    --class-weight none \
    --context-ms 400 \
    --time-bins 100 \
    --threshold 1.0 \
    --model-width 32 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --smoothness-weight "$smoothness_weight" \
    --flip-penalty-weight "$flip_penalty_weight" \
    --transition-ignore-steps "$transition_ignore_steps" \
    --scheduler cosine \
    --best-metric "$best_metric" \
    --init-checkpoint "$BASE_CKPT" \
    --lr "$lr" \
    --target-accuracy 0.95 \
    --save-epoch-checkpoints \
    --amp
}

train_state_ft_legacy() {
  local segment_windows="$1"
  local train_segments="$2"
  local val_segments="$3"
  local lr="$4"
  local smoothness_weight="$5"
  local flip_penalty_weight="$6"
  local transition_ignore_steps="$7"
  local best_metric="$8"
  exec "$PYTHON_BIN" scripts/train/train_sequence_scnn.py \
    --model time_channel_scnn \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --cache-dir "$LOG_ROOT/cache" \
    --epochs 8 \
    --train-segments-per-epoch "$train_segments" \
    --val-segments "$val_segments" \
    --segment-windows "$segment_windows" \
    --sequence-stride 1 \
    --batch-size 1 \
    --num-workers 8 \
    --sampling transition_mix \
    --class-weight none \
    --context-ms 400 \
    --time-bins 100 \
    --threshold 1.0 \
    --model-width 32 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --smoothness-weight "$smoothness_weight" \
    --flip-penalty-weight "$flip_penalty_weight" \
    --transition-ignore-steps "$transition_ignore_steps" \
    --scheduler cosine \
    --best-metric "$best_metric" \
    --init-checkpoint "$BASE_CKPT" \
    --lr "$lr" \
    --target-accuracy 0.95 \
    --save-epoch-checkpoints \
    --amp
}

eval_checkpoint() {
  local checkpoint="$1"
  exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
    --checkpoints "$checkpoint" \
    --data-root "$DATA_ROOT" \
    --output-json "$OUT/sliding_detection.json" \
    --output-score-cache "$OUT/score_cache.npz" \
    --split val \
    --max-sequences 16 \
    --batch-size 128 \
    --num-workers 8 \
    --score-transform raw \
    --ma-windows 50,80,100,150 \
    --ema-alphas 0.02,0.05,0.1 \
    --debounce-on-k 2,3,5,8 \
    --debounce-off-k 10,20,30,50 \
    --debounce-threshold-grid 24
}

case "$RUN_ID" in
  ft_ctx400_seg512_no_smooth)
    train_state_ft 8 512 1200 400 0.00001 0.0 0.0 0 valid_balanced_accuracy
    ;;
  ft_ctx400_seg512_low_smooth)
    train_state_ft 8 512 1200 400 0.00001 0.0001 0.0 0 valid_balanced_accuracy
    ;;
  ft_ctx400_seg1024_no_smooth)
    train_state_ft 8 1024 600 200 0.00001 0.0 0.0 0 valid_balanced_accuracy
    ;;
  ft_ctx400_seg1024_ignore50_no_smooth)
    train_state_ft 8 1024 600 200 0.00001 0.0 0.0 50 valid_balanced_accuracy
    ;;
  probe_seg512_no_smooth)
    train_state_ft_workers 2 512 160 80 0.000005 0.0 0.0 0 valid_balanced_accuracy 2
    ;;
  probe_seg1024_no_smooth)
    train_state_ft_workers 2 1024 80 40 0.000005 0.0 0.0 0 valid_balanced_accuracy 2
    ;;
  eval_probe_seg512_no_smooth_best)
    eval_checkpoint "$LOG_ROOT/probe_seg512_no_smooth/best.pt"
    ;;
  eval_probe_seg1024_no_smooth_best)
    eval_checkpoint "$LOG_ROOT/probe_seg1024_no_smooth/best.pt"
    ;;
  eval_ft_ctx400_seg512_no_smooth_best)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg512_no_smooth/best.pt"
    ;;
  eval_ft_ctx400_seg512_low_smooth_best)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg512_low_smooth/best.pt"
    ;;
  eval_ft_ctx400_seg1024_no_smooth_best)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg1024_no_smooth/best.pt"
    ;;
  eval_ft_ctx400_seg1024_ignore50_no_smooth_best)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg1024_ignore50_no_smooth/best.pt"
    ;;
  eval_ft_ctx400_seg512_no_smooth_epoch_*)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg512_no_smooth/${RUN_ID#eval_ft_ctx400_seg512_no_smooth_}.pt"
    ;;
  eval_ft_ctx400_seg512_low_smooth_epoch_*)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg512_low_smooth/${RUN_ID#eval_ft_ctx400_seg512_low_smooth_}.pt"
    ;;
  eval_ft_ctx400_seg1024_no_smooth_epoch_*)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg1024_no_smooth/${RUN_ID#eval_ft_ctx400_seg1024_no_smooth_}.pt"
    ;;
  eval_ft_ctx400_seg1024_ignore50_no_smooth_epoch_*)
    eval_checkpoint "$LOG_ROOT/ft_ctx400_seg1024_ignore50_no_smooth/${RUN_ID#eval_ft_ctx400_seg1024_ignore50_no_smooth_}.pt"
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
