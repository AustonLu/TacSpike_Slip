#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter02}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

train_context() {
  local context_ms="$1"
  local time_bins="$2"
  shift 2
  exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
    --model time_channel_scnn \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --num-workers 8 \
    --class-weight none \
    --context-ms "$context_ms" \
    --time-bins "$time_bins" \
    --threshold 1.0 \
    --model-width 32 \
    --hidden-dim 256 \
    --scheduler cosine \
    --amp \
    --batch-size 96 \
    --epochs 8 \
    --train-samples-per-epoch 60000 \
    --val-samples 20000 \
    --sampling random \
    --dropout 0.1 \
    --ignore-transition-ms 50 \
    "$@"
}

eval_context() {
  local train_run="$1"
  local eval_name="$2"
  local checkpoint="$LOG_ROOT/$train_run/best.pt"
  local eval_out="$OUT/$eval_name"
  mkdir -p "$eval_out"
  exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
    --checkpoints "$checkpoint" \
    --data-root "$DATA_ROOT" \
    --output-json "$eval_out/sliding_detection.json" \
    --output-score-cache "$eval_out/score_cache.npz" \
    --split val \
    --max-sequences 16 \
    --batch-size 128 \
    --num-workers 8 \
    --score-transform raw \
    --ma-windows 3,5,10,20,50 \
    --ema-alphas 0.1,0.2,0.4 \
    --debounce-on-k 2,3,5 \
    --debounce-off-k 2,3,5,10 \
    --debounce-threshold-grid 1
}

case "$RUN_ID" in
  ctx100_w32_h256_v1)
    train_context 100 50
    ;;
  ctx200_w32_h256_v1)
    train_context 200 50
    ;;
  ctx300_w32_h256_v1)
    train_context 300 75
    ;;
  ctx400_w32_h256_v1)
    train_context 400 100
    ;;
  ctx500_w32_h256_v1)
    train_context 500 100
    ;;
  eval_ctx100_w32_h256_v1)
    eval_context ctx100_w32_h256_v1 ctx100
    ;;
  eval_ctx200_w32_h256_v1)
    eval_context ctx200_w32_h256_v1 ctx200
    ;;
  eval_ctx300_w32_h256_v1)
    eval_context ctx300_w32_h256_v1 ctx300
    ;;
  eval_ctx400_w32_h256_v1)
    eval_context ctx400_w32_h256_v1 ctx400
    ;;
  eval_ctx500_w32_h256_v1)
    eval_context ctx500_w32_h256_v1 ctx500
    ;;
  retrain_ctx300_w48_h384_v1)
    train_context 300 75 \
      --model-width 48 \
      --hidden-dim 384 \
      --batch-size 72 \
      --epochs 14 \
      --train-samples-per-epoch 90000 \
      --label-smoothing 0.02 \
      --margin-loss-weight 0.02 \
      --margin-value 1.0
    ;;
  retrain_ctx400_w48_h384_v1)
    train_context 400 100 \
      --model-width 48 \
      --hidden-dim 384 \
      --batch-size 72 \
      --epochs 14 \
      --train-samples-per-epoch 90000 \
      --label-smoothing 0.02 \
      --margin-loss-weight 0.02 \
      --margin-value 1.0
    ;;
  eval_retrain_ctx300_w48_h384_v1)
    eval_context retrain_ctx300_w48_h384_v1 retrain_ctx300
    ;;
  eval_retrain_ctx400_w48_h384_v1)
    eval_context retrain_ctx400_w48_h384_v1 retrain_ctx400
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
