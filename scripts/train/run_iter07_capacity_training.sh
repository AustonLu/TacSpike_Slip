#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter07_capacity_training}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

COMMON_ARGS=(
  --model time_channel_scnn
  --data-root "$DATA_ROOT"
  --output-dir "$OUT"
  --num-workers 8
  --class-weight none
  --context-ms 500
  --time-bins 100
  --threshold 1.0
  --scheduler cosine
  --amp
  --sampling random
  --dropout 0.1
  --distill-alpha 0.0
  --ignore-transition-ms 50
)

case "$RUN_ID" in
  iter07_time_channel_w48_h384_ignore50_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --model-width 48 \
      --hidden-dim 384 \
      --batch-size 72 \
      --epochs 10 \
      --train-samples-per-epoch 70000 \
      --val-samples 20000
    ;;
  iter07_time_channel_w64_h512_ignore50_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --model-width 64 \
      --hidden-dim 512 \
      --batch-size 48 \
      --epochs 10 \
      --train-samples-per-epoch 60000 \
      --val-samples 20000
    ;;
  iter07_time_channel_w48_h384_ignore50_long_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --model-width 48 \
      --hidden-dim 384 \
      --batch-size 72 \
      --epochs 15 \
      --train-samples-per-epoch 90000 \
      --val-samples 20000
    ;;
  iter07_time_channel_w48_h384_ignore50_lr5e4_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --model-width 48 \
      --hidden-dim 384 \
      --batch-size 72 \
      --epochs 12 \
      --train-samples-per-epoch 90000 \
      --val-samples 20000 \
      --lr 5e-4
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
