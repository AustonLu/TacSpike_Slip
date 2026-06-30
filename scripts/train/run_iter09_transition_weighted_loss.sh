#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss}"
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
  --model-width 32
  --hidden-dim 256
  --scheduler cosine
  --amp
  --batch-size 96
  --epochs 10
  --train-samples-per-epoch 70000
  --val-samples 20000
  --sampling random
  --dropout 0.1
  --distill-alpha 0.0
)

case "$RUN_ID" in
  iter09_tw_near20_mid50_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --transition-weight-near-ms 20 \
      --transition-weight-mid-ms 50 \
      --transition-near-weight 0.25 \
      --transition-mid-weight 0.60
    ;;
  iter09_tw_near50_mid100_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --transition-weight-near-ms 50 \
      --transition-weight-mid-ms 100 \
      --transition-near-weight 0.35 \
      --transition-mid-weight 0.70
    ;;
  iter09_tw_near20_mid100_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --transition-weight-near-ms 20 \
      --transition-weight-mid-ms 100 \
      --transition-near-weight 0.20 \
      --transition-mid-weight 0.70
    ;;
  iter09_tw_near50_mid100_smooth02_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      --transition-weight-near-ms 50 \
      --transition-weight-mid-ms 100 \
      --transition-near-weight 0.35 \
      --transition-mid-weight 0.70 \
      --label-smoothing 0.02
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
