#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter02_training_objectives}"
SOURCE_LOG_ROOT="${SOURCE_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_snn_90pct}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"
TEACHER_CKPT="${TEACHER_CKPT:-$SOURCE_LOG_ROOT/ctx500_tb100_frame_cnn_teacher_v1/best.pt}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

COMMON_ARGS=(
  --model lite_scnn
  --data-root "$DATA_ROOT"
  --output-dir "$OUT"
  --num-workers 8
  --class-weight none
  --context-ms 500
  --time-bins 100
  --threshold 0.1
  --readout logit_mean
  --scheduler cosine
  --amp
  --batch-size 64
  --scnn-conv1-channels 32
  --scnn-conv2-channels 64
  --scnn-hidden-dim 256
  --readout-start-frac 0.5
)

FULL_ARGS=(
  --epochs 8
  --train-samples-per-epoch 60000
  --val-samples 20000
)

QUICK_ARGS=(
  --epochs 4
  --train-samples-per-epoch 12000
  --val-samples 6000
)

case "$RUN_ID" in
  iter02_random_wide2_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling random
    ;;
  iter02_random_wide2_distill_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --batch-size 56 \
      --sampling random \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  iter02_random_wide2_focal_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling random \
      --loss-type focal \
      --focal-gamma 2.0
    ;;
  iter02_balanced_wide2_focal_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --sampling balanced \
      --loss-type focal \
      --focal-gamma 2.0
    ;;
  iter02_balanced_wide2_margin_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --sampling balanced \
      --margin-loss-weight 0.1 \
      --margin-value 1.0
    ;;
  iter02_balanced_wide2_focal_distill_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --batch-size 56 \
      --sampling balanced \
      --loss-type focal \
      --focal-gamma 2.0 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
