#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter04_time_channel_refine}"
SOURCE_LOG_ROOT="${SOURCE_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_snn_90pct}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"
TEACHER_CKPT="${TEACHER_CKPT:-$SOURCE_LOG_ROOT/ctx500_tb100_frame_cnn_teacher_v1/best.pt}"

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
)

FULL_ARGS=(
  --epochs 10
  --train-samples-per-epoch 70000
  --val-samples 20000
)

case "$RUN_ID" in
  iter04_time_channel_thr1_random_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling random \
      --dropout 0.1 \
      --distill-alpha 0.0
    ;;
  iter04_time_channel_thr1_random_distill_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling random \
      --dropout 0.1 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  iter04_time_channel_thr1_alpha01_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling balanced \
      --dropout 0.1 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.1 \
      --distill-temperature 2.0
    ;;
  iter04_time_channel_thr1_alpha05_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling balanced \
      --dropout 0.1 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  iter04_time_channel_thr1_dropout02_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling balanced \
      --dropout 0.2 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  iter04_time_channel_thr1_smooth03_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --sampling balanced \
      --dropout 0.1 \
      --label-smoothing 0.03 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
