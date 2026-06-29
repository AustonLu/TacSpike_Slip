#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_snn_90pct}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"
TEACHER_CKPT="${TEACHER_CKPT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_temporal_streaming/ctx500_frame_cnn_v1/best.pt}"
TB100_TEACHER_CKPT="${TB100_TEACHER_CKPT:-$LOG_ROOT/ctx500_tb100_frame_cnn_teacher_v1/best.pt}"

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
  --sampling balanced
  --class-weight none
  --context-ms 500
  --threshold 0.1
  --readout logit_mean
  --scheduler cosine
  --amp
)

FULL_ARGS=(
  --epochs 8
  --train-samples-per-epoch 60000
  --val-samples 20000
)

QUICK_ARGS=(
  --epochs 4
  --train-samples-per-epoch 10000
  --val-samples 5000
)

TB500_ARGS=(--time-bins 500)
TB100_ARGS=(--time-bins 100)

WIDE_ARGS=(
  --scnn-conv1-channels 24
  --scnn-conv2-channels 48
  --scnn-hidden-dim 128
)

WIDE2_ARGS=(
  --scnn-conv1-channels 32
  --scnn-conv2-channels 64
  --scnn-hidden-dim 256
)

DEEP_ARGS=(
  --model deep_scnn
  --model-width 32
  --hidden-dim 256
)

case "$RUN_ID" in
  ctx500_lite_scnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 64
    ;;
  ctx500_lite_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 64
    ;;
  ctx500_lite_scnn_tail_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 64 \
      --readout-start-frac 0.5
    ;;
  ctx500_lite_scnn_tail_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 64 \
      --readout-start-frac 0.5
    ;;
  ctx500_wide_scnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 48 \
      "${WIDE_ARGS[@]}"
    ;;
  ctx500_wide_scnn_tail_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 48 \
      "${WIDE_ARGS[@]}" \
      --readout-start-frac 0.5
    ;;
  ctx500_wide_scnn_tail_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 48 \
      "${WIDE_ARGS[@]}" \
      --readout-start-frac 0.5
    ;;
  ctx500_wide_scnn_distill_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 40 \
      "${WIDE_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  ctx500_wide_scnn_distill_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 40 \
      "${WIDE_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  ctx500_wide_scnn_ignore50_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB500_ARGS[@]}" \
      --batch-size 48 \
      "${WIDE_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --ignore-transition-ms 50
    ;;
  ctx500_tb100_frame_cnn_teacher_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --model frame_cnn \
      --batch-size 96 \
      --model-width 32 \
      --dropout 0.15
    ;;
  ctx500_tb100_lite_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 128
    ;;
  ctx500_tb100_wide_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 96 \
      "${WIDE_ARGS[@]}" \
      --readout-start-frac 0.5
    ;;
  ctx500_tb100_wide2_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 64 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5
    ;;
  ctx500_tb100_wide2_scnn_smooth_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 64 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --label-smoothing 0.05
    ;;
  ctx500_tb100_wide2_scnn_ignore50_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 64 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --ignore-transition-ms 50
    ;;
  ctx500_tb100_wide2_scnn_ignore50_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 64 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --ignore-transition-ms 50
    ;;
  ctx500_tb100_wide2_scnn_distill_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 56 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --teacher-checkpoint "$TB100_TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  ctx500_tb100_wide2_scnn_distill_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 56 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5 \
      --teacher-checkpoint "$TB100_TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  ctx500_tb100_wide2_scnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      --batch-size 64 \
      "${WIDE2_ARGS[@]}" \
      --readout-start-frac 0.5
    ;;
  ctx500_tb100_deep_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      "${DEEP_ARGS[@]}" \
      --batch-size 64
    ;;
  ctx500_tb100_deep_scnn_ignore50_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      "${DEEP_ARGS[@]}" \
      --batch-size 64 \
      --ignore-transition-ms 50
    ;;
  ctx500_tb100_deep_scnn_distill_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      "${DEEP_ARGS[@]}" \
      --batch-size 56 \
      --teacher-checkpoint "$TB100_TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  ctx500_tb100_deep_scnn_distill_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      "${TB100_ARGS[@]}" \
      "${DEEP_ARGS[@]}" \
      --batch-size 56 \
      --teacher-checkpoint "$TB100_TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
