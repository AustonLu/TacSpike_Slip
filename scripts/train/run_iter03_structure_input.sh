#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter03_structure_input}"
SOURCE_LOG_ROOT="${SOURCE_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_snn_90pct}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"
TEACHER_CKPT="${TEACHER_CKPT:-$SOURCE_LOG_ROOT/ctx500_tb100_frame_cnn_teacher_v1/best.pt}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --output-dir "$OUT"
  --num-workers 8
  --class-weight none
  --context-ms 500
  --time-bins 100
  --threshold 0.1
  --scheduler cosine
  --amp
)

QUICK_ARGS=(
  --epochs 4
  --train-samples-per-epoch 16000
  --val-samples 8000
)

FULL_ARGS=(
  --epochs 10
  --train-samples-per-epoch 70000
  --val-samples 20000
)

case "$RUN_ID" in
  iter03_time_channel_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --model time_channel_scnn \
      --model-width 32 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 96 \
      --sampling balanced
    ;;
  iter03_time_channel_scnn_distill_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --model time_channel_scnn \
      --model-width 32 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 96 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  iter03_time_channel_scnn_thr1_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --threshold 1.0 \
      --model time_channel_scnn \
      --model-width 32 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 96 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  iter03_temporal_conv_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --model temporal_conv_scnn \
      --model-width 16 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 64 \
      --sampling balanced
    ;;
  iter03_temporal_conv_scnn_distill_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --model temporal_conv_scnn \
      --model-width 16 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 64 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  iter03_temporal_conv_scnn_thr1_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --threshold 1.0 \
      --model temporal_conv_scnn \
      --model-width 16 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 64 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.5 \
      --distill-temperature 2.0
    ;;
  iter03_wide3_scnn_quick_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${QUICK_ARGS[@]}" \
      --model lite_scnn \
      --readout logit_mean \
      --scnn-conv1-channels 48 \
      --scnn-conv2-channels 96 \
      --scnn-hidden-dim 384 \
      --readout-start-frac 0.5 \
      --batch-size 40 \
      --sampling balanced
    ;;
  iter03_time_channel_scnn_full_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --model time_channel_scnn \
      --model-width 32 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 96 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  iter03_time_channel_scnn_thr1_full_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --threshold 1.0 \
      --model time_channel_scnn \
      --model-width 32 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 96 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  iter03_temporal_conv_scnn_full_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --model temporal_conv_scnn \
      --model-width 16 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 64 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  iter03_temporal_conv_scnn_thr1_full_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --threshold 1.0 \
      --model temporal_conv_scnn \
      --model-width 16 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --batch-size 64 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  iter03_wide3_scnn_full_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      "${COMMON_ARGS[@]}" \
      "${FULL_ARGS[@]}" \
      --model lite_scnn \
      --readout logit_mean \
      --scnn-conv1-channels 48 \
      --scnn-conv2-channels 96 \
      --scnn-hidden-dim 384 \
      --readout-start-frac 0.5 \
      --batch-size 40 \
      --sampling balanced \
      --teacher-checkpoint "$TEACHER_CKPT" \
      --distill-alpha 0.3 \
      --distill-temperature 2.0
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
