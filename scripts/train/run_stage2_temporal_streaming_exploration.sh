#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_temporal_streaming}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

case "$RUN_ID" in
  ctx300_frame_cnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      --model frame_cnn \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT" \
      --epochs 6 \
      --train-samples-per-epoch 50000 \
      --val-samples 20000 \
      --batch-size 192 \
      --num-workers 8 \
      --sampling balanced \
      --class-weight none \
      --context-ms 300 \
      --time-bins 300 \
      --model-width 32 \
      --scheduler cosine \
      --amp
    ;;
  ctx500_frame_cnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      --model frame_cnn \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT" \
      --epochs 6 \
      --train-samples-per-epoch 50000 \
      --val-samples 20000 \
      --batch-size 128 \
      --num-workers 8 \
      --sampling balanced \
      --class-weight none \
      --context-ms 500 \
      --time-bins 500 \
      --model-width 32 \
      --scheduler cosine \
      --amp
    ;;
  ctx1000_frame_cnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      --model frame_cnn \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT" \
      --epochs 6 \
      --train-samples-per-epoch 50000 \
      --val-samples 20000 \
      --batch-size 64 \
      --num-workers 8 \
      --sampling balanced \
      --class-weight none \
      --context-ms 1000 \
      --time-bins 1000 \
      --model-width 32 \
      --scheduler cosine \
      --amp
    ;;
  ctx300_lite_scnn_v1)
    exec "$PYTHON_BIN" scripts/train/train_lite_scnn.py \
      --model lite_scnn \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT" \
      --epochs 6 \
      --train-samples-per-epoch 50000 \
      --val-samples 20000 \
      --batch-size 96 \
      --num-workers 8 \
      --sampling balanced \
      --class-weight none \
      --context-ms 300 \
      --time-bins 300 \
      --threshold 0.1 \
      --readout logit_mean \
      --scheduler cosine \
      --amp
    ;;
  stream_lite_t256_last_v1)
    exec "$PYTHON_BIN" scripts/train/train_streaming_scnn.py \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT" \
      --epochs 6 \
      --train-segments-per-epoch 30000 \
      --val-segments 10000 \
      --segment-steps 256 \
      --batch-size 128 \
      --num-workers 8 \
      --sampling balanced \
      --loss-mode last \
      --threshold 0.1 \
      --hidden-dim 64 \
      --scheduler cosine \
      --amp
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
