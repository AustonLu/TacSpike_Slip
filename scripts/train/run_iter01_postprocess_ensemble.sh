#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 MODE [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter01_postprocess_ensemble}"
SOURCE_LOG_ROOT="${SOURCE_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_snn_90pct}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$LOG_ROOT"

DISTILL_CKPT="$SOURCE_LOG_ROOT/ctx500_tb100_wide2_scnn_distill_v1/best.pt"
IGNORE50_CKPT="$SOURCE_LOG_ROOT/ctx500_tb100_wide2_scnn_ignore50_v1/best.pt"
DEEP_CKPT="$SOURCE_LOG_ROOT/ctx500_tb100_deep_scnn_distill_v1/best.pt"

case "$MODE" in
  smoothing_distill)
    exec "$PYTHON_BIN" scripts/train/evaluate_sequence_smoothing.py \
      --checkpoint "$DISTILL_CKPT" \
      --data-root "$DATA_ROOT" \
      --output-json "$LOG_ROOT/smoothing_distill_val32.json" \
      --split val \
      --max-sequences 32 \
      --batch-size 128 \
      --num-workers 8 \
      --ma-windows 5,10,20,50,100 \
      --ema-alphas 0.05,0.1,0.2,0.4 \
      --trigger-ks 3,5,10
    ;;
  smoothing_ignore50)
    exec "$PYTHON_BIN" scripts/train/evaluate_sequence_smoothing.py \
      --checkpoint "$IGNORE50_CKPT" \
      --data-root "$DATA_ROOT" \
      --output-json "$LOG_ROOT/smoothing_ignore50_val32.json" \
      --split val \
      --max-sequences 32 \
      --batch-size 128 \
      --num-workers 8 \
      --ma-windows 5,10,20,50,100 \
      --ema-alphas 0.05,0.1,0.2,0.4 \
      --trigger-ks 3,5,10
    ;;
  ensemble2_random)
    exec "$PYTHON_BIN" scripts/train/evaluate_checkpoint_ensemble.py \
      --checkpoints "$DISTILL_CKPT" "$IGNORE50_CKPT" \
      --data-root "$DATA_ROOT" \
      --output-json "$LOG_ROOT/ensemble2_val_random_100k.json" \
      --split val \
      --samples 100000 \
      --batch-size 96 \
      --num-workers 8 \
      --sampling random \
      --best-threshold-metric accuracy
    ;;
  ensemble2_balanced)
    exec "$PYTHON_BIN" scripts/train/evaluate_checkpoint_ensemble.py \
      --checkpoints "$DISTILL_CKPT" "$IGNORE50_CKPT" \
      --data-root "$DATA_ROOT" \
      --output-json "$LOG_ROOT/ensemble2_val_balanced_100k.json" \
      --split val \
      --samples 100000 \
      --batch-size 96 \
      --num-workers 8 \
      --sampling balanced \
      --best-threshold-metric accuracy
    ;;
  ensemble3_random)
    exec "$PYTHON_BIN" scripts/train/evaluate_checkpoint_ensemble.py \
      --checkpoints "$DISTILL_CKPT" "$IGNORE50_CKPT" "$DEEP_CKPT" \
      --data-root "$DATA_ROOT" \
      --output-json "$LOG_ROOT/ensemble3_val_random_100k.json" \
      --split val \
      --samples 100000 \
      --batch-size 96 \
      --num-workers 8 \
      --sampling random \
      --best-threshold-metric accuracy
    ;;
  ensemble3_balanced)
    exec "$PYTHON_BIN" scripts/train/evaluate_checkpoint_ensemble.py \
      --checkpoints "$DISTILL_CKPT" "$IGNORE50_CKPT" "$DEEP_CKPT" \
      --data-root "$DATA_ROOT" \
      --output-json "$LOG_ROOT/ensemble3_val_balanced_100k.json" \
      --split val \
      --samples 100000 \
      --batch-size 96 \
      --num-workers 8 \
      --sampling balanced \
      --best-threshold-metric accuracy
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    exit 2
    ;;
esac
