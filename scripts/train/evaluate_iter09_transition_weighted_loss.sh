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
if [[ ! -f "$OUT/best.pt" ]]; then
  echo "Missing checkpoint: $OUT/best.pt" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/train/evaluate_lite_scnn.py \
  --checkpoint "$OUT/best.pt" \
  --data-root "$DATA_ROOT" \
  --output-json "$OUT/eval_val_random_100k.json" \
  --split val \
  --samples 100000 \
  --batch-size 96 \
  --num-workers 8 \
  --sampling random \
  --best-threshold-metric accuracy

"$PYTHON_BIN" scripts/train/evaluate_lite_scnn.py \
  --checkpoint "$OUT/best.pt" \
  --data-root "$DATA_ROOT" \
  --output-json "$OUT/eval_val_balanced_100k.json" \
  --split val \
  --samples 100000 \
  --batch-size 96 \
  --num-workers 8 \
  --sampling balanced \
  --best-threshold-metric accuracy

"$PYTHON_BIN" scripts/train/evaluate_transition_buckets.py \
  --checkpoint "$OUT/best.pt" \
  --data-root "$DATA_ROOT" \
  --output-json "$OUT/eval_val_random_100k_transition_buckets.json" \
  --split val \
  --samples 100000 \
  --batch-size 96 \
  --num-workers 8 \
  --sampling random
