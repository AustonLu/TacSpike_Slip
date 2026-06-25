#!/usr/bin/env bash
set -euo pipefail

cd /lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PY=/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python
DATA=/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0
OUT=/lamport/makkapakka/jiajunlu/logs/tacspike_stage2/main_lite_scnn_v1_random_weighted

mkdir -p "$OUT"

exec "$PY" scripts/train/train_lite_scnn.py \
  --data-root "$DATA" \
  --output-dir "$OUT" \
  --epochs 10 \
  --train-samples-per-epoch 50000 \
  --val-samples 20000 \
  --batch-size 512 \
  --num-workers 8 \
  --device cuda \
  --threshold 0.1 \
  --beta 0.85 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --readout membrane \
  --sampling random \
  --class-weight inverse_frequency
