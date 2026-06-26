#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 <run_id>}"

cd /lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2

PY="${PY:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"
DATA="${DATA:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
BASE="${BASE:-/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_explore}"
OUT="$BASE/$RUN_ID"

mkdir -p "$OUT"

common_args=(
  --data-root "$DATA"
  --output-dir "$OUT"
  --num-workers 8
  --device cuda
  --target-accuracy 0.94
)

case "$RUN_ID" in
  lite_longtrain_v1)
    args=(
      --model lite_scnn
      --epochs 15
      --train-samples-per-epoch 200000
      --val-samples 50000
      --batch-size 512
      --threshold 0.1
      --beta 0.85
      --lr 0.001
      --weight-decay 0.0001
      --readout logit_mean
      --sampling balanced
      --class-weight none
      --scheduler cosine
      --amp
    )
    ;;
  lite_scaled_v1)
    args=(
      --model lite_scnn
      --epochs 10
      --train-samples-per-epoch 50000
      --val-samples 20000
      --batch-size 512
      --threshold 0.1
      --beta 0.85
      --lr 0.001
      --weight-decay 0.0001
      --readout logit_mean
      --sampling balanced
      --class-weight none
      --scheduler cosine
      --amp
      --input-scale 4.0
    )
    ;;
  lite_pool2_v1)
    args=(
      --model lite_scnn
      --epochs 10
      --train-samples-per-epoch 50000
      --val-samples 20000
      --batch-size 256
      --threshold 0.1
      --beta 0.85
      --lr 0.001
      --weight-decay 0.0001
      --readout logit_mean
      --sampling balanced
      --class-weight none
      --scheduler cosine
      --amp
      --spatial-pool 2
    )
    ;;
  frame_cnn_sum_v1)
    args=(
      --model frame_cnn
      --epochs 10
      --train-samples-per-epoch 50000
      --val-samples 20000
      --batch-size 512
      --lr 0.001
      --weight-decay 0.0001
      --sampling balanced
      --class-weight none
      --scheduler cosine
      --amp
      --model-width 32
      --temporal-mode sum
    )
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac

exec "$PY" scripts/train/train_lite_scnn.py "${common_args[@]}" "${args[@]}"
