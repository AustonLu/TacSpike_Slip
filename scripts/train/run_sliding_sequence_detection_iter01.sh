#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter01}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

case "$RUN_ID" in
  original20_lite_longtrain_val32)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints /lamport/makkapakka/jiajunlu/logs/tacspike_stage2_explore/lite_longtrain_v1/best.pt \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --split val \
      --max-sequences 32 \
      --batch-size 512 \
      --num-workers 8 \
      --score-transform raw \
      --ma-windows 3,5,10,20,50 \
      --ema-alphas 0.1,0.2,0.4 \
      --debounce-on-k 2,3,5 \
      --debounce-off-k 2,3,5,10
    ;;
  ctx500_best5_val16)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore50_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter07_capacity_training/iter07_time_channel_w48_h384_ignore50_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near20_mid100_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near50_mid100_smooth02_v1/best.pt \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --split val \
      --max-sequences 16 \
      --batch-size 96 \
      --num-workers 8 \
      --score-transform zscore \
      --weights 1,1,1,1,1 \
      --ma-windows 3,5,10,20,50 \
      --ema-alphas 0.1,0.2,0.4 \
      --debounce-on-k 2,3,5 \
      --debounce-off-k 2,3,5,10
    ;;
  ctx500_best5_val16_threshgrid)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore50_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter07_capacity_training/iter07_time_channel_w48_h384_ignore50_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near20_mid100_v1/best.pt \
        /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near50_mid100_smooth02_v1/best.pt \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split val \
      --max-sequences 16 \
      --batch-size 96 \
      --num-workers 8 \
      --score-transform zscore \
      --weights 1,1,1,1,1 \
      --ma-windows 50 \
      --ema-alphas "" \
      --debounce-on-k 2,3,5 \
      --debounce-off-k 2,3,5,10 \
      --debounce-threshold-grid 301
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
