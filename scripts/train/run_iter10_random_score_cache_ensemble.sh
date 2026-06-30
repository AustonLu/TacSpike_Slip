#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter10_random_score_cache_ensemble}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT/caches"

cache_one() {
  local name="$1"
  local checkpoint="$2"
  "$PYTHON_BIN" scripts/train/cache_random_window_scores.py \
    --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT" \
    --output-npz "$OUT/caches/${name}.npz" \
    --split val \
    --samples 100000 \
    --sampling random \
    --seed 123 \
    --batch-size 96 \
    --num-workers 8
}

case "$RUN_ID" in
  iter10_core_snn_seed123_random100k)
    cache_one iter04_time_channel_thr1_random_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter04_time_channel_refine/iter04_time_channel_thr1_random_v1/best.pt
    cache_one iter04_time_channel_thr1_random_distill_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter04_time_channel_refine/iter04_time_channel_thr1_random_distill_v1/best.pt
    cache_one iter06_time_channel_random_ignore50_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore50_v1/best.pt
    cache_one iter06_time_channel_random_ignore100_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_v1/best.pt
    cache_one iter06_time_channel_random_ignore150_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore150_v1/best.pt
    cache_one iter06_time_channel_random_ignore100_smooth03_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_smooth03_v1/best.pt
    cache_one iter07_time_channel_w48_h384_ignore50_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter07_capacity_training/iter07_time_channel_w48_h384_ignore50_v1/best.pt
    cache_one iter09_tw_near20_mid100_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near20_mid100_v1/best.pt
    cache_one iter09_tw_near50_mid100_smooth02_v1 \
      /lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near50_mid100_smooth02_v1/best.pt

    "$PYTHON_BIN" scripts/train/search_score_cache_ensemble.py \
      --score-caches "$OUT"/caches/*.npz \
      --output-json "$OUT/ensemble_search_seed123_random100k.json" \
      --search-trials 10000 \
      --top-k 100 \
      --max-subset-size 6 \
      --seed 123
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
