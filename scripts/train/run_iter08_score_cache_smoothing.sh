#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cache}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter08_score_cache_smoothing}"
ITER06_ROOT="${ITER06_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter06_transition_aware_training}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$LOG_ROOT/cache" "$LOG_ROOT/search"

cache_one() {
  local run_id="$1"
  local cache_name="$2"
  "$PYTHON_BIN" scripts/train/cache_sequence_scores.py \
    --checkpoint "$ITER06_ROOT/$run_id/best.pt" \
    --data-root "$DATA_ROOT" \
    --output-npz "$LOG_ROOT/cache/$cache_name" \
    --split val \
    --batch-size 128 \
    --num-workers 2
}

search_one() {
  local output_name="$1"
  shift
  "$PYTHON_BIN" scripts/train/evaluate_score_cache_smoothing.py \
    --score-caches "$@" \
    --output-json "$LOG_ROOT/search/$output_name" \
    --transform raw
}

case "$MODE" in
  cache_ignore50)
    cache_one iter06_time_channel_random_ignore50_v1 iter06_ignore50_val_scores.npz
    ;;
  cache_ignore100)
    cache_one iter06_time_channel_random_ignore100_v1 iter06_ignore100_val_scores.npz
    ;;
  cache_ignore150)
    cache_one iter06_time_channel_random_ignore150_v1 iter06_ignore150_val_scores.npz
    ;;
  cache_ignore100_smooth03)
    cache_one iter06_time_channel_random_ignore100_smooth03_v1 iter06_ignore100_smooth03_val_scores.npz
    ;;
  search_single_ignore50)
    search_one single_ignore50_smoothing.json "$LOG_ROOT/cache/iter06_ignore50_val_scores.npz"
    ;;
  search_four_mean)
    search_one four_mean_smoothing.json \
      "$LOG_ROOT/cache/iter06_ignore50_val_scores.npz" \
      "$LOG_ROOT/cache/iter06_ignore100_val_scores.npz" \
      "$LOG_ROOT/cache/iter06_ignore150_val_scores.npz" \
      "$LOG_ROOT/cache/iter06_ignore100_smooth03_val_scores.npz"
    ;;
  search_four_weighted)
    "$PYTHON_BIN" scripts/train/evaluate_score_cache_smoothing.py \
      --score-caches \
        "$LOG_ROOT/cache/iter06_ignore50_val_scores.npz" \
        "$LOG_ROOT/cache/iter06_ignore100_val_scores.npz" \
        "$LOG_ROOT/cache/iter06_ignore150_val_scores.npz" \
        "$LOG_ROOT/cache/iter06_ignore100_smooth03_val_scores.npz" \
      --output-json "$LOG_ROOT/search/four_weighted_smoothing.json" \
      --weights 0.3868919847113909,0.2702462751023454,0.2598238086792704,0.08303793150699335 \
      --transform zscore
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    exit 2
    ;;
esac
