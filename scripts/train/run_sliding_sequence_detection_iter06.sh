#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter06}"
STREAM_CACHE_ROOT="${STREAM_CACHE_ROOT:-/lamport/makkapakka/jiajunlu/cache/tacspike_stream_cache_v3_sparse}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts/train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

build_cache() {
  "$PYTHON_BIN" scripts/train/build_stream_cache.py \
    --data-root "$DATA_ROOT" \
    --cache-root "$STREAM_CACHE_ROOT" \
    --splits train,val \
    --spatial-pool 4 \
    --polarity-mode both \
    --dtype float16 \
    --cache-format sparse \
    --compression gzip \
    --chunk-steps 512 \
    --output-json "$OUT/stream_cache_sanity.json"
  exec "$PYTHON_BIN" scripts/train/check_stream_cache.py \
    --data-root "$DATA_ROOT" \
    --stream-cache-root "$STREAM_CACHE_ROOT" \
    --split val \
    --spatial-pool 4 \
    --polarity-mode both \
    --cache-dtype float16 \
    --cache-format sparse \
    --samples 24 \
    --length 384 \
    --output-json "$OUT/stream_cache_consistency.json"
}

train_stream_cache() {
  local segment_steps="$1"
  local train_segments="$2"
  local val_segments="$3"
  local lr="$4"
  local transition_ignore="$5"
  local conv1="$6"
  local conv2="$7"
  local hidden="$8"
  local dropout="$9"
  local epochs="${10}"
  local batch_size="${11}"
  local workers="${12}"
  exec "$PYTHON_BIN" scripts/train/train_stream_cache_scnn.py \
    --stream-cache-root "$STREAM_CACHE_ROOT" \
    --output-dir "$OUT" \
    --index-dir "$LOG_ROOT/index" \
    --epochs "$epochs" \
    --train-segments-per-epoch "$train_segments" \
    --val-segments "$val_segments" \
    --segment-steps "$segment_steps" \
    --batch-size "$batch_size" \
    --num-workers "$workers" \
    --cache-format sparse \
    --sampling transition_mix \
    --loss-mode all \
    --warmup-steps 50 \
    --transition-ignore-steps "$transition_ignore" \
    --positive-weight 1.0 \
    --smoothness-weight 0.0 \
    --threshold 0.1 \
    --conv1-channels "$conv1" \
    --conv2-channels "$conv2" \
    --hidden-dim "$hidden" \
    --dropout "$dropout" \
    --lr "$lr" \
    --scheduler cosine \
    --best-metric valid_balanced_accuracy \
    --target-accuracy 0.95 \
    --save-epoch-checkpoints \
    --amp
}

eval_stream_cache() {
  local train_run="$1"
  local checkpoint_name="${2:-best.pt}"
  exec "$PYTHON_BIN" scripts/train/evaluate_stream_cache_scnn.py \
    --checkpoint "$LOG_ROOT/$train_run/$checkpoint_name" \
    --stream-cache-root "$STREAM_CACHE_ROOT" \
    --output-json "$OUT/stream_cache_detection.json" \
    --output-score-cache "$OUT/stream_cache_scores.npz" \
    --split val \
    --max-sequences 16 \
    --chunk-steps 2048 \
    --ma-windows 20,50,80,100,150 \
    --ema-alphas 0.02,0.05,0.1 \
    --debounce-on-k 2,3,5,8 \
    --debounce-off-k 10,20,30,50 \
    --debounce-threshold-grid 24
}

case "$RUN_ID" in
  build_cache)
    build_cache
    ;;
  sanity_probe_l128)
    train_stream_cache 128 400 120 0.0003 0 16 32 64 0.0 1 8 2
    ;;
  stream_l384_wide)
    train_stream_cache 384 18000 4000 0.0003 0 32 64 128 0.05 8 12 8
    ;;
  stream_l256_wide)
    train_stream_cache 256 24000 5000 0.0003 0 32 64 128 0.05 8 16 8
    ;;
  stream_l512_wide)
    train_stream_cache 512 14000 3000 0.0002 0 32 64 128 0.05 8 8 8
    ;;
  stream_l384_ignore30)
    train_stream_cache 384 18000 4000 0.0003 30 32 64 128 0.05 8 12 8
    ;;
  stream_l384_large)
    train_stream_cache 384 16000 4000 0.0002 0 32 64 256 0.10 8 10 8
    ;;
  eval_sanity_probe_l128)
    eval_stream_cache sanity_probe_l128 best.pt
    ;;
  eval_stream_l384_wide)
    eval_stream_cache stream_l384_wide best.pt
    ;;
  eval_stream_l256_wide)
    eval_stream_cache stream_l256_wide best.pt
    ;;
  eval_stream_l512_wide)
    eval_stream_cache stream_l512_wide best.pt
    ;;
  eval_stream_l384_ignore30)
    eval_stream_cache stream_l384_ignore30 best.pt
    ;;
  eval_stream_l384_large)
    eval_stream_cache stream_l384_large best.pt
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
