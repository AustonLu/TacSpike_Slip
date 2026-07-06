#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter07}"
STREAM_CACHE_ROOT="${STREAM_CACHE_ROOT:-/lamport/makkapakka/jiajunlu/cache/tacspike_stream_cache_v3_sparse}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts/train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

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
  shift 12
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
    --amp \
    "$@"
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
    --ma-windows 20,50,80,100,150,200 \
    --ema-alphas 0.02,0.05,0.1 \
    --debounce-on-k 2,3,5,8 \
    --debounce-off-k 10,20,30,50 \
    --debounce-threshold-grid 24
}

score_adapter() {
  local train_cache="$1"
  local val_cache="$2"
  exec "$PYTHON_BIN" scripts/train/train_score_adapter.py \
    --train-score-cache "$train_cache" \
    --val-score-cache "$val_cache" \
    --output-json "$OUT/score_adapter.json" \
    --output-checkpoint "$OUT/score_adapter.pt" \
    --score-reduce mean_zscore \
    --ma-windows 20,50,100,200,400 \
    --eval-ma-windows 20,50,80,100,150,200,300,400 \
    --ema-alphas 0.02,0.05,0.1 \
    --debounce-on-k 2,3,5,8 \
    --debounce-off-k 10,20,30,50 \
    --debounce-threshold-grid 24 \
    --transition-ignore-steps 30 \
    --epochs 40 \
    --batch-size 8192 \
    --hidden-dim 32 \
    --dropout 0.10 \
    --lr 0.001 \
    --weight-decay 0.0001
}

case "$RUN_ID" in
  multiscale_l384_sqrt)
    train_stream_cache 384 18000 4000 0.0003 30 32 64 128 0.05 8 8 8 \
      --feature-mode multiscale \
      --feature-windows 1,20,50,100,200,400 \
      --multiscale-normalization sqrt
    ;;
  multiscale_l384_mean)
    train_stream_cache 384 18000 4000 0.0003 30 32 64 128 0.05 8 8 8 \
      --feature-mode multiscale \
      --feature-windows 1,20,50,100,200,400 \
      --multiscale-normalization mean
    ;;
  multiscale_l512_sqrt)
    train_stream_cache 512 14000 3000 0.0002 30 32 64 128 0.05 8 6 8 \
      --feature-mode multiscale \
      --feature-windows 1,20,50,100,200,400 \
      --multiscale-normalization sqrt
    ;;
  multitau_l384_ignore30)
    train_stream_cache 384 16000 4000 0.0003 30 16 32 64 0.05 8 12 8 \
      --model-type multitau \
      --multi-tau-betas 0.65,0.85,0.95 \
      --multi-tau-fusion mean
    ;;
  multiscale_multitau_l384)
    train_stream_cache 384 14000 3000 0.0002 30 16 32 64 0.05 8 6 8 \
      --model-type multitau \
      --multi-tau-betas 0.65,0.85,0.95 \
      --multi-tau-fusion mean \
      --feature-mode multiscale \
      --feature-windows 1,20,50,100,200,400 \
      --multiscale-normalization sqrt
    ;;
  eval_multiscale_l384_sqrt)
    eval_stream_cache multiscale_l384_sqrt best.pt
    ;;
  eval_multiscale_l384_mean)
    eval_stream_cache multiscale_l384_mean best.pt
    ;;
  eval_multiscale_l512_sqrt)
    eval_stream_cache multiscale_l512_sqrt best.pt
    ;;
  eval_multitau_l384_ignore30)
    eval_stream_cache multitau_l384_ignore30 best.pt
    ;;
  eval_multiscale_multitau_l384)
    eval_stream_cache multiscale_multitau_l384 best.pt
    ;;
  adapter_iter04_ctx400)
    score_adapter \
      "$LOG_ROOT/../tacspike_sliding_sequence_detection_iter04/subset_ctx400_train16_fast/score_cache.npz" \
      "$LOG_ROOT/../tacspike_sliding_sequence_detection_iter04/subset_ctx400_val16/score_cache.npz"
    ;;
  adapter_iter04_seqft)
    score_adapter \
      "$LOG_ROOT/../tacspike_sliding_sequence_detection_iter04/subset_seqft_train16_fast/score_cache.npz" \
      "$LOG_ROOT/../tacspike_sliding_sequence_detection_iter04/subset_seqft_val16/score_cache.npz"
    ;;
  adapter_iter04_best5)
    score_adapter \
      "$LOG_ROOT/../tacspike_sliding_sequence_detection_iter04/subset_best5_train16_fast/score_cache.npz" \
      "$LOG_ROOT/../tacspike_sliding_sequence_detection_iter04/subset_best5_val16_fast/score_cache.npz"
    ;;
  adapter_iter07_multiscale_l384_sqrt)
    score_adapter \
      "$LOG_ROOT/multiscale_l384_sqrt/train_stream_cache_scores.npz" \
      "$LOG_ROOT/eval_multiscale_l384_sqrt/stream_cache_scores.npz"
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
