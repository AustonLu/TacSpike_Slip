#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter03}"
ITER02_LOG_ROOT="${ITER02_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter02}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

train_stream() {
  local segment_steps="$1"
  local supervise_tail="$2"
  local transition_ignore="$3"
  shift 3
  exec "$PYTHON_BIN" scripts/train/train_streaming_scnn.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --epochs 12 \
    --train-segments-per-epoch 60000 \
    --val-segments 20000 \
    --segment-steps "$segment_steps" \
    --batch-size 96 \
    --num-workers 8 \
    --sampling random \
    --loss-mode all \
    --warmup-steps 50 \
    --supervise-tail-steps "$supervise_tail" \
    --transition-ignore-steps "$transition_ignore" \
    --positive-weight 1.0 \
    --negative-weight 1.0 \
    --smoothness-weight 0.001 \
    --flip-penalty-weight 0.01 \
    --threshold 0.1 \
    --hidden-dim 128 \
    --scheduler cosine \
    --best-metric valid_accuracy \
    --amp \
    "$@"
}

eval_stream() {
  local train_run="$1"
  local eval_name="$2"
  exec "$PYTHON_BIN" scripts/train/evaluate_streaming_sequence_detection.py \
    --checkpoint "$LOG_ROOT/$train_run/best.pt" \
    --data-root "$DATA_ROOT" \
    --output-json "$OUT/$eval_name/streaming_detection.json" \
    --split val \
    --max-sequences 16 \
    --ma-windows 3,5,10,20,50 \
    --ema-alphas 0.1,0.2,0.4 \
    --debounce-on-k 2,3,5 \
    --debounce-off-k 2,3,5,10 \
    --debounce-threshold-grid 1
}

train_sequence() {
  local segment_windows="$1"
  local sampling="$2"
  local smoothness_weight="$3"
  local flip_penalty_weight="$4"
  local transition_ignore_steps="$5"
  shift 5
  exec "$PYTHON_BIN" scripts/train/train_sequence_scnn.py \
    --model time_channel_scnn \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --cache-dir "$LOG_ROOT/cache" \
    --epochs 10 \
    --train-segments-per-epoch 2500 \
    --val-segments 1000 \
    --segment-windows "$segment_windows" \
    --sequence-stride 1 \
    --batch-size 2 \
    --num-workers 8 \
    --sampling "$sampling" \
    --class-weight none \
    --context-ms 400 \
    --time-bins 100 \
    --threshold 1.0 \
    --model-width 32 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --smoothness-weight "$smoothness_weight" \
    --flip-penalty-weight "$flip_penalty_weight" \
    --transition-ignore-steps "$transition_ignore_steps" \
    --scheduler cosine \
    --best-metric valid_accuracy \
    --target-accuracy 0.95 \
    --amp \
    "$@"
}

eval_sequence() {
  local train_run="$1"
  local eval_name="$2"
  exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
    --checkpoints "$LOG_ROOT/$train_run/best.pt" \
    --data-root "$DATA_ROOT" \
    --output-json "$OUT/$eval_name/sliding_detection.json" \
    --output-score-cache "$OUT/$eval_name/score_cache.npz" \
    --split val \
    --max-sequences 16 \
    --batch-size 128 \
    --num-workers 8 \
    --score-transform raw \
    --ma-windows 3,5,10,20,50,100 \
    --ema-alphas 0.05,0.1,0.2,0.4 \
    --debounce-on-k 1,2,3,5 \
    --debounce-off-k 2,3,5,10,20 \
    --debounce-threshold-grid 16
}

case "$RUN_ID" in
  stream_t400_all_ignore25_smooth_v1)
    train_stream 400 0 25
    ;;
  stream_t400_tail200_ignore25_smooth_v1)
    train_stream 400 200 25
    ;;
  stream_t512_tail256_ignore50_smooth_v1)
    train_stream 512 256 50 --batch-size 72
    ;;
  eval_stream_t400_all_ignore25_smooth_v1)
    eval_stream stream_t400_all_ignore25_smooth_v1 stream_t400_all
    ;;
  eval_stream_t400_tail200_ignore25_smooth_v1)
    eval_stream stream_t400_tail200_ignore25_smooth_v1 stream_t400_tail200
    ;;
  eval_stream_t512_tail256_ignore50_smooth_v1)
    eval_stream stream_t512_tail256_ignore50_smooth_v1 stream_t512_tail256
    ;;
  seq_ctx400_s32_transition_mix_smooth_v1)
    train_sequence 32 transition_mix 0.001 0.01 0
    ;;
  seq_ctx400_s64_transition_mix_smooth_v1)
    train_sequence 64 transition_mix 0.001 0.01 0 \
      --batch-size 1 \
      --train-segments-per-epoch 1800 \
      --val-segments 800
    ;;
  seq_ctx400_s32_end_bal_ignore4_v1)
    train_sequence 32 end_balanced 0.0005 0.005 4
    ;;
  seq_ft_ctx400_s32_transition_mix_lr1e4_v1)
    train_sequence 32 transition_mix 0.0001 0.001 0 \
      --init-checkpoint "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --lr 1e-4 \
      --epochs 6
    ;;
  seq_ft_ctx400_s32_random_lr5e5_v1)
    train_sequence 32 random 0.0001 0.001 0 \
      --init-checkpoint "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --lr 5e-5 \
      --epochs 6
    ;;
  seq_ft_ctx400_s32_tail16_ignore4_lr1e4_v1)
    train_sequence 32 transition_mix 0.0001 0.001 4 \
      --init-checkpoint "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --lr 1e-4 \
      --epochs 6 \
      --tail-windows 16
    ;;
  eval_seq_ctx400_s32_transition_mix_smooth_v1)
    eval_sequence seq_ctx400_s32_transition_mix_smooth_v1 seq_ctx400_s32_transition_mix
    ;;
  eval_seq_ctx400_s64_transition_mix_smooth_v1)
    eval_sequence seq_ctx400_s64_transition_mix_smooth_v1 seq_ctx400_s64_transition_mix
    ;;
  eval_seq_ctx400_s32_end_bal_ignore4_v1)
    eval_sequence seq_ctx400_s32_end_bal_ignore4_v1 seq_ctx400_s32_end_bal_ignore4
    ;;
  eval_seq_ft_ctx400_s32_transition_mix_lr1e4_v1)
    eval_sequence seq_ft_ctx400_s32_transition_mix_lr1e4_v1 seq_ft_ctx400_s32_transition_mix_lr1e4
    ;;
  eval_seq_ft_ctx400_s32_random_lr5e5_v1)
    eval_sequence seq_ft_ctx400_s32_random_lr5e5_v1 seq_ft_ctx400_s32_random_lr5e5
    ;;
  eval_seq_ft_ctx400_s32_tail16_ignore4_lr1e4_v1)
    eval_sequence seq_ft_ctx400_s32_tail16_ignore4_lr1e4_v1 seq_ft_ctx400_s32_tail16_ignore4_lr1e4
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
