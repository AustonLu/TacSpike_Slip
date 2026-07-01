#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: $0 RUN_ID [GPU_ID]}"
GPU_ID="${2:-0}"

PROJECT_DIR="${PROJECT_DIR:-/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2}"
DATA_ROOT="${DATA_ROOT:-/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0}"
LOG_ROOT="${LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter04}"
ITER02_LOG_ROOT="${ITER02_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter02}"
ITER03_LOG_ROOT="${ITER03_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter03}"
ITER01_LOG_ROOT="${ITER01_LOG_ROOT:-/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter01}"
PYTHON_BIN="${PYTHON_BIN:-/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts/train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUT="$LOG_ROOT/$RUN_ID"
mkdir -p "$OUT"

cache_checkpoint() {
  local checkpoint="$1"
  local split="$2"
  local output_npz="$3"
  exec "$PYTHON_BIN" scripts/train/cache_sequence_scores.py \
    --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT" \
    --output-npz "$output_npz" \
    --split "$split" \
    --batch-size 128 \
    --num-workers 8
}

cache_multi_checkpoint() {
  local split="$1"
  local output_json="$2"
  local output_npz="$3"
  shift 3
  exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
    --checkpoints "$@" \
    --data-root "$DATA_ROOT" \
    --output-json "$output_json" \
    --output-score-cache "$output_npz" \
    --split "$split" \
    --max-sequences 16 \
    --max-windows-per-sequence 12000 \
    --batch-size 256 \
    --num-workers 2 \
    --seed 123 \
    --score-transform raw \
    --ma-windows 100 \
    --ema-alphas 0.05 \
    --debounce-on-k 5 \
    --debounce-off-k 20 \
    --debounce-threshold-grid 1
}

cache_multi_checkpoint_full_val() {
  local output_json="$1"
  local output_npz="$2"
  shift 2
  exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
    --checkpoints "$@" \
    --data-root "$DATA_ROOT" \
    --output-json "$output_json" \
    --output-score-cache "$output_npz" \
    --split val \
    --max-sequences 16 \
    --batch-size 256 \
    --num-workers 2 \
    --seed 123 \
    --score-transform raw \
    --ma-windows 100 \
    --ema-alphas 0.05 \
    --debounce-on-k 5 \
    --debounce-off-k 20 \
    --debounce-threshold-grid 1
}

case "$RUN_ID" in
  cache_ctx400_train)
    cache_checkpoint "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" train "$OUT/ctx400_train_score_cache.npz"
    ;;
  cache_ctx400_val)
    cache_checkpoint "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" val "$OUT/ctx400_val_score_cache.npz"
    ;;
  cache_seqft_train)
    cache_checkpoint "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" train "$OUT/seqft_train_score_cache.npz"
    ;;
  cache_seqft_val)
    cache_checkpoint "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" val "$OUT/seqft_val_score_cache.npz"
    ;;
  audit_val_labels)
    exec "$PYTHON_BIN" scripts/train/audit_sequence_state_labels.py \
      --score-cache "$ITER03_LOG_ROOT/eval_seq_ft_ctx400_s32_transition_mix_lr1e4_v1/seq_ft_ctx400_s32_transition_mix_lr1e4/score_cache.npz" \
      --output-json "$OUT/audit_val_labels.json"
    ;;
  postprocess_ctx400_seqft)
    exec "$PYTHON_BIN" scripts/train/evaluate_state_postprocess_search.py \
      --score-caches \
        "$ITER03_LOG_ROOT/eval_seq_ft_ctx400_s32_transition_mix_lr1e4_v1/seq_ft_ctx400_s32_transition_mix_lr1e4/score_cache.npz" \
        "$ITER03_LOG_ROOT/iter02_ctx400_score_cache.npz" \
      --output-json "$OUT/postprocess_ctx400_seqft.json" \
      --max-subset-size 2 \
      --ma-windows 50,80,100,150 \
      --ema-alphas 0.02,0.05,0.1 \
      --debounce-on-k 2,3,5,8 \
      --debounce-off-k 10,20,30,50 \
      --gap-fill 0,20,50 \
      --min-on 0,20,50 \
      --threshold-grid-size 24 \
      --top-k 100
    ;;
  subset_ctx400_train64)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split train \
      --max-sequences 64 \
      --batch-size 128 \
      --num-workers 8 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_ctx400_train16_fast)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split train \
      --max-sequences 16 \
      --max-windows-per-sequence 12000 \
      --batch-size 256 \
      --num-workers 2 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_seqft_train16_fast)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split train \
      --max-sequences 16 \
      --max-windows-per-sequence 12000 \
      --batch-size 256 \
      --num-workers 2 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_ctx400_val16_fast)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split val \
      --max-sequences 16 \
      --batch-size 256 \
      --num-workers 2 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_seqft_val16_fast)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split val \
      --max-sequences 16 \
      --batch-size 256 \
      --num-workers 2 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_best5_train16_fast)
    cache_multi_checkpoint \
      train \
      "$OUT/sliding_detection.json" \
      "$OUT/score_cache.npz" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore50_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter07_capacity_training/iter07_time_channel_w48_h384_ignore50_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near20_mid100_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near50_mid100_smooth02_v1/best.pt"
    ;;
  subset_best5_val16_fast)
    cache_multi_checkpoint \
      val \
      "$OUT/sliding_detection.json" \
      "$OUT/score_cache.npz" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore50_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter07_capacity_training/iter07_time_channel_w48_h384_ignore50_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near20_mid100_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near50_mid100_smooth02_v1/best.pt"
    ;;
  subset_best5_val16_full)
    cache_multi_checkpoint_full_val \
      "$OUT/sliding_detection.json" \
      "$OUT/score_cache.npz" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore100_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter06_transition_aware_training/iter06_time_channel_random_ignore50_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter07_capacity_training/iter07_time_channel_w48_h384_ignore50_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near20_mid100_v1/best.pt" \
      "$ITER03_LOG_ROOT/../tacspike_auto_iter09_transition_weighted_loss/iter09_tw_near50_mid100_smooth02_v1/best.pt"
    ;;
  subset_seqft_train64)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split train \
      --max-sequences 64 \
      --batch-size 128 \
      --num-workers 8 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_ctx400_val16)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER02_LOG_ROOT/ctx400_w32_h256_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split val \
      --max-sequences 16 \
      --batch-size 128 \
      --num-workers 8 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  subset_seqft_val16)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split val \
      --max-sequences 16 \
      --batch-size 128 \
      --num-workers 8 \
      --seed 123 \
      --score-transform raw \
      --ma-windows 100 \
      --ema-alphas 0.05 \
      --debounce-on-k 5 \
      --debounce-off-k 20 \
      --debounce-threshold-grid 1
    ;;
  state_head_subset_ctx400_seqft)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train64/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train64/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16/score_cache.npz" \
      --output-dir "$OUT" \
      --epochs 40 \
      --chunk-len 4096 \
      --chunk-stride 2048 \
      --batch-size 8 \
      --num-workers 4 \
      --lr 0.001 \
      --hidden-dim 64 \
      --layers 5 \
      --kernel-size 9 \
      --dropout 0.1 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.001 \
      --seed 0 \
      --device cuda
    ;;
  state_head_subset_fast_ctx400_seqft)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train16_fast/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16_fast/score_cache.npz" \
      --output-dir "$OUT" \
      --epochs 40 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 2 \
      --lr 0.001 \
      --hidden-dim 64 \
      --layers 5 \
      --kernel-size 9 \
      --dropout 0.1 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.001 \
      --seed 0 \
      --device cuda
    ;;
  state_head_subset_fast_regularized)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train16_fast/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16_fast/score_cache.npz" \
      --output-dir "$OUT" \
      --epochs 30 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 2 \
      --lr 0.0003 \
      --weight-decay 0.001 \
      --hidden-dim 32 \
      --layers 3 \
      --kernel-size 7 \
      --dropout 0.35 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 100 \
      --smoothness-weight 0.005 \
      --seed 2 \
      --device cuda
    ;;
  state_head_subset_fast_ctx400_seqft_causal_v2)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train16_fast/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16_fast/score_cache.npz" \
      --output-dir "$OUT" \
      --epochs 40 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 2 \
      --lr 0.001 \
      --hidden-dim 64 \
      --layers 5 \
      --kernel-size 9 \
      --dropout 0.1 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.001 \
      --seed 0 \
      --device cuda
    ;;
  state_head_subset_fast_all3)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train16_fast/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16_fast/score_cache.npz" \
      --extra-train-score-caches \
        "$LOG_ROOT/subset_best5_train16_fast/score_cache.npz" \
      --extra-val-score-caches \
        "$LOG_ROOT/subset_best5_val16_fast/score_cache.npz" \
      --extra-feature-specs raw,ma:50,ma:100,ema:0.05 \
      --output-dir "$OUT" \
      --epochs 30 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 2 \
      --lr 0.0005 \
      --weight-decay 0.0005 \
      --hidden-dim 48 \
      --layers 4 \
      --kernel-size 9 \
      --dropout 0.2 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.002 \
      --seed 3 \
      --device cuda
    ;;
  state_head_subset_fast_all3_causal_v2)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train16_fast/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16_fast/score_cache.npz" \
      --extra-train-score-caches \
        "$LOG_ROOT/subset_best5_train16_fast/score_cache.npz" \
      --extra-val-score-caches \
        "$LOG_ROOT/subset_best5_val16_full/score_cache.npz" \
      --extra-feature-specs raw,ma:50,ma:100,ema:0.05 \
      --output-dir "$OUT" \
      --epochs 30 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 2 \
      --lr 0.0005 \
      --weight-decay 0.0005 \
      --hidden-dim 48 \
      --layers 4 \
      --kernel-size 9 \
      --dropout 0.2 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.002 \
      --seed 3 \
      --device cuda
    ;;
  state_head_subset_fast_all3_v2)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/subset_ctx400_train16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_train16_fast/score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/subset_ctx400_val16_fast/score_cache.npz" \
        "$LOG_ROOT/subset_seqft_val16_fast/score_cache.npz" \
      --extra-train-score-caches \
        "$LOG_ROOT/subset_best5_train16_fast/score_cache.npz" \
      --extra-val-score-caches \
        "$LOG_ROOT/subset_best5_val16_fast/score_cache.npz" \
      --extra-feature-specs raw,ma:50,ma:100,ema:0.05 \
      --output-dir "$OUT" \
      --epochs 30 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 2 \
      --lr 0.0005 \
      --weight-decay 0.0005 \
      --hidden-dim 48 \
      --layers 4 \
      --kernel-size 9 \
      --dropout 0.2 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.002 \
      --seed 3 \
      --device cuda
    ;;
  state_head_ctx400_seqft)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/cache_ctx400_train/ctx400_train_score_cache.npz" \
        "$LOG_ROOT/cache_seqft_train/seqft_train_score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/cache_ctx400_val/ctx400_val_score_cache.npz" \
        "$LOG_ROOT/cache_seqft_val/seqft_val_score_cache.npz" \
      --output-dir "$OUT" \
      --epochs 35 \
      --chunk-len 4096 \
      --chunk-stride 2048 \
      --batch-size 8 \
      --num-workers 4 \
      --lr 0.001 \
      --hidden-dim 64 \
      --layers 5 \
      --kernel-size 9 \
      --dropout 0.1 \
      --ma-feature-windows 20,50,100,200 \
      --ema-feature-alphas 0.02,0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 50 \
      --smoothness-weight 0.001 \
      --seed 0 \
      --device cuda
    ;;
  state_head_ctx400_seqft_small)
    exec "$PYTHON_BIN" scripts/train/train_score_state_head.py \
      --train-score-caches \
        "$LOG_ROOT/cache_ctx400_train/ctx400_train_score_cache.npz" \
        "$LOG_ROOT/cache_seqft_train/seqft_train_score_cache.npz" \
      --val-score-caches \
        "$LOG_ROOT/cache_ctx400_val/ctx400_val_score_cache.npz" \
        "$LOG_ROOT/cache_seqft_val/seqft_val_score_cache.npz" \
      --output-dir "$OUT" \
      --epochs 25 \
      --chunk-len 2048 \
      --chunk-stride 1024 \
      --batch-size 16 \
      --num-workers 4 \
      --lr 0.0005 \
      --hidden-dim 32 \
      --layers 4 \
      --kernel-size 7 \
      --dropout 0.2 \
      --ma-feature-windows 20,50,100 \
      --ema-feature-alphas 0.05,0.1 \
      --eval-ma-windows 1,20,50,100 \
      --debounce-on-k 1,2,3,5 \
      --debounce-off-k 5,10,20,50 \
      --transition-ignore 100 \
      --smoothness-weight 0.002 \
      --seed 1 \
      --device cuda
    ;;
  seq_ft_ctx400_state_loss_v2)
    exec "$PYTHON_BIN" scripts/train/train_sequence_scnn.py \
      --model time_channel_scnn \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUT" \
      --cache-dir "$LOG_ROOT/cache" \
      --epochs 8 \
      --train-segments-per-epoch 3000 \
      --val-segments 1200 \
      --segment-windows 64 \
      --sequence-stride 1 \
      --batch-size 1 \
      --num-workers 8 \
      --sampling transition_mix \
      --class-weight none \
      --context-ms 400 \
      --time-bins 100 \
      --threshold 1.0 \
      --model-width 32 \
      --hidden-dim 256 \
      --dropout 0.1 \
      --smoothness-weight 0.001 \
      --flip-penalty-weight 0.01 \
      --transition-ignore-steps 8 \
      --scheduler cosine \
      --best-metric valid_accuracy \
      --init-checkpoint "$ITER03_LOG_ROOT/seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt" \
      --lr 0.00005 \
      --target-accuracy 0.95 \
      --amp
    ;;
  eval_seq_ft_ctx400_state_loss_v2)
    exec "$PYTHON_BIN" scripts/train/evaluate_sliding_detection.py \
      --checkpoints "$LOG_ROOT/seq_ft_ctx400_state_loss_v2/best.pt" \
      --data-root "$DATA_ROOT" \
      --output-json "$OUT/sliding_detection.json" \
      --output-score-cache "$OUT/score_cache.npz" \
      --split val \
      --max-sequences 16 \
      --batch-size 128 \
      --num-workers 8 \
      --score-transform raw \
      --ma-windows 50,80,100,150 \
      --ema-alphas 0.02,0.05,0.1 \
      --debounce-on-k 2,3,5,8 \
      --debounce-off-k 10,20,30,50 \
      --debounce-threshold-grid 24
    ;;
  *)
    echo "Unknown RUN_ID: $RUN_ID" >&2
    exit 2
    ;;
esac
