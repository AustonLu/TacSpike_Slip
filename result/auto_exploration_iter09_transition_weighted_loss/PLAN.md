# Auto Exploration Iter09 计划：transition-weighted loss

日期：2026-06-30

## 背景

Iter06 证明 `ignore_transition_ms=50` 能提升稳定区指标，但 strict all-window 仍卡在 `89.631%`。完全过滤 transition 样本会丢掉边界附近的 hard samples；Iter08 的后处理路径又受到 score cache 工程瓶颈限制。

本轮改为训练侧 sample weighting：保留 transition 附近样本，但降低其 loss 权重，减少硬标签噪声对模型的破坏。

## 本轮目标

在保持 `time_channel_scnn + LIF + 500 ms context / 100 bins` 的前提下，用 transition-distance weighted loss 把 strict 100k random tuned accuracy 推到 `>=90%`。

## 实验清单

1. `iter09_tw_near20_mid50_v1`
   - `distance <20 ms`: weight 0.25
   - `20-50 ms`: weight 0.60
   - `>50 ms`: weight 1.0

2. `iter09_tw_near50_mid100_v1`
   - `distance <50 ms`: weight 0.35
   - `50-100 ms`: weight 0.70
   - `>100 ms`: weight 1.0

3. `iter09_tw_near20_mid100_v1`
   - `distance <20 ms`: weight 0.20
   - `20-100 ms`: weight 0.70
   - `>100 ms`: weight 1.0

4. `iter09_tw_near50_mid100_smooth02_v1`
   - 同 2
   - 加 `label_smoothing=0.02`

## 评估

每个模型训练后运行：

- 100k random validation tuned accuracy
- 100k balanced validation tuned accuracy
- transition bucket / filtered metrics

## 判定

若 single model strict 100k random tuned accuracy 达到或超过 `90%`，本轮视为达标。
