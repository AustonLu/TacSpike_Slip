# 01 transition-weighted 训练记录

日期：2026-06-30

## 目的

Iter06 中 `ignore_transition_ms=50` 的稳定区指标可以超过 90%，但 strict all-window 指标仍停在 89.631%。本轮改为保留 transition 附近样本，同时按距离降低 loss 权重，希望减少 onset/offset 硬标签噪声对模型的破坏。

## 配置

公共配置：

- 模型：`time_channel_scnn`
- 神经元：LIF
- 上下文：`context_ms=500`
- 时间 bin：`time_bins=100`
- 采样：natural random
- epoch：10
- 每 epoch 训练样本：70k
- 验证样本：20k
- batch size：96
- optimizer：AdamW + cosine scheduler
- 模型规模：`width=32`, `hidden=256`

## 训练结果

| run_id | 权重策略 | best epoch | 20k val accuracy | 20k ROC-AUC | 100k random tuned accuracy | 100k ROC-AUC | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `iter09_tw_near20_mid50_v1` | 0-20 ms: 0.25, 20-50 ms: 0.60 | 10 | 88.475% | 91.148% | 89.261% | 91.066% | 未优于 Iter06 |
| `iter09_tw_near50_mid100_v1` | 0-50 ms: 0.35, 50-100 ms: 0.70 | 10 | 88.180% | 91.309% | 89.287% | 91.276% | 未优于 Iter06 |
| `iter09_tw_near20_mid100_v1` | 0-20 ms: 0.20, 20-100 ms: 0.70 | 10 | 88.325% | 91.140% | 89.545% | 91.227% | 本轮最佳，但仍未达标 |
| `iter09_tw_near50_mid100_smooth02_v1` | 0-50 ms: 0.35, 50-100 ms: 0.70, label smoothing 0.02 | 10 | 88.360% | 91.168% | 89.465% | 91.107% | 未达标 |

## 观察

1. transition-weighted loss 没有带来主指标提升，最佳单模型 `89.545%`，低于 Iter06 的 `89.631%`。
2. 训练日志中的 `sample_weight_mean` 约为 0.992-0.995，说明 transition 附近窗口在自然随机采样中的占比很小，简单降权对整体 loss 的作用偏弱。
3. 所有 run 的最佳 epoch 都是最后一轮，说明没有明显过拟合，但继续同方向加 epoch 不太可能带来足够大的增益。
4. ROC-AUC 仍维持在 91% 以上，模型排序能力尚可，主要瓶颈是阈值附近和 transition/onset 边界样本。
