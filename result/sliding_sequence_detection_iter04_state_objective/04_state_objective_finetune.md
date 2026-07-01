# 04 状态目标 SNN 微调

## 目的

验证是否可以直接微调 SNN score extractor，使其学习连续状态目标，而不是只在已有 score 后面接状态头。

## 配置

从 Iter03 最优 `seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt` 初始化，继续训练：

```text
context_ms=400
time_bins=100
model_width=32
hidden_dim=256
epochs=8
train_segments_per_epoch=3000
val_segments=1200
segment_windows=64
sampling=transition_mix
lr=5e-5
smoothness_weight=0.001
flip_penalty_weight=0.01
transition_ignore_steps=8
best_metric=valid_accuracy
```

## 训练结果

训练脚本内部的抽样 validation 最好出现在 epoch 2：

| 指标 | 数值 |
|---|---:|
| sampled valid accuracy | 64.620% |
| sampled valid balanced accuracy | 64.720% |
| sampled valid F1 | 56.882% |
| sampled valid ROC-AUC | 70.842% |

之后继续训练到 epoch 8 没有恢复到原始模型水平，说明该状态损失设置破坏了原有 extractor。

## 统一 sliding validation

对 `seq_ft_ctx400_state_loss_v2/best.pt` 使用和其他实验一致的 16 条 validation sequence 滑动检测口径评估：

| 指标 | 数值 |
|---|---:|
| strict accuracy | 81.776% |
| balanced accuracy | 67.407% |
| F1 | 51.731% |
| precision | 89.440% |
| recall | 36.389% |
| specificity | 98.424% |
| segment recall | 45.0% |
| onset delay p95 | 2737.4 ms |
| missed slip segments | 11 / 20 |

最佳后处理：

```text
ma_150_debounce_on8_off10
```

## 判断

本次 end-to-end 状态目标微调失败。主要表现是 recall 和 segment recall 大幅下降，模型变得过于保守，漏检大量 slip segment。该路线目前不如冻结 extractor 后接轻量状态头。

可能原因：

- 当前 sequence training segment 太短，只有 64ms，和 `400ms` context / long-state detection 目标不匹配；
- `transition_mix` 抽样与完整 sequence distribution 不一致；
- `smoothness + flip penalty` 配置让模型倾向保持 no-slip；
- 直接用 `valid_accuracy` 选 checkpoint 会偏向多数类/保守预测；
- fine-tune 学习率虽然低，但仍足以破坏已训练好的 score extractor。

后续若继续这条路线，应先改为更长 segment，例如 512ms 或 1024ms，并以 full-sequence score cache 的 strict accuracy / balanced accuracy 作为选择指标，而不是训练脚本内部的短片段 accuracy。
