# 03 Multi-timescale SNN 探索结果

## 目标

本项探索验证：并联不同 LIF 时间常数的 SNN 分支，是否能同时捕捉短时事件突变和长时 slip 状态，从而提升连续滑移检测。

## 实现

新增 `TacSpikeMultiTauStreamingSCNN`，由多个 `TacSpikeStreamingLiteSCNN` 分支组成：

```text
branch_fast: beta = 0.65
branch_mid:  beta = 0.85
branch_slow: beta = 0.95
fusion: logits mean
```

本轮配置：

```text
run = multitau_l384_ignore30
input = raw 1 ms bins
segment_steps = 384
transition_ignore_steps = 30
branch conv1/conv2/hidden = 16/32/64
parameter_count = 114,918
epochs = 8
```

## 训练结果

| run | latest epoch | best epoch | valid accuracy | valid balanced accuracy | valid F1 | valid ROC-AUC | valid PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| multitau_l384_ignore30 | 8 | 7 | 70.13% | 70.28% | 70.01% | 75.85% | 77.67% |

相对 raw 1 ms single-tau streaming SNN，multi-tau 有一定稳定性提升，但明显弱于 multi-scale causal features。

## 最终序列评估

| run | best postprocess | accuracy | balanced accuracy | F1 | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| multitau_l384_ignore30 | ma_200 + debounce on2/off50 | 87.70% | 81.89% | 75.16% | 82.06% | 69.33% |

## 结论

multi-timescale LIF 对最终序列检测有帮助，但不是主要突破口。它的 sequence-level accuracy 达到 87.70%，与 multi-scale sqrt 的 88.57% 接近，但训练端 validation balanced accuracy 只有 70.28%，说明模型本身判别能力偏弱，主要依赖后处理 smoothing/debounce 才获得较高 sequence accuracy。

关键判断：

- 单纯增加 LIF 时间尺度不能替代显式历史特征。
- raw 1 ms 输入过稀疏，multi-tau branch 仍难以从 sparse event stream 中稳定累计长时 evidence。
- multi-tau 作为附加模块可以保留，但应优先和 multi-scale causal features 结合，而不是单独作为主线。

## 下一步建议

如果继续做 multi-tau，建议：

1. 使用 multi-scale causal input + multi-tau branch 的组合，但需要控制参数量和训练时间。
2. 尝试 `fusion=linear`，让模型学习 fast/mid/slow 分支权重，而不是简单平均。
3. 降低 slow branch firing rate 或加入 firing regularization，避免高 beta 分支只形成宽泛状态偏置。
