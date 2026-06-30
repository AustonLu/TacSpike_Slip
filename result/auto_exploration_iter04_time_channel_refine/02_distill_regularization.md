# 02. Distillation Alpha / Regularization

状态：完成

## 目的

在 balanced sampling 下扫描蒸馏权重和正则化，检查是否能提升 time-channel SNN 的 score separability。

## 配置

共同配置：

- 模型：`time_channel_scnn`
- 输入：`500 ms / 100 bins`
- LIF threshold：`1.0`
- sampling：`balanced`
- epoch：`10`
- validation：`20000` balanced samples

## 结果

| Run | 变化项 | 20k Val acc | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| `iter04_time_channel_thr1_alpha01_v1` | distill alpha=0.1 | 86.14% | 90.90% | 91.72% |
| `iter04_time_channel_thr1_alpha05_v1` | distill alpha=0.5 | 85.81% | 90.79% | 91.80% |
| `iter04_time_channel_thr1_dropout02_v1` | dropout=0.2 | 86.04% | 91.18% | 92.03% |
| `iter04_time_channel_thr1_smooth03_v1` | label smoothing=0.03 | 86.17% | 91.09% | 91.96% |

## 对比

Iter03 baseline `distill_alpha=0.3, dropout=0.1`：

- 20k Val acc：`86.13%`
- ROC-AUC：`91.04%`
- PR-AUC：`91.81%`

本项最好 accuracy 是 label smoothing 的 `86.17%`，只比 Iter03 多 `0.04` 个百分点；dropout=0.2 的 ROC-AUC 稍高，但 accuracy 没有明显提高。

## 结论

蒸馏权重和轻量正则化不是剩余 1% 的主要瓶颈。balanced sampling 下这些配置基本和 Iter03 持平，因此没有扩大做 100k 评估。
