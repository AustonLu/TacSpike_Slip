# 01. Sampling / Prior Refinement

状态：完成

## 目的

Iter03 的最佳 time-channel SNN 使用 balanced sampling，100k random tuned accuracy 为 `88.98%`。本项检查 random sampling 是否能进一步优化自然分布 accuracy。

## 配置

共同配置：

- 模型：`time_channel_scnn`
- 输入：`500 ms / 100 bins`
- LIF threshold：`1.0`
- width：`32`
- hidden：`256`
- dropout：`0.1`
- epoch：`10`
- 每 epoch 训练样本：`70000`
- validation：`20000` random samples

## 训练结果

| Run | Sampling | Distill | 20k Val acc | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|
| `iter04_time_channel_thr1_random_v1` | random | 否 | 88.62% | 90.88% | 83.83% |
| `iter04_time_channel_thr1_random_distill_v1` | random | alpha=0.3 | 88.67% | 90.96% | 84.03% |

## 100k 结果

| Run | Sampling | Default acc | Tuned acc | Balanced acc at tuned | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| `iter04_time_channel_thr1_random_v1` | random 100k | 88.72% | 89.45% | 85.46% | 91.06% | 84.66% |
| `iter04_time_channel_thr1_random_v1` | balanced 100k | 83.18% | 86.00% | 86.00% | 91.07% | 92.00% |
| `iter04_time_channel_thr1_random_distill_v1` | random 100k | 88.75% | 88.98% | 84.35% | 90.83% | 84.43% |
| `iter04_time_channel_thr1_random_distill_v1` | balanced 100k | 83.41% | 85.01% | 85.01% | 90.91% | 91.91% |

## 结论

Random sampling 是本轮最有效的单模型改进。`iter04_time_channel_thr1_random_v1` 把 100k random tuned accuracy 从 Iter03 的 `88.98%` 提升到 `89.45%`，同时 balanced tuned accuracy 到 `86.00%`。

Random + distillation 没有进一步提升 tuned accuracy，说明 teacher soft label 在 random sampling 下更像 regularizer，而不是增强 ranking 的关键因素。
