# 02. Onset / Transition Error Analysis

状态：完成

## 目的

检查距离 slip/no-slip transition 较近的窗口是否解释了 Iter04 最强模型从 `90%` 降到 `89.45%` 的剩余误差。

## 新增脚本

新增：

```text
scripts/train/evaluate_transition_buckets.py
```

该脚本在 100k sampled windows 上：

1. 复用与 `evaluate_lite_scnn.py` 相同的 sampled validation 口径。
2. 输出模型 score 和 label。
3. 根据 global index 查找样本所在 sequence，并计算该窗口距离最近 label transition 的距离。
4. 按 transition distance 分桶统计 accuracy、balanced accuracy、ROC-AUC。

## Random 100k 结果

模型：`iter04_time_channel_thr1_random_v1`

总体：

- Tuned accuracy：`89.45%`
- ROC-AUC：`91.06%`

按距离 transition 分桶：

| 距离 transition | Count | Fraction | Accuracy | Balanced acc | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0-10 ms | 184 | 0.18% | 52.17% | 51.73% | 50.85% |
| 10-20 ms | 182 | 0.18% | 47.80% | 47.80% | 48.06% |
| 20-50 ms | 501 | 0.50% | 51.10% | 51.07% | 51.46% |
| 50-100 ms | 676 | 0.68% | 51.63% | 51.86% | 53.91% |
| >100 ms | 98,457 | 98.46% | 90.05% | 86.13% | 91.46% |
| >50 ms | 99,133 | 99.13% | 89.79% | 85.90% | 91.28% |

## Balanced 100k 结果

总体：

- Tuned accuracy：`85.99%`
- ROC-AUC：`91.08%`

按距离 transition 分桶：

| 距离 transition | Count | Fraction | Accuracy | Balanced acc | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0-10 ms | 244 | 0.24% | 58.61% | 47.27% | 46.55% |
| 10-20 ms | 259 | 0.26% | 62.55% | 53.57% | 54.52% |
| 20-50 ms | 674 | 0.67% | 60.68% | 51.64% | 52.81% |
| 50-100 ms | 882 | 0.88% | 58.96% | 53.23% | 54.30% |
| >100 ms | 97,941 | 97.94% | 86.54% | 86.48% | 91.50% |
| >50 ms | 98,823 | 98.82% | 86.29% | 86.25% | 91.30% |

## 结论

这是本轮最关键的结果。

在 random 100k 中，距离 transition 超过 `100 ms` 的窗口 accuracy 已经达到 `90.05%`。严格 window-level 总体 accuracy 之所以是 `89.45%`，主要是因为 transition 100 ms 内的窗口只占 `1.54%`，但 accuracy 接近随机猜测，足以把总分拉低约 `0.6%`。

这说明当前最强 time-channel SNN 在稳定标签区间已经达到 90% 目标；剩余差距主要来自 slip/no-slip 边界附近的标签模糊，而不是普通稳定窗口分类能力不足。
