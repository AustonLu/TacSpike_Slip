# Slip Onset 分析

状态：首轮完成

目的：统计错误是否集中在 slip 开始/结束边界附近，并评估边界窗口对整体指标的影响。

## 实现

使用 `scripts/train/evaluate_sequence_smoothing.py` 在完整 sequence 上统计每个 window 到最近 label transition 的距离，并按以下时间桶计算指标：

- `0-10 ms`
- `10-20 ms`
- `20-50 ms`
- `>50 ms`

分析对象：

- `ctx100_frame_cnn_v1`
- `ctx100_lite_scnn_v1`
- `ctx500_frame_cnn_v1`

`ctx100` 使用 validation split 随机 32 条 sequence，总窗口数 `671,348`。`ctx500` 因动态 500 ms voxelize 较慢，使用 16 条 sequence quick check，总窗口数 `369,291`。

## `ctx100_frame_cnn_v1`

使用 raw best accuracy threshold。

| 距离 transition | Count | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0-10 ms | 1,691 | 58.19% | 58.19% | 57.44% | 63.56% |
| 10-20 ms | 1,752 | 57.13% | 57.16% | 56.21% | 62.77% |
| 20-50 ms | 4,908 | 60.31% | 60.50% | 59.13% | 68.12% |
| >50 ms | 662,997 | 82.56% | 74.67% | 65.67% | 84.81% |

Onset 检测延迟：

- onset 数：45
- missed onset：5
- median delay：0 ms
- mean delay：186.18 ms
- max delay：2767 ms

## `ctx100_lite_scnn_v1`

使用 raw best accuracy threshold。

| 距离 transition | Count | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0-10 ms | 1,691 | 59.61% | 59.61% | 56.96% | 63.20% |
| 10-20 ms | 1,752 | 59.70% | 59.76% | 57.42% | 63.06% |
| 20-50 ms | 4,908 | 60.59% | 60.91% | 57.86% | 67.35% |
| >50 ms | 662,997 | 80.75% | 72.71% | 62.38% | 79.20% |

Onset 检测延迟：

- onset 数：45
- missed onset：4
- median delay：0 ms
- mean delay：312.20 ms
- max delay：3107 ms

## `ctx500_frame_cnn_v1`

使用 raw best accuracy threshold，评估 16 条 sequence。

| 距离 transition | Count | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 0-10 ms | 760 | 60.66% | 60.66% | 66.67% | 65.95% |
| 10-20 ms | 789 | 61.72% | 61.70% | 68.01% | 67.02% |
| 20-50 ms | 2,248 | 62.54% | 62.79% | 67.71% | 67.11% |
| >50 ms | 365,494 | 88.90% | 85.84% | 79.17% | 92.58% |

Onset 检测延迟：

- onset 数：20
- missed onset：2
- median delay：0 ms
- mean delay：5.83 ms
- max delay：102 ms

## 结论

Onset 附近确实是明显困难区域。`ctx100` 中距离 transition 50 ms 内的 accuracy 只有约 `57-61%`，远低于非边界窗口的 `80-83%`。`ctx500` 也类似：50 ms 内 accuracy 只有约 `61-63%`，而非边界窗口达到约 `88.90%`。这说明当前逐窗口标签在滑移开始/结束附近存在较强模糊性，模型并不是简单欠训练。

不过边界窗口占比很低。`ctx100` 的 32 条 sequence 中，50 ms 内边界窗口只占约 `1.24%`；`ctx500` 的 16 条 sequence 中也只有约 `1.03%`。所以仅仅剔除 onset 附近窗口不可能把总体 accuracy 从 85-89% 稳定推到 90% 以上。更大的问题仍是不同 sequence 的可分性差异很大，有些 sequence 几乎可达 95% 以上，有些 sequence 接近失效。

后续更应该关注：

- 按物体/批次/sequence 类型分组分析；
- 对难 sequence 做可视化和数据质量检查；
- sequence-level 检测指标要同时报告 missed onset 和长延迟，而不能只报告逐窗口 accuracy。
