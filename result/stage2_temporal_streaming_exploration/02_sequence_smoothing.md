# Sequence-Level Smoothing 实验

状态：完成

目的：在完整 sequence 上对逐窗口输出进行平滑，观察实用检测精度是否高于单窗口 accuracy。

## 实现

新增脚本：

```bash
scripts/train/evaluate_sequence_smoothing.py
```

该脚本按 sequence 顺序推理，收集 raw score / label，然后在同一组 score 上比较：

- causal moving average：`5/10/20/50/100 ms`
- EMA：`alpha=0.05/0.1/0.2/0.4`
- 连续 K 窗口触发：`K=3/5/10`

注意：`ctx500_frame_cnn_v1` 的动态 500 ms voxelize 很慢。32 条 validation sequence 的完整评估超过 20 分钟仍未完成，因此本轮改为保留 8 条和 16 条 sequence 的 quick check。该结果可用于判断趋势，但不能替代完整 validation 结论。

## `ctx100` 首轮结果

评估子集：validation split 随机 32 条 sequence，总窗口数 `671,348`。

### `ctx100_frame_cnn_v1`

| 方法 | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| raw best threshold | 82.27% | 74.54% | 65.49% | 77.82% | 84.62% |
| raw default threshold | 82.11% | 77.39% | 69.29% | 77.82% | 84.62% |
| MA 50 ms best | **82.99%** | **78.28%** | **70.65%** | **79.16%** | **85.77%** |
| EMA 0.1 best | 82.78% | 77.55% | 69.71% | 78.72% | 85.40% |
| trigger K=3 | 81.93% | 73.28% | 63.44% | 65.70% | 73.28% |

### `ctx100_lite_scnn_v1`

| 方法 | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| raw best threshold | 80.50% | 72.61% | 62.27% | 72.07% | 79.06% |
| raw default threshold | 68.67% | 70.00% | 59.35% | 72.07% | 79.06% |
| MA 50 ms best | **80.97%** | **72.98%** | **62.89%** | **73.15%** | **79.95%** |
| EMA 0.1 best | 80.74% | 72.77% | 62.53% | 72.60% | 79.51% |
| trigger K=3 | 80.46% | 72.32% | 61.79% | 60.41% | 72.32% |

## `ctx500_frame_cnn_v1` Quick Check

### 8 条 sequence

评估窗口数：`177,979`。该子集较容易，不能代表完整验证集。

| 方法 | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| raw best threshold | 95.57% | 95.56% | 93.39% | 97.54% |
| EMA 0.05 best | **95.95%** | **95.96%** | **93.95%** | **97.67%** |
| EMA 0.1 best | 95.93% | 95.93% | 93.92% | 97.67% |
| MA 20 ms best | 95.93% | 95.91% | 93.91% | 97.66% |
| MA 50 ms best | 95.92% | 95.93% | 93.91% | 97.65% |

Onset 延迟：`8` 个 onset，missed onset `0`，median delay `0 ms`，mean delay `114.88 ms`，max delay `830 ms`。

### 16 条 sequence

评估窗口数：`369,291`。该子集比 8 条更接近 100k random 结果，但仍不是完整 validation。

| 方法 | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| raw best threshold | 88.62% | 85.67% | 78.91% | 92.40% |
| MA 100 ms best | **89.16%** | **86.91%** | **80.25%** | 92.74% |
| EMA 0.05 best | 89.14% | 86.82% | 80.17% | **92.76%** |
| MA 50 ms best | 89.12% | 86.56% | 79.99% | 92.74% |
| EMA 0.1 best | 89.10% | 86.69% | 80.05% | 92.74% |

Onset 延迟：`20` 个 onset，missed onset `2`，median delay `0 ms`，mean delay `5.83 ms`，max delay `102 ms`。

## 结论

Sequence smoothing 对 `ctx100` 的提升较小，只能带来约 `0.5-0.8%` 的 accuracy 增益。

对 `ctx500_frame_cnn_v1`，平滑在 16 条 sequence 上把 raw `88.62%` 提升到最高 `89.16%`，接近但仍未稳定达到 `90%`。8 条 sequence 子集可以超过 `95%`，说明不同 sequence 的可分性差异非常大，不能只看小子集结果。

当前最稳妥的结论是：

- 平滑能减少短时抖动，通常提高 F1 和 balanced accuracy；
- 平滑不是主要突破点，不能可靠地把全局 100k window-level 指标推过 `90%`；
- 如果后续继续做 sequence-level 指标，需要按完整 validation sequence 或按物体/批次分组报告，避免被容易 sequence 高估。
