# 01. 完整 Sequence Smoothing

状态：完成

## 目的

验证当前最好 SNN 的错误是否主要来自逐窗输出抖动。如果是，causal moving average、EMA 或 consecutive trigger 应该能显著提升完整 validation sequences 的 accuracy。

## 评估配置

远程脚本：

```bash
scripts/train/run_iter01_postprocess_ensemble.sh smoothing_distill
scripts/train/run_iter01_postprocess_ensemble.sh smoothing_ignore50
```

输入：

- `ctx500_tb100_wide2_scnn_distill_v1/best.pt`
- `ctx500_tb100_wide2_scnn_ignore50_v1/best.pt`

评估：

- split：`val`
- selected sequences：32
- total windows：`671,348`
- smoothing：MA `5/10/20/50/100`，EMA `0.05/0.1/0.2/0.4`，trigger `3/5/10`

原始 JSON：

- `remote_summaries/smoothing_distill_val32.json`
- `remote_summaries/smoothing_ignore50_val32.json`

## Distill 模型结果

| 方法 | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| raw | 86.00% | 81.44% | 75.51% | 88.92% |
| EMA 0.4 | 86.01% | 81.36% | 75.44% | 88.92% |
| MA 5 | 86.01% | 81.40% | 75.48% | 88.92% |
| MA 10 | 86.00% | 81.37% | 75.44% | 88.92% |

Onset delay：

- num_onsets：45
- missed_onsets：5
- median delay：0 ms
- mean delay：333.9 ms
- max delay：4590 ms

## Ignore50 模型结果

| 方法 | Accuracy | Balanced acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| raw | 85.47% | 80.19% | 73.93% | 87.75% |
| EMA 0.05 | 85.48% | 80.14% | 73.88% | 87.82% |
| MA 5 | 85.47% | 80.12% | 73.85% | 87.76% |
| EMA 0.4 | 85.47% | 80.23% | 73.97% | 87.76% |

Onset delay：

- num_onsets：45
- missed_onsets：4
- median delay：0 ms
- mean delay：335.2 ms
- max delay：4465 ms

## 结论

Sequence smoothing 几乎没有提升 accuracy，最多只有约 `0.01%` 的变化。说明当前主要错误不是简单的逐窗抖动；模型分数本身的排序和类别分离仍然不足。

另外，平均 onset delay 被少数长延迟拉高，说明有些 slip 段模型长时间没有稳定触发。下一轮如果继续做时序输出，重点不应是简单平滑，而应是训练目标中显式加入 onset/segment 约束。
