# 02 Score Cache + Temporal Adapter 探索结果

## 目标

本项探索验证：如果前端已经能产生较强 slip score，轻量 temporal/state adapter 是否能显著提升连续检测。这用于判断瓶颈主要在前端表征，还是在 sequence-level 后处理。

## 实现

新增 `scripts/train/train_score_adapter.py`，支持两类 score cache：

- 新格式：`scores`, `labels`, `seq_offsets`
- 旧格式：`raw_score_matrix`, `labels`, `seq_offsets`

对旧格式支持：

```text
score_reduce = row | mean | mean_zscore
```

adapter 输入特征：

```text
raw score
causal MA(20, 50, 100, 200, 400)
raw - causal MA
short-long contrast: 20-100, 50-200, 100-400
```

模型为小型 MLP：

```text
Linear(input_dim, 32) + ReLU + Dropout
Linear(32, 32) + ReLU + Dropout
Linear(32, 1)
```

训练使用 BCE，并忽略 transition 附近 30 ms。

## 使用的 score cache

本轮复用 Iter04 的已有 score cache：

| run | train cache | val cache |
|---|---|---|
| adapter_iter04_best5 | subset_best5_train16_fast | subset_best5_val16_fast |
| adapter_iter04_ctx400 | subset_ctx400_train16_fast | subset_ctx400_val16 |
| adapter_iter04_seqft | subset_seqft_train16_fast | subset_seqft_val16 |

## 结果

| run | adapter valid accuracy | valid balanced accuracy | valid F1 | valid ROC-AUC | best sequence method | sequence accuracy | sequence balanced accuracy | sequence F1 | precision | recall |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| adapter_iter04_best5 | 95.27% | 69.44% | 53.28% | 81.29% | ma_150 + debounce on2/off50 | 95.57% | 67.98% | 52.89% | 99.89% | 35.97% |
| adapter_iter04_ctx400 | 80.76% | 64.45% | 44.96% | 90.04% | raw + debounce on3/off50 | 86.06% | 77.23% | 69.13% | 85.20% | 58.16% |
| adapter_iter04_seqft | 77.40% | 58.09% | 28.19% | 87.31% | raw + debounce on5/off50 | 84.72% | 73.79% | 63.80% | 87.54% | 50.19% |

## 结论

score adapter 没有真正解决 95% 目标，且暴露出一个重要问题：accuracy 很高不等于滑移检测好。

`adapter_iter04_best5` 的 sequence accuracy 达到 95.57%，但 balanced accuracy 只有 67.98%，recall 只有 35.97%。这说明该配置主要是预测 no-slip 或极度保守触发，不能作为有效滑移检测方案。

`adapter_iter04_ctx400` 的 ROC-AUC 达到 90.04%，说明前端分数中包含可用判别信息；但转换成连续状态检测后 accuracy 只有 86.06%、balanced accuracy 77.23%，仍未超过多尺度 SNN。

关键判断：

- 后端 adapter 和 debounce 能改善表观 accuracy，但无法从根本上补足 slip recall。
- 仅靠 score smoothing 会把问题推向高 precision / 低 recall，容易得到虚高 accuracy。
- 当前瓶颈仍主要在前端表征、标签噪声/状态定义和训练目标，而不是后处理器容量。

## 下一步建议

adapter 方向可以保留为诊断工具，但不应作为主线。后续若继续使用，应改为：

1. 以 balanced accuracy / event recall / detection delay 为主目标，而不是 raw accuracy。
2. 使用 sequence-aware calibration，在每条 sequence 内校准阈值漂移。
3. 将 adapter 做成 SNN-compatible state head，并与前端联合训练，而不是离线 score MLP。
