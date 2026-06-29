# 加宽 SCNN 实验

状态：完成

目的：检查当前 3.8 万参数 Lite-SCNN 是否容量不足，并测试更大的 LIF-SCNN 是否缩小 CNN/SNN 差距。

## 配置

- `wide`：conv `24/48`，hidden `128`，参数量 `110,128`
- `wide2`：conv `32/64`，hidden `256`，参数量 `282,688`
- 主输入：`context_ms=500`，`time_bins=100`
- readout：`logit_mean`

## Quick 结果

| Run | 参数量 | Val acc | ROC-AUC | 备注 |
|---|---:|---:|---:|---|
| `ctx500_tb100_lite_scnn_quick_v1` | 38,304 | 76.30% | 86.19% | 原始 Lite-SCNN |
| `ctx500_tb100_wide_scnn_quick_v1` | 110,128 | 76.94% | 86.02% | 小幅加宽 |
| `ctx500_tb100_wide2_scnn_quick_v1` | 282,688 | 79.14% | 86.42% | 中等加宽 |
| `ctx500_tb100_wide2_scnn_smooth_quick_v1` | 282,688 | 79.14% | 86.48% | label smoothing 0.05 |

加宽到 `wide2` 有明确收益，但 label smoothing 在 quick run 中没有额外收益。

## Full 结果

| Run | 参数量 | Best epoch | 20k balanced val acc | ROC-AUC | 备注 |
|---|---:|---:|---:|---:|---|
| `ctx500_tb100_wide2_scnn_ignore50_v1` | 282,688 | 8 | 83.61% | 90.15% | transition 过滤 |
| `ctx500_tb100_wide2_scnn_distill_v1` | 282,688 | 6 | 84.07% | 90.32% | CNN 蒸馏 |

## 100k 复评

| Run | Sampling | Default acc | Tuned acc | ROC-AUC |
|---|---|---:|---:|---:|
| `ctx500_tb100_wide2_scnn_distill_v1` | random | 86.99% | 87.58% | 89.65% |
| `ctx500_tb100_wide2_scnn_distill_v1` | balanced | 83.46% | 83.50% | 89.60% |
| `ctx500_tb100_wide2_scnn_ignore50_v1` | random | 87.48% | 87.68% | 89.57% |
| `ctx500_tb100_wide2_scnn_ignore50_v1` | balanced | 83.22% | 83.22% | 89.46% |

## 结论

网络规模确实是因素之一：从 38k 参数到 283k 参数后，完整训练的 balanced validation 从上一轮 SNN 的 `80.09%` 提升到 `83-84%`。但这不是唯一瓶颈，因为继续加深到 DeepSCNN 并没有稳定超过 wide2。当前推荐主模型是 `wide2 + distillation`。
