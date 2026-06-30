# 03. Wide3 Sequential SCNN

状态：完成

## 目的

验证单纯扩大原 sequential Lite-SCNN 是否还能显著提升精度。

## 配置

- 模型：`lite_scnn`
- 输入：`context_ms=500`，`time_bins=100`
- 宽度：conv `48/96`，hidden `384`
- readout：`logit_mean`
- readout 起点：`readout_start_frac=0.5`
- distillation：`alpha=0.3`，temperature `2.0`
- LIF threshold：`0.1`

## Quick 结果

| Run | Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---:|---:|---:|---:|
| `iter03_wide3_scnn_quick_v1` | 82.15% | 88.09% | 89.47% | 634,464 |

## Full 结果

| Run | Best epoch | Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---:|---:|---:|---:|---:|
| `iter03_wide3_scnn_full_v1` | 9 | 84.07% | 90.60% | 91.61% | 634,464 |

## 观察

Wide3 比之前 wide2 quick 有提升，但 full training 只到 `84.07%`，低于 time-channel SNN 的 `86.13%`。它的 ROC-AUC 达到 `90.60%`，说明更大容量有帮助，但 accuracy 没有同步突破。

## 结论

单纯扩大 sequential SCNN 的收益已经有限。后续不应把主要资源继续投到更宽 sequential SCNN，而应集中优化 time-channel SNN。
