# DeepSCNN 结构对照

状态：完成

目的：检查更深的 LIF-SCNN、BatchNorm 和第三个卷积 LIF block 是否能解决当前 SNN 精度瓶颈。

## 配置

模型：`TacSpikeDeepSCNN`

结构特点：

- 3 个卷积 LIF block
- BatchNorm2d / BatchNorm1d
- hidden `256`
- readout：`logit_mean`
- 输入：`context_ms=500`，`time_bins=100`

## Quick 结果

| Run | 参数量 | Val acc | ROC-AUC | 备注 |
|---|---:|---:|---:|---|
| `ctx500_tb100_deep_scnn_quick_v1` | 619,522 | 78.72% | 85.45% | 无蒸馏 |
| `ctx500_tb100_deep_scnn_ignore50_quick_v1` | 619,522 | 79.80% | 86.23% | transition 过滤 |
| `ctx500_tb100_deep_scnn_distill_quick_v1` | 619,522 | 82.48% | 88.33% | CNN 蒸馏 |

DeepSCNN quick + distillation 明显高于 wide2 distill quick 的 `79.96%`，因此补跑 full。

## Full 结果

| Run | 参数量 | Best epoch | 20k balanced val acc | ROC-AUC | 备注 |
|---|---:|---:|---:|---:|---|
| `ctx500_tb100_deep_scnn_distill_v1` | 619,522 | 7 | 83.74% | 90.77% | full run |
| `ctx500_tb100_wide2_scnn_distill_v1` | 282,688 | 6 | 84.07% | 90.32% | 当前主模型对照 |

DeepSCNN 的 ROC-AUC 略高，但 accuracy 没有超过 Wide2-SCNN，而且训练曲线震荡较大：

- epoch 3 validation accuracy 掉到 `50.15%`
- epoch 6 validation accuracy 掉到 `59.34%`
- best epoch 7 恢复到 `83.74%`

## 结论

单纯加深网络不是当前最有效方向。DeepSCNN 参数量约为 Wide2-SCNN 的 2.2 倍，训练吞吐更低，验证 accuracy 还略低。它提示容量和归一化有帮助，但结构稳定性需要重新设计，例如降低 firing rate、加入更稳的 readout、调整 BatchNorm 在时间循环中的使用方式，或改用更适合 SNN 的 normalization。
