# 02. Temporal-Conv LIF-SCNN

状态：完成

## 目的

验证 3D temporal convolution 是否能比逐 bin 2D SCNN 更好提取局部时间模式，同时保留 LIF 隐层。

## 新增模型

新增 `TacSpikeTemporalConvSCNN`：

- 输入：`[B, T, C, H, W]`
- 前端：Conv3d over `[C, T, H, W]`
- 隐层：Conv3d-BN-LIF
- 分类头：FC-BN-LIF + linear logits
- 使用 LIF，不使用 IAF

## Quick 结果

| Run | Threshold | Distill | Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---:|---|---:|---:|---:|---:|
| `iter03_temporal_conv_scnn_quick_v1` | 0.1 | 否 | 74.05% | 79.36% | 81.55% | 1,129,602 |
| `iter03_temporal_conv_scnn_distill_quick_v1` | 0.1 | 是 | 78.01% | 86.71% | 88.60% | 1,129,602 |
| `iter03_temporal_conv_scnn_thr1_quick_v1` | 1.0 | 是 | 82.70% | 89.48% | 90.78% | 1,129,602 |

## Full 结果

| Run | Best epoch | Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---:|---:|---:|---:|---:|
| `iter03_temporal_conv_scnn_thr1_full_v1` | 6 | 84.25% | 89.87% | 91.02% | 1,129,602 |

## 观察

提高阈值同样明显改善 temporal-conv 的稳定性。但 full training 只达到 `84.25%`，低于 time-channel SNN 的 `86.13%`，且参数更多。

## 结论

3D temporal convolution 有收益，但不是当前主路线。它说明输入表示确实重要，但在该数据集和当前模型规模下，直接 time-channel 表示更强、更高效。
