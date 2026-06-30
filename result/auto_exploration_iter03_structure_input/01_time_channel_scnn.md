# 01. Time-Channel LIF-SCNN

状态：完成

## 目的

验证当前 sequential Lite-SCNN 是否因为逐 bin 2D convolution + LIF state 递推而丢失了时间位置身份。Time-channel SNN 将 `[B, T, C, H, W]` reshape 为 `[B, T*C, H, W]`，接近 FrameCNN 的输入表示，但把隐层激活换成 LIF surrogate spike。

## 新增模型

新增 `TacSpikeTimeChannelSCNN`：

- 输入：`500 ms / 100 bins`
- 前端：time-channel 2D convolution
- 隐层：Conv-BN-LIF 堆叠
- 分类头：FC-BN-LIF + linear logits
- 仍使用 LIF，不使用 IAF

## Quick 结果

| Run | Threshold | Distill | Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---:|---|---:|---:|---:|---:|
| `iter03_time_channel_scnn_quick_v1` | 0.1 | 否 | 64.89% | 70.66% | 69.61% | 823,234 |
| `iter03_time_channel_scnn_distill_quick_v1` | 0.1 | 是 | 76.18% | 82.96% | 84.52% | 823,234 |
| `iter03_time_channel_scnn_thr1_quick_v1` | 1.0 | 是 | 82.81% | 89.38% | 90.48% | 823,234 |

## 观察

`threshold=0.1` 时隐藏层 firing rate 过高，模型容易在 validation 上塌缩到单类预测。提高到 `threshold=1.0` 后，firing rate 降到更合理范围，quick accuracy 从 `76.18%` 提升到 `82.81%`。

这说明 time-channel 结构本身不是无效，主要问题是 BatchNorm 后 LIF 阈值尺度不匹配。

## Full 结果

配置：

- `threshold=1.0`
- `sampling=balanced`
- `distill_alpha=0.3`
- `temperature=2.0`
- epoch：`10`
- 每 epoch 训练样本：`70000`
- validation：`20000` balanced samples

| Run | Best epoch | Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---:|---:|---:|---:|---:|
| `iter03_time_channel_scnn_thr1_full_v1` | 8 | 86.13% | 91.04% | 91.81% | 823,234 |

## 结论

Time-channel + LIF hidden 是 Iter03 最有效方向。它显著超过 previous wide2/deep SNN 的 20k validation 结果，并且 100k random tuned accuracy 后续达到 `88.98%`。下一轮应围绕该结构继续优化，而不是继续单纯扩宽 sequential SCNN。
