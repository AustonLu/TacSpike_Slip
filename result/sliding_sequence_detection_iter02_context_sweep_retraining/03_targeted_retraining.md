# 03 针对性重训练

日期：2026-07-01

## 配置

根据 02 的结果，选择 300ms 和 400ms 做 targeted retraining。

目标：

- 300ms：验证更短 context 是否可以通过更大模型和更长训练追平 400/500ms。
- 400ms：验证当前最佳折中 context 是否可以通过重训练超过 90%。

共同配置：

- model：`time_channel_scnn`
- model width：48
- hidden dim：384
- epochs：14
- train samples/epoch：90000
- batch size：72
- sampling：random
- class weight：none
- ignore transition：50ms
- label smoothing：0.02
- margin loss weight：0.02
- margin value：1.0

| run id | context | time bins |
|---|---:|---:|
| `retrain_ctx300_w48_h384_v1` | 300ms | 75 |
| `retrain_ctx400_w48_h384_v1` | 400ms | 100 |

## 训练期结果

| run | best epoch | best val accuracy | balanced accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|---:|
| retrain_ctx300 | 10 | 0.87655 | 0.82521 | 0.87187 | 0.76510 |
| retrain_ctx400 | 10 | 0.88000 | 0.82556 | 0.87601 | 0.76843 |

这两个训练期结果都低于同 context 的 sweep 模型。说明本轮“加容量 + label smoothing + margin loss”的组合没有改善 window-level validation。

## Sequence-Level 评估

使用已生成的 score cache 做同口径 fast evaluation。

| run | best method | accuracy | balanced accuracy | F1 | segment recall | missed | false alarms/min | p95 delay |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| retrain_ctx300 | `ma_50_debounce_on5_off10` | 0.88479 | 0.84716 | 0.78110 | 0.900 | 2 | 5.524 | 762.2ms |
| retrain_ctx400 | `ma_50_debounce_on5_off2` | 0.88952 | 0.84581 | 0.78498 | 0.850 | 3 | 8.774 | 419.4ms |

## 判断

重训练失败，没有超过 context sweep baseline：

- 300ms：从 `0.89496` 降到 `0.88479`。
- 400ms：从 `0.89714` 降到 `0.88952`。

可能原因：

1. label smoothing 和 margin loss 对当前硬标签 window accuracy 不利。
2. 更大模型在 16 条 sequence 评估口径上没有带来更好泛化。
3. 训练目标仍然是独立 window classification，没有直接优化 sequence-level false alarm、missed segment 和 state switching。

结论：下一轮不要继续简单加容量或套 label smoothing/margin。若目标是稳定超过 90% 甚至逼近 95%，需要真正 sequence-aware 训练或 streaming LIF-SNN，而不是继续在 window-level loss 上微调。
