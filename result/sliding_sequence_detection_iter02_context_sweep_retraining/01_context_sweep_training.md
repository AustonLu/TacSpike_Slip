# 01 Context Sweep 训练

日期：2026-07-01

## 配置

本项训练 5 个同构 `time_channel_scnn`，只改变 temporal context 和 time bins。

共同配置：

- model width：32
- hidden dim：256
- sampling：random
- class weight：none
- ignore transition：50ms
- epochs：8
- train samples/epoch：60000
- val samples：20000
- batch size：96
- scheduler：cosine
- AMP：开启

| run id | context | time bins |
|---|---:|---:|
| `ctx100_w32_h256_v1` | 100ms | 50 |
| `ctx200_w32_h256_v1` | 200ms | 50 |
| `ctx300_w32_h256_v1` | 300ms | 75 |
| `ctx400_w32_h256_v1` | 400ms | 100 |
| `ctx500_w32_h256_v1` | 500ms | 100 |

远程日志根目录：

`/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter02`

## 训练期 Validation 结果

| context | best epoch | best val accuracy | ROC-AUC | F1 |
|---:|---:|---:|---:|---:|
| 100ms | 7 | 0.85495 | 0.86226 | 0.70255 |
| 200ms | 7 | 0.87315 | 0.89097 | 0.74938 |
| 300ms | 7 | 0.87850 | 0.90003 | 0.76256 |
| 400ms | 7 | 0.88265 | 0.89898 | 0.77171 |
| 500ms | 7 | 0.88600 | 0.91024 | 0.77864 |

## 观察

训练期 random validation accuracy 随 context 增加基本单调提升，说明长上下文确实提供了有效信息。提升幅度在 300ms 之后变小：

- 100ms 到 200ms：+1.82%
- 200ms 到 300ms：+0.54%
- 300ms 到 400ms：+0.42%
- 400ms 到 500ms：+0.34%

因此 500ms 不是唯一选择，`300-400ms` 已经接近长上下文收益平台区。
