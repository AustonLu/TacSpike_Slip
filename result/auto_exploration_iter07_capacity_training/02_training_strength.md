# 02 training strength 记录

## 计划

在 `width=48/hidden=384` 上调整训练强度：

| run_id | epochs | train samples/epoch | lr | 备注 |
|---|---:|---:|---:|---|
| `iter07_time_channel_w48_h384_ignore50_long_v1` | 15 | 90000 | 1e-3 | 更充分训练 |
| `iter07_time_channel_w48_h384_ignore50_lr5e4_v1` | 12 | 90000 | 5e-4 | 更小 learning rate |

## 待记录

| run_id | best epoch | 20k val acc | 100k random tuned | ROC-AUC | 结论 |
|---|---:|---:|---:|---:|---|
| `iter07_time_channel_w48_h384_ignore50_long_v1` | 未启动 | - | - | - | w48 基础配置已低于 Iter06，暂不扩展 |
| `iter07_time_channel_w48_h384_ignore50_lr5e4_v1` | 未启动 | - | - | - | w48 基础配置已低于 Iter06，暂不扩展 |

## 决策

`w48_h384` 和 `w64_h512` 在 quick 容量实验中都没有超过 Iter06，因此本轮停止 training-strength 分支。下一轮应转向边界建模或轻量 sequence-level 方法，而不是继续扩大 time-channel backbone。
