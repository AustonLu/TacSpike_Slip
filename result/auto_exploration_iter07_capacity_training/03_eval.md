# 03 evaluation 记录

## 计划

每个模型完成后运行：

1. `evaluate_lite_scnn.py --sampling random --samples 100000 --best-threshold-metric accuracy`
2. `evaluate_lite_scnn.py --sampling balanced --samples 100000 --best-threshold-metric accuracy`
3. `evaluate_transition_buckets.py --sampling random --samples 100000`

## 待记录

| run_id | strict random tuned | balanced tuned | >100 ms tuned | 是否达标 |
|---|---:|---:|---:|---|
| `iter07_time_channel_w48_h384_ignore50_v1` | 89.447% | 86.019% | 90.035% | 否 |
| `iter07_time_channel_w64_h512_ignore50_v1` | 88.711% | 84.606% | 89.305% | 否 |
| `iter07_time_channel_w48_h384_ignore50_long_v1` | 未启动 | - | - | 否 |
| `iter07_time_channel_w48_h384_ignore50_lr5e4_v1` | 未启动 | - | - | 否 |

## 观察

`w48` 仍然呈现与 Iter06 类似的现象：排除 transition 100 ms 内窗口后刚好超过 90%，但 strict all-window 未达标。`w64` 连 filtered 指标也下降，说明更大容量不是当前瓶颈。
