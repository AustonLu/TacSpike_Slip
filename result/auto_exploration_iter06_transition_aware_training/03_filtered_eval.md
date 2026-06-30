# 03 filtered transition 评估记录

## 计划

对每个 Iter06 checkpoint 运行 `evaluate_transition_buckets.py`，输出：

- strict 100k random tuned accuracy
- strict 100k balanced tuned accuracy
- `filtered_transition_metrics.gt_50_ms`
- `filtered_transition_metrics.gt_100_ms`
- `filtered_transition_metrics.gt_150_ms`
- transition buckets 中 0-10/10-20/20-50/50-100/>100 ms 的错误分布

## 待记录

| run_id | strict random tuned | >50 ms tuned | >100 ms tuned | >150 ms tuned | 结论 |
|---|---:|---:|---:|---:|---|
| `iter06_time_channel_random_ignore50_v1` | 89.631% | 89.972% | 90.233% | 90.451% | 稳定区达标，strict 被 transition 拉低 |
| `iter06_time_channel_random_ignore100_v1` | 89.447% | 89.779% | 90.033% | 90.256% | 稳定区勉强达标 |
| `iter06_time_channel_random_ignore150_v1` | 89.553% | 89.895% | 90.155% | 90.383% | 稳定区达标 |
| `iter06_time_channel_random_ignore100_smooth03_v1` | 89.463% | 89.790% | 90.041% | 90.260% | smoothing 未解决 strict 指标 |

## 观察

所有模型在排除距离 transition 100 ms 内的窗口后都达到或超过 90%，其中 `ignore50` 最好，`>100 ms` 为 `90.233%`。严格全窗口指标仍未到 90%，说明错误仍主要来自 onset/offset 附近的硬标签边界。
