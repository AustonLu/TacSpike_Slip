# 04. Best 100k Evaluation

状态：完成

## 最佳单模型

`iter04_time_channel_thr1_random_v1`：

| Sampling | Default acc | Tuned acc | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| random 100k | 88.72% | 89.45% | 91.06% | 84.66% |
| balanced 100k | 83.18% | 86.00% | 91.07% | 92.00% |

## 最佳 Ensemble

`ensemble_timechannel3`：

| Sampling | Default acc | Tuned acc | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| random 100k | 89.36% | 89.50% | 91.24% | 84.85% |

## 是否达到 90%

没有达到。

本轮最强 `89.50%` 比 Iter03 的 `88.98%` 提升 `0.52` 个百分点，但距离 `90%` 仍差 `0.50` 个百分点。

## 当前最佳记录

| 阶段 | 方法 | Random tuned acc | Balanced tuned acc |
|---|---|---:|---:|
| Iter03 | time-channel SNN | 88.98% | 85.45% |
| Iter04 | random-sampling time-channel SNN | 89.45% | 86.00% |
| Iter04 | time-channel3 ensemble | 89.50% | 未评估 |

## 结论

Iter04 继续缩小了差距，但仍没有满足用户设定的 `90%` 精度目标。下一轮需要跳出轻量调参，优先考虑数据/标签层面的诊断，例如按 sequence 分析剩余错误、过滤/重定义 onset 附近标签、或使用 sequence-level decision 而不是独立 window-level 目标。
