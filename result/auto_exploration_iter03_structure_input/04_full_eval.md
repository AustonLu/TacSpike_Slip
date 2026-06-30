# 04. Full Training 和 100k 评估

状态：完成

## Full Training 对比

| Run | 模型 | Best epoch | 20k Val acc | ROC-AUC | PR-AUC | 参数量 |
|---|---|---:|---:|---:|---:|---:|
| `iter03_time_channel_scnn_thr1_full_v1` | time-channel LIF-SCNN | 8 | 86.13% | 91.04% | 91.81% | 823,234 |
| `iter03_temporal_conv_scnn_thr1_full_v1` | temporal-conv LIF-SCNN | 6 | 84.25% | 89.87% | 91.02% | 1,129,602 |
| `iter03_wide3_scnn_full_v1` | sequential wide3 LIF-SCNN | 9 | 84.07% | 90.60% | 91.61% | 634,464 |

## 100k 评估

只对最强候选 `iter03_time_channel_scnn_thr1_full_v1` 做 100k random/balanced 评估。

| Sampling | Default acc | Tuned acc | Balanced acc at tuned | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| random 100k | 88.52% | 88.98% | 84.60% | 90.24% | 82.79% |
| balanced 100k | 85.38% | 85.45% | 85.45% | 90.35% | 91.14% |

## 对比历史最好

| 方法 | 100k random tuned | 100k balanced tuned | ROC-AUC |
|---|---:|---:|---:|
| Iter01 SNN ensemble | 88.08% | 84.14% | 约 90.20% |
| Stage2 best SNN single | 87.68% | 83.50% | 约 89.60% |
| Stage2 FrameCNN upper-bound | 88.72% | 85.18% | 约 91.2% |
| Iter03 time-channel SNN | 88.98% | 85.45% | 90.24% / 90.35% |

## 结论

Iter03 没有达到 `90%`，但首次把 SNN 的 100k random tuned accuracy 提到 `88.98%`，超过此前 FrameCNN random 100k `88.72%` 和 Iter01 SNN ensemble `88.08%`。

距离 `90%` 还差约 `1.02` 个百分点。下一轮应在 time-channel SNN 上继续做针对性精修，而不是换回 sequential SCNN。
