# Auto Exploration Iter05 总结

日期：2026-06-30

## 本轮目标

解释 Iter04 最强 SNN 为什么卡在 `89.50%`，重点分析 sequence smoothing、transition/onset 标签模糊和 bad sequence。

## 最关键结果

`iter04_time_channel_thr1_random_v1` 在 random 100k 上：

| 评估范围 | Accuracy | Balanced acc | ROC-AUC |
|---|---:|---:|---:|
| 全部窗口 | 89.45% | 85.46% | 91.06% |
| 距 transition >50 ms | 89.79% | 85.90% | 91.28% |
| 距 transition >100 ms | 90.05% | 86.13% | 91.46% |

transition 100 ms 内窗口只占 `1.54%`，但 accuracy 接近随机猜测，足以把总体从 `90.05%` 拉到 `89.45%`。

## 是否达到 90%

严格全窗口 100k random accuracy：没有达到。

忽略 transition 100 ms 内窗口：达到，`90.05%`。

## 主要结论

1. 当前最强 time-channel SNN 在稳定标签区间已经达到 90%。
2. 严格 window-level 指标未达标，主要由 slip/no-slip transition 附近标签模糊窗口拉低。
3. 继续做普通模型调参的收益会很小；下一步应做 transition-aware training 或重新定义评估口径。
4. 现有 sequence smoothing 脚本不适合直接做完整 validation sequence，需先做 score cache，否则动态 voxelize 太慢且易触发文件句柄问题。

## 下一轮建议

下一轮优先做 transition-aware training：

- 支持 random sampling 下的 `ignore_transition_ms`。
- 训练 `time_channel_scnn + random sampling + ignore100`。
- 同时报告 strict 100k random 和 ignore-transition 100k random。
- 如果 strict 仍卡在 89-90，但 ignore100 稳定超过 90，应把结论表述为“稳定状态滑移检测达到 90%，边界硬标签是严格 window accuracy 的主要限制”。

## 本轮产物

- `01_sequence_smoothing.md`
- `02_onset_transition.md`
- `03_bad_sequences.md`
- `04_decision.md`
- 新增脚本：
  - `scripts/train/evaluate_transition_buckets.py`
- 结果：
  - `remote_summaries/time_channel_random_transition_random_100k.json`
  - `remote_summaries/time_channel_random_transition_balanced_100k.json`
