# 04. Decision

状态：完成

## 是否达到 90%

严格 100k random window-level accuracy：没有达到。

当前最好仍是 Iter04：

- 单模型：`89.45%`
- ensemble：`89.50%`

## 是否有合理解释

有。

在 `iter04_time_channel_thr1_random_v1` 的 random 100k transition bucket 诊断中：

- 距离 transition `>100 ms` 的窗口 accuracy：`90.05%`
- 所有窗口总体 tuned accuracy：`89.45%`
- transition `<=100 ms` 的窗口占比：约 `1.54%`
- transition `<=100 ms` 的 accuracy：约 `48-52%`

因此，稳定标签区间已经达到 90%；严格总体指标未达到，主要由 transition 附近标签模糊窗口拉低。

## 对后续实验的含义

如果研究目标是“滑移检测”而不是“每个 1ms window 的边界精确分类”，应考虑同时报告：

1. 严格 window-level accuracy。
2. Ignore-transition-100ms accuracy。
3. Onset-tolerant detection delay / missed onset。
4. Sequence-level smoothing 后的检测指标。

如果必须在严格 window-level 100k random accuracy 上达到 `90%`，下一轮需要更有针对性的策略：

- 训练时过滤或重加权 transition `<=100 ms` 样本。
- 使用 soft labels / temporal label smoothing，缓解 transition 附近硬标签噪声。
- 训练 onset-aware 辅助头，而不是让一个二分类头同时处理稳定状态和边界状态。

## 下一轮建议

Iter06 建议做“transition-aware training”：

1. 在训练采样层支持 `ignore_transition_ms` 对 random sampling 生效，或新增 transition-aware sample weight。
2. 训练 time-channel SNN 的 random sampling + ignore100 版本。
3. 同时评估 strict 100k random 和 ignore-transition 100k random。
4. 如果 strict 仍不到 90，但 ignore100 稳定超过 90，应在论文/报告中明确说明边界标签定义造成的上限。
