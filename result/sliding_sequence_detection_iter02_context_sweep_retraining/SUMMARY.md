# Sliding Sequence Detection Iter02 总结

日期：2026-07-01

## 本轮完成内容

本轮按 workflow 完成：

1. 先写 `PLAN.md`。
2. 训练 `100/200/300/400/500ms` 五个同构 SNN 做 context sweep。
3. 对五个 checkpoint 做同口径 sequence-level sliding detection。
4. 根据 sweep 结果对 300ms 和 400ms 做 targeted retraining。
5. 记录所有结果和结论。

## 最佳结果

| run | train val acc | sequence acc | balanced acc | F1 | segment recall | false alarms/min | p95 delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctx100 | 0.85495 | 0.87495 | 0.82827 | 0.75743 | 0.900 | 13.973 | 195.7ms |
| ctx200 | 0.87315 | 0.88912 | 0.85932 | 0.79374 | 0.900 | 7.799 | 150.3ms |
| ctx300 | 0.87850 | 0.89496 | 0.86893 | 0.80593 | 0.900 | 2.112 | 221.1ms |
| ctx400 | 0.88265 | 0.89714 | 0.87346 | 0.81099 | 0.900 | 1.625 | 264.4ms |
| ctx500 | 0.88600 | 0.89733 | 0.86891 | 0.80849 | 0.900 | 2.925 | 356.3ms |
| retrain ctx300 | 0.87655 | 0.88479 | 0.84716 | 0.78110 | 0.900 | 5.524 | 762.2ms |
| retrain ctx400 | 0.88000 | 0.88952 | 0.84581 | 0.78498 | 0.850 | 8.774 | 419.4ms |

## 核心结论

400ms 是当前最合理的窗口长度。它几乎追平 500ms 的 sequence accuracy，同时 balanced accuracy、F1、false alarm/min 和 p95 delay 更好。

本轮没有达到 90%，最佳 `ctx500` 为 `89.733%`，`ctx400` 为 `89.714%`。targeted retraining 反而下降，说明简单加容量、label smoothing 和 margin loss 不是正确方向。

## 对 95% 目标的判断

在当前 strict per-window sequence accuracy 口径下，95% 暂时看起来不是靠普通 window-level SNN 训练能达到的目标。主要瓶颈是连续状态检测和标签边界，而不是单纯 context 不够长。

要继续逼近 95%，下一步应进入真正 sequence-aware/streaming LIF-SNN：

- 用 400ms 作为主参考 context。
- 建立更完整的 sequence score/cache 评估。
- 训练时直接处理连续状态、onset/offset 和 false alarm run。
- 同时报告综合指标，而不是只报告逐窗口 accuracy。
