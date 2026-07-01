# 04 决策与下一步

日期：2026-07-01

## 关于 Window 长度

本轮结果支持适当缩短 500ms。

推荐当前主线使用 `400ms / 100 bins`：

- 基本追平 500ms：`89.714%` vs `89.733%`。
- balanced accuracy 和 F1 高于 500ms。
- false alarm/min 低于 500ms。
- p95 delay 短于 500ms。

`300ms / 75 bins` 可作为轻量候选，但目前比 400ms 低约 0.22 个百分点，且 false alarm/min 更高。

`100ms` 和 `200ms` 暂时不建议作为主路线，因为连续检测状态抖动和误报明显偏多。

## 为什么仍未达到 95%

本轮最佳单模型 sequence-level accuracy 为 `89.733%`，仍未达到 95%。主要原因不是单一 window 长度，而是任务与训练目标不匹配：

1. 当前模型训练目标仍是独立 window classification，评估目标却是连续状态检测。
2. hard error 集中在少数 sequence 和 onset/offset 区域，单纯延长 context 或调阈值不能解决。
3. 严格逐 ms accuracy 会惩罚合理的物理延迟和标签边界不确定性。
4. 单模型已接近 90%，但 95% 需要显式建模状态持续性、转移边界和误报 run。

## 下一步建议

下一轮不建议继续简单训练更多普通 window-level SNN。更合理的路线是：

1. 固定 `400ms` 作为短期主 context，保留 `300ms` 作为轻量候选。
2. 做 sequence-level score cache 扩展到更多 validation/test sequence，确认 400ms 的优势是否稳定。
3. 训练真正 sequence-aware 或 streaming LIF-SNN：
   - 每 1ms 输入新事件 bin。
   - 保留 LIF hidden state。
   - 在 sequence chunk 上 BPTT。
   - loss 中加入状态平滑、false alarm run、onset/offset 容忍或软标签。
4. 重新定义综合指标：不要只追逐逐窗口 95%，应同时报告 segment recall、false alarm/min、onset delay 和 window accuracy。

如果必须继续以 strict per-window accuracy 为 95% 目标，现有数据和训练口径下很可能不现实；需要先做标签边界审计和事件级容忍指标，否则 95% 可能主要反映标签定义而不是实际滑移检测能力。
