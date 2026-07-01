# 03 指标判断与后续建议

日期：2026-07-01

## 本轮判断

这次探索确认：滑移检测应按连续序列评价，而不是只看独立随机 window accuracy。原因很直接：实际系统每 1ms 输出一次状态，关键问题是 slip 段是否被及时检测、no-slip 段是否持续误报、状态是否抖动。

本轮结果显示：

- 原始 window size：accuracy `79.60%`，误报 run `478`，状态抖动严重。
- 500ms best5 ensemble：accuracy `89.78%`，false alarm runs/min `0.325`，状态转移数接近真实标签转移数。
- 更细的 debounce threshold 搜索没有进一步提升，说明瓶颈不只是后处理阈值。

因此，后续主线应继续采用较长 temporal context 或真正 streaming sequence model，而不是回到短 window 模型。

## 当前 89.78% 的含义

`89.78%` 是在 validation split 中 16 条完整 sequence 上，用同一批序列调 threshold 和后处理得到的诊断结果，不应直接等同于最终部署泛化指标。它的价值是说明：

1. 长上下文对连续状态稳定性非常有效。
2. 已有 SNN 模型在 sequence-level detection 上已经接近 90%。
3. 离 90% 的差距主要来自少数 hard sequence 的漏检和长 false-positive window。

## 下一步优先级

### 1. 做 sequence-level cache

必须优先做缓存。500ms 动态 window 构造太慢，本轮 16 条 sequence 的 best5 推理需要十几分钟。后续应为 validation/test sequence 预生成：

- labels
- sequence offsets
- per-checkpoint scores
- 可选的 500ms voxel tensor cache

这会把 smoothing、debounce、hysteresis、threshold、ensemble weight 搜索从十几分钟降到秒级或分钟级。

### 2. 扩大 sequence-level 验证

本轮只使用 16 条或 32 条 validation sequence。下一轮要至少检查：

- 更多 validation sequence
- 固定阈值从 validation 应用到 held-out sequence
- 按接触形状分组的结果：flat、sharp、sphere、sphere2

否则 89.78% 附近的结果无法判断是否稳定。

### 3. 训练时引入 sequence 目标

已有模型是在 window-level 目标下训练，后处理只能有限修正。要稳定超过 90%，建议训练时加入 sequence-aware 目标：

- onset/offset 附近 soft label 或 ignore band
- false positive window 的惩罚
- transition-aware sampling
- sequence chunk BPTT 或 streaming state 训练

### 4. 尝试 streaming LIF-SNN

本轮 500ms 模型本质上仍是 sliding window：每 1ms 重新输入过去 500ms 的内容。真正 streaming 模型应每 1ms 只输入新事件 bin，并保留神经元状态。它可能带来：

- 更低计算量
- 更自然的连续状态记忆
- 对 onset/offset 的更好动态建模

但 streaming 模型需要新的训练与评估代码，不应和本轮后处理搜索混在一起。

## 是否达标

按本轮 sequence-level accuracy，最佳结果为 `89.78%`，尚未严格达到 90%。但相比原始 window 的 `79.60%`，已经证明长上下文是当前最有效方向。

下一轮应围绕 500ms sequence cache、扩大验证集、sequence-aware 训练和 streaming LIF-SNN 展开。
