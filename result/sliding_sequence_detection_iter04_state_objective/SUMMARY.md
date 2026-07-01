# Sliding Sequence Detection Iter04 总结：连续状态目标

## 本轮目标

把训练和评估从“独立 window 分类”转向“连续 slip state 检测”，目标仍是 validation selected sequence strict accuracy 达到 `95%`。

## 关键结果

| 实验 | strict accuracy | balanced accuracy | F1 | segment recall | onset delay p95 |
|---|---:|---:|---:|---:|---:|
| Iter03 seq-ft + MA/debounce | 89.888% | 87.764% | 81.532% | 95.0% | 231.2 ms |
| 后处理扩展 smoke best | 89.902% | 87.766% | 81.549% | 95.0% | 231.3 ms |
| 双路 causal state head | 90.041% | 87.110% | 81.322% | 90.0% | 435.7 ms |
| 三路 causal state head | 90.198% | 88.503% | 82.288% | 95.0% | 237.5 ms |
| 状态目标 SNN fine-tune | 81.776% | 67.407% | 51.731% | 45.0% | 2737.4 ms |

本轮最佳：

```text
三路 causal temporal state head
strict accuracy = 90.198%
```

## 主要结论

`95%` 不是被 transition 标签边界天然卡死。标签审计显示 validation set 只有 29 次 label transition，`±100ms` transition 窗口只占 `1.396%`，即使 perfect detector 整体延迟 `500ms`，strict accuracy 仍可达 `97.211%`。

后处理不是主要瓶颈。更复杂的 MA/debounce/gap fill/min duration 搜索只能把结果稳定在 `89.9%` 左右。

轻量序列状态头有效但提升有限。三路状态头将 strict accuracy 推到 `90.198%`，并保持 `95%` segment recall，但仍存在大量 false positive / false negative window，离 `95%` 还差约 `4.8` 个百分点。

直接状态目标微调当前失败。短序列片段训练加 smoothness/flip penalty 让模型过于保守，统一 sliding validation 只有 `81.776%`，不能继续作为当前主线。

## 对“为什么没到 95%”的判断

当前瓶颈更像是上游 SNN score extractor 的跨序列可分性不足，而不是解码器太弱。证据：

- 单模型、后处理扩展、状态头、三路融合都在 `89.8% - 90.2%` 附近平台化；
- 三路融合改善了 segment recall 和 delay，但 strict accuracy 只提升约 `0.3%`；
- 标签边界审计说明 `95%` 理论上可达；
- 状态目标 fine-tune 失败说明现有 sequence training objective 还没有正确优化 full-sequence strict metric。

## 下一轮建议

1. 不再只训练 score 后处理，转向改上游 extractor。
2. 重新设计 sequence fine-tune：更长 segment，例如 `512ms/1024ms`，并用 full validation score cache 的 sequence strict accuracy 选模型。
3. 去掉或显著降低 flip penalty，避免模型退化成保守 no-slip。
4. 用 balanced accuracy 或 event recall 作为早停辅助指标，避免多数类 accuracy 误导。
5. 若继续状态头路线，应加入 per-sequence calibration 或 domain/adaptor，而不是只扩大轻量 TCN。
