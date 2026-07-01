# Sliding Sequence Detection Iter03 总结

日期：2026-07-01

## 本轮目标

按 workflow 将训练目标从独立 window classification 转为连续状态检测，尝试达到 `95%` sequence-level accuracy。

## 完成内容

1. 改造了 1ms streaming LIF-SNN 的训练目标和完整序列评估。
2. 新增 sliding-window sequence-aware 训练脚本。
3. 尝试了从零训练、从 Iter02 `ctx400` checkpoint fine-tune、状态解码和 score cache ensemble。
4. 拉回了所有关键 summary / evaluation JSON / score cache。

## 最好结果

| method | accuracy | balanced acc | F1 | segment recall | missed | false alarms/min | p95 delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| Iter03 best | 0.899023 | 0.877657 | 0.815493 | 0.95 | 1 | 1.787 | 231.3ms |
| Iter02 ctx400 baseline | 0.897140 | 0.873460 | 0.810990 | 0.90 | 2 | 1.625 | 264.4ms |

本轮没有达到 90%，也没有接近 95%。最优方案只比 Iter02 提升约 `0.19` 个百分点。

## 结论

`400ms sliding-window SNN + 连续状态后处理` 仍是当前最好路线，但它在现有数据和 strict per-window 指标下已经接近瓶颈。1ms streaming 从零训练明显不足，sequence-aware fine-tune 和状态解码只能带来小幅改进。

下一步应优先做标签/序列级误差审计和 tolerance-based detection 指标，而不是继续盲目扩大模型或重复调 debounce。
