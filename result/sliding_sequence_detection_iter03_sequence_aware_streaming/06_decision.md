# 06. 本轮决策

## 最好结果

本轮最好结果是：

| method | accuracy | balanced acc | F1 | segment recall | missed | false alarms/min | p95 delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seq_ft_ctx400_s32_transition_mix_lr1e4_v1` + `ma_100_debounce_on5_off20` | 0.899023 | 0.877657 | 0.815493 | 0.95 | 1 | 1.787 | 231.3ms |

与 Iter02 `ctx400` 的 `0.89714` 相比，只提升约 `0.19` 个百分点。与目标 `95%` 相比，仍差约 `5.10` 个百分点。

## 主要结论

1. 纯 1ms streaming LIF-SNN 方向当前不成立，完整 sequence accuracy 只有 `85.70%`。
2. 400ms sliding-window 表征仍是主路线。
3. 从已有 400ms checkpoint 做 sequence-aware fine-tune 有小幅收益，但不足以越过 90%。
4. 更强的连续状态后处理和 score ensemble 不能解决 95% 缺口。
5. 当前严格逐 ms/window accuracy 很可能被标签边界、少数 hard sequence 和定义本身限制。

## 下一轮建议

如果目标仍是严格 per-window accuracy 接近 95%，下一轮不应继续堆类似的 smoothness/debounce/ensemble，而应做更底层的验证：

- 标签审计：逐条检查错分最多的 sequence，尤其是 onset/offset 附近是否与实际滑移时刻一致。
- 使用 tolerance-based detection 指标：允许 onset/offset 有合理毫秒级误差，同时报告 segment recall、false alarm/min、delay。
- 训练集标签重定义或软标签：对 transition 附近使用 ramp/uncertain label，而不是硬 0/1。
- 若必须保持 SNN，考虑两阶段结构：400ms SNN score extractor + 轻量状态头；状态头可以先用非脉冲 CRF/HMM/TCN 验证上限，再决定是否 SNN 化。
