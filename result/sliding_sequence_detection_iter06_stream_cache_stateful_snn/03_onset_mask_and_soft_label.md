# 03 Onset Mask 训练记录

本轮只实际执行了 onset 附近 ignore loss，没有实现 soft label。原因是先前 fine-tune 中 onset 附近的标签不确定性被怀疑会造成训练不稳定，因此先用最小改动验证“忽略 transition 附近监督”是否能提升连续状态检测。

## 对比设置

固定模型和大部分训练参数，仅比较：

| run | segment length | transition ignore | best epoch | best sampled valid balanced acc |
|---|---:|---:|---:|---:|
| `stream_l384_wide` | 384 ms | 0 ms | 8 | 67.501% |
| `stream_l384_ignore30` | 384 ms | 30 ms | 7 | 68.673% |

`transition_ignore_steps=30` 的含义是：对每个训练/验证 segment，距离标签跳变点 30 ms 内的位置不参与 BCE loss 和 valid metrics。

## 结果

| run | valid acc | valid balanced acc | valid F1 | valid ROC-AUC | valid PR-AUC |
|---|---:|---:|---:|---:|---:|
| `stream_l384_wide` | 67.344% | 67.501% | 67.065% | 72.268% | 74.367% |
| `stream_l384_ignore30` | 68.452% | 68.673% | 67.722% | 73.478% | 75.595% |

## 结论

忽略 onset 附近 30 ms 的标签可以带来约 `+1.17%` sampled valid balanced accuracy 和约 `+1.21%` ROC-AUC，方向是正的，但幅度不足以解释和 90%/95% 目标之间的差距。

这说明 onset 标签噪声确实存在影响，但当前主要瓶颈更可能是 raw 1 ms streaming SNN 的长期证据整合能力不足，而不是 onset 附近几十毫秒的标签边界问题。

## 后续建议

soft label 仍值得尝试，但优先级应低于“强特征/长上下文表征”的改造。合理的 soft label 方向是：

- 在 slip onset 前后 30 至 80 ms 内用 ramp label 替代硬 0/1；
- 对 offset 也做单独 ramp 或 ignore，因为状态恢复时间可能与 onset 不对称；
- 训练指标不要只看逐毫秒 BCE，要同时看 full-sequence event recall、false alarm rate 和 delay。
