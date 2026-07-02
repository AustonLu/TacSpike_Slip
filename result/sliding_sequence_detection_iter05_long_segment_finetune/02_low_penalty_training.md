# 02 去除或降低平滑/切换惩罚

## 目的

验证 Iter04 fine-tune 中 `smoothness_weight=0.001`、`flip_penalty_weight=0.01` 是否导致模型过度保守和漏检。

## 对比

| run | smoothness | flip penalty | transition ignore |
|---|---:|---:|---:|
| Iter04 failed fine-tune | 0.001 | 0.01 | 8 |
| `ft_ctx400_seg512_no_smooth` | 0 | 0 | 0 |
| `ft_ctx400_seg512_low_smooth` | 0.0001 | 0 | 0 |
| `ft_ctx400_seg1024_no_smooth` | 0 | 0 | 0 |
| `ft_ctx400_seg1024_ignore50_no_smooth` | 0 | 0 | 50 |

## 结果

与 Iter04 失败结果相比：

| 配置 | strict accuracy | recall | segment recall | delay p95 |
|---|---:|---:|---:|---:|
| Iter04 failed fine-tune | 81.776% | 36.389% | 45.0% | 2737.4 ms |
| 512/no smooth | 88.878% | 79.914% | 100.0% | 1013.4 ms |
| 512/low smooth | 88.892% | 79.834% | 100.0% | 1016.7 ms |
| 1024/no smooth | 89.650% | 81.756% | 90.0% | 291.6 ms |
| 1024/ignore50/no smooth | 89.703% | 77.938% | 85.0% | 354.2 ms |

## 判断

去掉 `flip_penalty` 是必要的。它确实避免了 Iter04 那种严重漏检：segment recall 从 `45%` 恢复到 `85%-100%`。

但降低惩罚不是充分条件。512ms 的 no-smooth/low-smooth 几乎一样，说明 `smoothness=0.0001` 对最终表现影响很小；1024ms 的 transition ignore 反而降低了 segment recall。

本轮更合理的结论是：过强的平滑/切换惩罚会破坏模型，但即使去掉这些惩罚，end-to-end fine-tune 仍没有超过现有状态头方案。
