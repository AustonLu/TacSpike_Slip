# 02 标准评估记录

日期：2026-06-30

## 评估设置

每个 checkpoint 均使用同一套验证流程：

- `val random 100k`：自然分布随机窗口，按 accuracy 搜索最佳阈值。
- `val balanced 100k`：正负类平衡采样窗口，按 accuracy 搜索最佳阈值。
- `transition buckets`：在同一套 `random 100k` 上按最近 label transition 距离拆分统计。

## 主指标

| run_id | random 100k tuned accuracy | random ROC-AUC | balanced 100k tuned accuracy | >100 ms tuned accuracy | 是否达标 |
|---|---:|---:|---:|---:|---|
| `iter09_tw_near20_mid50_v1` | 89.261% | 91.066% | 85.649% | 89.861% | 否 |
| `iter09_tw_near50_mid100_v1` | 89.287% | 91.276% | 85.597% | 89.877% | 否 |
| `iter09_tw_near20_mid100_v1` | 89.545% | 91.227% | 85.979% | 90.148% | 否 |
| `iter09_tw_near50_mid100_smooth02_v1` | 89.465% | 91.107% | 85.881% | 90.063% | 否 |

## 与历史最佳对比

| 方法 | random 100k tuned accuracy | ROC-AUC | 备注 |
|---|---:|---:|---|
| Iter06 最佳单模型：`iter06_time_channel_random_ignore50_v1` | 89.631% | 91.828% | 当前单模型历史最佳 |
| Iter06 最佳 weighted ensemble | 89.953% | 91.764% | 当前总体历史最佳，距离 90% 仅 0.047 个百分点 |
| Iter09 最佳单模型：`iter09_tw_near20_mid100_v1` | 89.545% | 91.227% | 低于 Iter06 单模型 |

## transition 相关观察

`iter09_tw_near20_mid100_v1` 在过滤最近 transition 距离 `>100 ms` 后达到 90.148%，但 strict all-window 仍为 89.545%。这与 Iter06 的模式一致：稳定标签区已经可以越过 90%，主指标被 onset/offset 附近窗口拉低。

本轮没有解决该问题，原因主要有两点：

1. transition 附近窗口占比较小，降权后对全局 loss 影响不足。
2. 降权只是在训练中弱化边界样本，并没有给模型提供更准确的 onset 对齐、软标签或序列状态约束。

## 结论

Iter09 没有达到 90%。transition-weighted loss 作为单独策略收益不明显，不建议继续在同一形式上微调 near/mid 权重。下一轮应优先利用已有 checkpoint 的互补性，做固定随机窗口 score cache 和更充分的 ensemble/校准搜索；如果仍不到 90%，再转回显式 onset/offset 标签噪声处理或序列状态建模。
