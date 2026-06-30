# Auto Exploration Iter09 总结

日期：2026-06-30

## 本轮目标

用 transition-distance weighted loss 保留边界 hard samples，同时降低硬标签噪声权重，尝试把 strict 100k random tuned accuracy 推到 `>=90%`。

## 当前状态

已完成训练、100k random/balanced 标准评估和 transition bucket 评估。

## 最佳结果

| 类别 | 配置 | 100k random tuned accuracy | ROC-AUC | >100 ms tuned accuracy |
|---|---|---:|---:|---:|
| Iter09 最佳单模型 | `iter09_tw_near20_mid100_v1` | 89.545% | 91.227% | 90.148% |
| 历史最佳单模型 | `iter06_time_channel_random_ignore50_v1` | 89.631% | 91.828% | 90.233% |
| 历史最佳总体 | Iter06 weighted ensemble | 89.953% | 91.764% | 未重新统计 |

## 是否达到 90%

没有。

本轮最佳 strict all-window 指标为 `89.545%`，低于历史最佳单模型 `89.631%`，也低于历史最佳 weighted ensemble `89.953%`。

## 结论

1. transition-weighted loss 单独使用没有改善主指标。
2. 由于自然采样中 transition 附近窗口占比很低，训练时 `sample_weight_mean` 接近 1.0，简单降权策略对整体梯度影响有限。
3. 稳定标签区仍可超过 90%，`iter09_tw_near20_mid100_v1` 在 `>100 ms` filtered 指标上达到 `90.148%`。
4. strict 指标瓶颈仍集中在 onset/offset 附近，但本轮没有引入新的边界标签建模能力，因此无法跨过 90%。

## 下一轮建议

Iter10 不再继续调 transition 权重。优先做固定随机窗口 score cache，在同一套 100k validation indices 上缓存多个 SNN checkpoint 的分数，然后用 numpy 快速搜索：

1. 更多 checkpoint 的 weighted ensemble。
2. raw/zscore/rank/minmax score transform。
3. threshold calibration。
4. 若可行，加入基于固定窗口顺序的轻量后处理分析。

原因：Iter06 weighted ensemble 已达到 `89.953%`，距离 90% 仅 0.047 个百分点；Iter09 虽单模型较弱，但可能与 Iter06/Iter04 checkpoint 形成少量互补。若固定 score cache 的 ensemble 仍不能达标，再转向显式 onset/offset 软标签、边界容忍评价或序列状态模型。
