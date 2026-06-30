# Auto Exploration Iter10 总结

日期：2026-06-30

## 本轮目标

用固定 random-window score cache 消除重复 voxelize 开销，扩展 SNN checkpoint ensemble/score calibration 搜索，目标将 `val random 100k tuned accuracy` 推到 `>=90%`。

## 当前状态

已完成：

- 9 个历史 LIF-SNN checkpoint 的 fixed `val random 100k` score cache。
- cache-based ensemble 搜索。
- 最佳 5 模型 ensemble 的 `random seed456` 与 `balanced seed123` 复核。

## 最佳结果

| 口径 | 方法 | tuned accuracy | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| random 100k seed123 | best5 zscore mean | 90.009% | 91.793% | 86.088% |
| random 100k seed456 | best5 zscore mean fixed | 89.955% | 92.025% | 86.064% |
| balanced 100k seed123 | best5 zscore mean fixed | 86.612% | 91.885% | 92.794% |

最佳成员：

1. `iter06_time_channel_random_ignore100_v1`
2. `iter06_time_channel_random_ignore50_v1`
3. `iter07_time_channel_w48_h384_ignore50_v1`
4. `iter09_tw_near20_mid100_v1`
5. `iter09_tw_near50_mid100_smooth02_v1`

权重均为 0.2，score transform 为 `zscore`。

## 是否达到 90%

达到，但只在主搜索口径上刚刚达到。

- 本轮主指标：`val random 100k seed123 tuned accuracy = 90.009%`
- 复核口径：`val random 100k seed456 tuned accuracy = 89.955%`

因此可以说：当前分支首次把 SNN ensemble 的 100k random validation accuracy 推过 90%，但超过幅度只有 0.009 个百分点，跨 seed 稳定性还不足。

## 结论

1. 固定 score cache 是有效工程改进，解决了 Iter08 中 ensemble 搜索过慢的问题。
2. 多个 LIF-SNN checkpoint 存在少量互补性，简单 5 模型均值即可把 seed123 random100k 推到 `90.009%`。
3. 提升主要来自 ensemble/score calibration，不是单模型结构突破；当前最佳单模型仍是 Iter06 的 `89.631%`。
4. random456 复核回落到 `89.955%`，说明 90% 结果非常贴边，不应过度解读。
5. balanced 口径仍只有 `86.612%`，类别均衡能力没有达到 90%。

## 下一步建议

如果目标是工程上先有一个超过 90% 的 validation 结果，本轮可以作为阶段性达标点提交。

如果目标是稳定、可发表或可部署的 90% 以上结果，建议下一轮不要继续堆 validation ensemble 搜索，而是：

1. 固定 best5 ensemble，在 test split 或多个 random seeds 上报告均值、标准差和置信区间。
2. 做 label/sequence audit，找出 random seed 间波动来自哪些 sequence 或 transition 区域。
3. 若继续训练侧探索，优先做显式 onset/offset 软标签或 sequence-level 状态模型，而不是继续扩大单个 SCNN。
