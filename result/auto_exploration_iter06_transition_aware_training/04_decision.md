# 04 本轮决策

## 结果

本轮最佳 single model：

- `iter06_time_channel_random_ignore50_v1`
- strict 100k random tuned accuracy：`89.631%`
- ROC-AUC：`91.828%`
- filtered >100 ms tuned accuracy：`90.233%`

本轮最佳 ensemble：

- `weighted_ensemble_iter04_iter06_five_random_100k`
- strict 100k random tuned accuracy：`89.953%`
- ROC-AUC：`91.764%`

## 决策

没有达到严格 90% 目标。

本轮证明 transition-aware training 能提升分数可分性和稳定区表现，但不能单独解决 strict all-window accuracy。继续只调阈值或 ensemble 的收益已经非常小；下一轮应转向：

1. 更强 time-channel LIF-SCNN 容量，例如 `width=48/64`、`hidden=384/512`。
2. 更充分训练，例如更长 epoch、更大训练采样量、不同 learning rate/weight decay。
3. 显式边界建模，例如 transition/onset 辅助头或 transition-aware sample weighting，而不是简单过滤。
