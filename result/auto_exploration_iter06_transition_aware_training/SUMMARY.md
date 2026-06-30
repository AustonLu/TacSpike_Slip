# Auto Exploration Iter06 总结

日期：2026-06-30

## 本轮目标

通过 transition-aware training 检查 90% 卡点是否主要来自 onset/offset 附近硬标签噪声。

## 当前状态

已完成训练、100k random/balanced 评估、transition-filtered 评估和 weighted SNN ensemble 搜索。

## 最佳结果

| 类别 | 配置 | 100k random tuned accuracy | ROC-AUC |
|---|---|---:|---:|
| 最佳单模型 | `iter06_time_channel_random_ignore50_v1` | 89.631% | 91.828% |
| 最佳普通 ensemble | `ensemble_iter04_iter06_five_random_100k` | 89.915% | 91.610% |
| 最佳 weighted ensemble | `weighted_ensemble_iter04_iter06_five_random_100k` | 89.953% | 91.764% |

最佳 filtered 指标：

| 配置 | strict | >50 ms | >100 ms | >150 ms |
|---|---:|---:|---:|---:|
| `iter06_time_channel_random_ignore50_v1` | 89.631% | 89.972% | 90.233% | 90.451% |

## 是否达到 90%

没有。

最佳 validation-tuned weighted SNN ensemble 为 `89.953%`，严格意义上仍低于 90%。单模型最好为 `89.631%`。

## 结论

1. `random + ignore_transition_ms=50` 是本轮最有效的训练设置，能提升 strict accuracy 和 ROC-AUC。
2. 过滤更宽的 `100/150 ms` 不能进一步提升 strict 指标，说明 transition 附近仍有有用 hard samples，不能简单全部丢弃。
3. 稳定标签区已经超过 90%，`ignore50` 在 `>100 ms` 窗口上达到 `90.233%`。
4. strict all-window accuracy 仍被 onset/offset 附近窗口拉低。
5. ensemble/weighted ensemble 只能把结果推到 `89.953%`，后处理收益接近耗尽。

## 下一轮建议

Iter07 不再优先做后处理，应探索：

1. 更大 time-channel LIF-SCNN：`width=48/64`、`hidden=384/512`。
2. 更充分训练：增加 epoch、训练采样量，调低/调高 learning rate，对比 weight decay。
3. 显式边界处理：transition-aware sample weighting、onset/offset auxiliary target、或 sequence smoothing 的 score cache 版本。
