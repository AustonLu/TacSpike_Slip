# Auto Exploration Iter08 总结

日期：2026-06-30

## 本轮目标

用 score cache 支持完整 validation sequence 上的 smoothing/hysteresis 快速搜索，检查轻量 streaming 后处理能否把 accuracy 推到 `>=90%`。

## 当前状态

已尝试完整 validation score cache 和扩展 weighted ensemble 搜索，但均因动态 HDF5 读取/voxelize 过慢而停止。

## 最佳结果

本轮没有产生新的有效 accuracy 改善结果。当前全局最佳仍是 Iter06：

| 配置 | 100k random tuned accuracy | ROC-AUC |
|---|---:|---:|
| `weighted_ensemble_iter04_iter06_five_random_100k` | 89.953% | 91.764% |
| `iter06_time_channel_random_ignore50_v1` | 89.631% | 91.828% |

## 是否达到 90%

没有。

## 结论

1. 完整 validation sequence score cache 运行超过 10 分钟仍未写出结果，瓶颈是 HDF5 读取和动态 voxelize。
2. 扩展 ensemble 权重搜索也因多 checkpoint 重复 voxelize 过慢，不适合作为当前自动探索内循环。
3. 继续做后处理前应先实现高效 score/window cache，否则迭代成本过高。
4. 当前最有价值的下一步是训练侧 transition/onset 处理，而不是继续扩大模型或后处理搜索。

## 下一轮建议

Iter09 建议做 transition-aware loss/sample weighting：

1. 为训练集构建 transition distance lookup。
2. 对 transition 附近窗口降权而不是完全忽略，例如 `0-20 ms: 0.2`、`20-50 ms: 0.5`、`>50 ms: 1.0`。
3. 同时保留 random natural sampling，避免改变类别先验。
4. 评估 strict 100k random 与 filtered >100 ms 指标。
