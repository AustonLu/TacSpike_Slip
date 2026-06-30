# 01 random sampling 的 transition 过滤修正

## 问题

`sample_epoch_indices(..., sampling="random", ignore_transition_ms=...)` 原实现直接在 `[0, len(dataset))` 上均匀采样，没有读取 label-index cache，因此 `--ignore-transition-ms` 只对 `balanced` sampling 生效。

## 修改

当 `sampling="random"` 且 `ignore_transition_ms > 0` 时：

1. 调用 `build_label_index_cache`。
2. 分别过滤 `slip_transition_distance` 和 `no_slip_transition_distance`。
3. 合并过滤后的 slip/no-slip index pool。
4. 从合并池中等概率抽样，保留过滤后自然类别比例。

## 验证点

本轮训练日志中的 `args.ignore_transition_ms` 应正确记录；若 strict 指标和 filtered 指标发生系统性变化，则说明过滤路径已经进入训练。
