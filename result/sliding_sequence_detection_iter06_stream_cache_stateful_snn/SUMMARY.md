# Iter06 总结：Stream Cache + Stateful SNN

## 本轮目标

本轮按 `docs/plan/automated_snn_accuracy_exploration_workflow.md` 执行，核心目标是把任务从独立 window 分类推进到真正的连续状态检测：

- 构造 sequence-level 1 ms stream cache；
- 训练 stateful streaming SNN；
- 对 full sequence 做 causal readout/postprocess；
- 判断该路线是否能达到 95% strict accuracy。

## 已完成

1. 新增 sparse stream cache pipeline，避免重复 window voxelization。
2. 新增 stream cache sanity check，24 个随机片段重建一致，`max_abs_diff=0.0`，标签一致。
3. 新增 stateful streaming SCNN 训练脚本。
4. 新增 full-sequence stateful evaluation 脚本，评估时同一 sequence 内跨 chunk 传递 LIF state。
5. 完成 256/384/512 ms BPTT 和 384 ms onset-ignore 对比实验。

## 最佳结果

当前最佳 run：`stream_l384_ignore30`

```text
sampled valid balanced accuracy = 68.673%
full-sequence strict accuracy   = 87.876%
full-sequence balanced accuracy = 82.978%
full-sequence F1                = 76.221%
event recall                    = 85.0%
delay p95                       = 312.4 ms
false alarms/min                = 9.586
```

目标未达成：

```text
target strict accuracy = 95.0%
current best           = 87.876%
gap                    = 7.124%
```

## 与上一轮对比

上一轮 window-based SNN 最优：

```text
strict accuracy = 90.108%
balanced accuracy = 88.044%
F1 = 81.934%
segment recall = 95.0%
delay p95 = 678.7 ms
```

本轮 stateful streaming SNN 虽然 delay p95 更短，但 strict accuracy、balanced accuracy、F1 和 segment recall 都更差。原因不是 stream cache 错误，而是当前 raw 1 ms stateful SNN 没有学到足够强的长上下文证据。

## 主要结论

1. sparse stream cache 是正确的数据工程方向，后续实验应继续使用。
2. 纯 1 ms 输入的 Lite stateful SCNN 当前不够强，不能稳定达到 90%，更不能接近 95%。
3. onset ignore 有正向作用，但只带来约 1% 左右 sampled validation 提升。
4. 更长 BPTT 并不自动提升效果，512 ms 配置低于 384 ms。
5. 后处理能显著提升 strict accuracy，但当前 score 分离度不足，后处理无法弥补模型表征瓶颈。

## 下一轮优先建议

下一轮不建议继续只调当前 Lite stateful SCNN 的 epoch/lr/segment length。优先尝试：

1. 基于 stream cache 构造在线多尺度 feature stream，例如 50/100/200/400 ms causal accumulated features。
2. 用上一轮较强 window-based SNN/CNN 产生 score cache，再训练轻量 temporal/state adapter 做连续状态检测。
3. 设计 multi-timescale SNN，把不同 beta 的 LIF 分支或 causal temporal convolution 加入模型。
4. 将 checkpoint selection 和 loss 更贴近 sequence-level 目标，而不是只依赖 sampled segment BCE。

## 远程结果文件

本轮 JSON 结果保存在：

```text
result/sliding_sequence_detection_iter06_stream_cache_stateful_snn/remote_summaries/
```

其中：

- `stream_cache_sanity.json`
- `stream_cache_consistency.json`
- `stream_l*_summary.json`
- `eval_stream_l*.json`

