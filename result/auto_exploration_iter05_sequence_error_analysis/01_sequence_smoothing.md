# 01. Sequence Smoothing

状态：中止，记录原因

## 目的

对 `iter04_time_channel_thr1_random_v1` 做完整 validation sequence smoothing，检查 causal moving average、EMA、连续触发是否能把 accuracy 推过 `90%`。

## 执行情况

首次 full-sequence 评估使用：

```text
max_sequences=999
batch_size=256
num_workers=8
```

结果失败，远程日志报错：

```text
OSError: [Errno 24] Too many open files
```

原因是现有 `evaluate_sequence_smoothing.py` 按 sequence 反复创建 `DataLoader`，每个 loader 又打开多个 worker 和 HDF5 文件句柄，完整 validation sequence 下容易触发文件句柄上限。

随后尝试：

```text
num_workers=0
max_sequences=999
```

以及：

```text
max_sequences=999
max_windows_per_sequence=5000
num_workers=0
```

这两个版本没有文件句柄错误，但速度仍然过慢。`head5000` 版本运行超过 1 小时仍未完成，因此本轮停止该路线，避免把迭代时间耗在低效动态 voxelize 上。

## 结论

当前 sequence smoothing 脚本不适合直接用于 `500 ms / 100 bins` time-channel 模型的完整 validation sequence 评估。后续如果要做完整 sequence-level smoothing，应先实现离线缓存 score 的脚本，或者一次性按连续 global indices 批量预测，避免每条 sequence 反复构建 DataLoader。

本轮改用 100k sampled transition bucket 诊断，它与当前主指标口径一致，并能直接判断 onset/transition 是否解释剩余误差。
