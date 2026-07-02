# 01. Stream Cache 设计与 Sanity Check

## 目标

本项将原始 TacSpike HDF5 转换为面向连续状态检测的 sequence-level stream cache。该 cache 不重复保存滑动 window，而是每条 sequence 保存 1 ms 级事件 bin、标签和时间信息，用于 stateful SNN 的连续 chunk 训练。

## 已实现内容

新增数据与脚本：

```text
src/tacspike/data/stream_cache.py
scripts/train/build_stream_cache.py
scripts/train/check_stream_cache.py
```

新增训练/评估脚本后续使用该 cache：

```text
scripts/train/train_stream_cache_scnn.py
scripts/train/evaluate_stream_cache_scnn.py
```

cache 采用显式 `t_label` 边界分箱：

```text
bin_edges = [t_label[0] - 1ms, t_label[0], t_label[1], ...]
```

这样每个 cache bin 与数据集原始 1 ms label/window 对齐，不再用整条 sequence 的线性比例分箱，避免事件落入相邻 bin 的边界误差。

## Dense Cache 尝试和失败原因

先尝试了 dense HDF5 cache：

```text
event_bins: [T_ms, 2, 32, 32]
labels:     [T_ms]
```

第一次使用 HDF5 自动 chunk，chunk 形状类似：

```text
(1726, 1, 4, 4)
```

随机读取 128/384/512 ms segment 时需要解压大量小 chunk，训练进程 CPU 满载，GPU 几乎空闲。

随后改成：

```text
chunks = (512, 2, 32, 32)
```

虽然 chunk 与连续 segment 更匹配，但 gzip dense HDF5 随机读取仍然导致 CPU 成为瓶颈。`sanity_probe_l128` 长时间无法完成一个小 epoch，GPU utilization 接近 0。

结论：压缩 dense stream cache 不适合本项目当前训练访问模式。

## Sparse Cache 方案

由于 TacSpike 事件极稀疏，最终改为 sparse stream cache。每条 sequence 保存：

```text
sparse/t: 非零 pooled event 的 ms bin index
sparse/c: polarity channel
sparse/y: pooled y
sparse/x: pooled x
sparse/v: count
labels:   [T_ms]
t_label:  [T_ms]
valid_mask
transition_distance
```

训练时对每个 segment 读取 `[start, stop)` 内的 sparse entries，然后在内存中重建：

```text
[segment_steps, 2, 32, 32]
```

该方式保留了 stream cache 的 sequence-level 定义，同时避免从磁盘反复读取大块全零 dense tensor。

## Sanity Check

远程路径：

```text
/lamport/makkapakka/jiajunlu/cache/tacspike_stream_cache_v3_sparse
/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter06/build_cache
```

cache 数量：

| split | sequences | total bins | positive fraction |
|---|---:|---:|---:|
| train | 1091 | 20,293,980 | 0.273441 |
| val | 234 | 4,173,170 | 0.285752 |

随机一致性检查：

```text
samples = 24
length = 384
all_label_match = true
max_abs_diff = 0.0
sum_abs_diff = 0.0
```

说明 sparse stream cache 与原始 HDF5 在随机抽样片段上完全一致。

## Probe 结果

`sanity_probe_l128` 使用 sparse cache 成功跑完 1 个小 epoch：

```text
segment_steps = 128
train_segments = 400
val_segments = 120
model width = 16/32, hidden = 64
```

该 probe 的目的只是验证训练链路，不用于精度比较。关键工程指标：

```text
train items_per_second = 5728.30
val items_per_second = 11179.83
```

这说明 sparse cache 已经解决 dense cache 随机读取导致 GPU 长时间空闲的问题，可以进入主实验。

## 当前判断

Stream cache 是必要的，但必须采用稀疏或训练友好的格式。压缩 dense HDF5 虽然概念简单，但不适合高频随机 segment 训练。Sparse cache 在本数据集上更符合事件稀疏性，也为后续 `L=256/384/512` truncated BPTT 提供了可用的数据管线。
