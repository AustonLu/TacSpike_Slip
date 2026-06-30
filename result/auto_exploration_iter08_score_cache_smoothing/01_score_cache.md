# 01 score cache 记录

## 计划

使用 `scripts/train/cache_sequence_scores.py` 缓存完整 validation split 的顺序分数。

缓存对象：

| cache | checkpoint |
|---|---|
| `iter06_ignore50_val_scores.npz` | `iter06_time_channel_random_ignore50_v1` |
| `iter06_ignore100_val_scores.npz` | `iter06_time_channel_random_ignore100_v1` |
| `iter06_ignore150_val_scores.npz` | `iter06_time_channel_random_ignore150_v1` |
| `iter06_ignore100_smooth03_val_scores.npz` | `iter06_time_channel_random_ignore100_smooth03_v1` |

## 待记录

| cache | windows | sequences | seconds | 状态 |
|---|---:|---:|---:|---|
| `iter06_ignore50_val_scores.npz` | 未完成 | 未完成 | >600s | 完整 validation cache 过慢，已停止 |
| `iter06_ignore100_val_scores.npz` | TBD | TBD | TBD | TBD |
| `iter06_ignore150_val_scores.npz` | TBD | TBD | TBD | TBD |
| `iter06_ignore100_smooth03_val_scores.npz` | TBD | TBD | TBD | TBD |

## 观察

`iter06_ignore50_val_scores.npz` 缓存任务运行超过 10 分钟仍未写出结果文件，GPU 占用很低，主要耗时来自 HDF5 读取和动态 voxelize。该现象与 Iter05 中完整 sequence smoothing 过慢一致。

本轮因此停止完整 validation sequence cache。后续若要继续 sequence smoothing，应先优化数据侧，例如：

1. 缓存已 voxelize 的窗口或模型 score。
2. 只对固定随机 100k window 做 score cache。
3. 用更高效的 HDF5 访问/批量 sequence 读取方式替代逐 window 动态读取。
