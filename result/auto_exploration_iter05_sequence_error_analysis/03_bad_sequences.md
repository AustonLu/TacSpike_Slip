# 03. Bad Sequence Analysis

状态：部分完成

## 目的

原计划通过完整 sequence 推理统计每条 sequence 的 accuracy，定位拖低总体指标的困难 sequence。

## 实际情况

完整 sequence 推理在当前动态 voxelize 数据读取实现下过慢：

- full sequence + `num_workers=8`：触发 `Too many open files`
- full sequence + `num_workers=0`：运行很久未完成
- all sequence head5000 + `num_workers=0`：超过 1 小时仍未完成

因此本轮没有得到完整 per-sequence accuracy。

## 已获得的替代信息

`evaluate_transition_buckets.py` 在构建 transition distance lookup 时同时记录了每条 validation sequence 的：

- `sequence_id`
- `windows`
- `positive_fraction`
- `num_transitions`

但它没有按 sequence 聚合预测结果。这个信息已经保存在：

```text
remote_summaries/time_channel_random_transition_random_100k.json
remote_summaries/time_channel_random_transition_balanced_100k.json
```

## 结论

本轮没有完成真正的 bad sequence ranking。下一轮如果继续做 sequence 诊断，应新增 score cache：

1. 一次性对全 validation 或 sampled validation 输出 `{global_index, sequence_id, local_index, label, score}`。
2. 在本地或远程用 numpy/pandas 做 groupby，不再按 sequence 反复重读 HDF5。
3. 这样可以同时支持 smoothing、bad sequence analysis 和 transition bucket analysis。
