# Sliding Sequence Detection Iter05 总结：长序列状态微调

## 本轮目标

按 `docs/plan/snn_slip_detection_experiment_plan.md` 的路线，针对 Iter04 的失败原因优先探索三件事：

1. 将 sequence fine-tune 从短片段改为 `512ms/1024ms` 长 segment；
2. 去掉或大幅降低 `flip_penalty/smoothness`，避免模型过度保守；
3. 保存每个 epoch checkpoint，用 full sliding validation 选择模型。

## 最佳结果

本轮最佳是小规模 probe：

```text
probe_seg1024_no_smooth/best.pt
strict accuracy = 90.108%
balanced accuracy = 88.044%
F1 = 81.934%
segment recall = 95.0%
delay p95 = 678.7 ms
```

该结果低于 Iter04 最佳三路 causal state head：

```text
Iter04 best strict accuracy = 90.198%
```

## 主要结果表

| 实验 | strict accuracy | balanced accuracy | F1 | segment recall | delay p95 |
|---|---:|---:|---:|---:|---:|
| Iter04 三路 state head | 90.198% | 88.503% | 82.288% | 95.0% | 237.5 ms |
| Iter05 probe 1024/no-smooth | 90.108% | 88.044% | 81.934% | 95.0% | 678.7 ms |
| Iter05 probe 512/no-smooth | 89.966% | 87.647% | 81.552% | 95.0% | 485.6 ms |
| Iter05 1024/no-smooth best | 89.650% | 87.151% | 80.915% | 90.0% | 291.6 ms |
| Iter05 512/no-smooth best | 88.878% | 86.040% | 79.409% | 100.0% | 1013.4 ms |
| Iter04 failed fine-tune | 81.776% | 67.407% | 51.731% | 45.0% | 2737.4 ms |

## 结论

本轮验证了三个判断：

1. 长 segment 和去掉 flip/smoothness 惩罚可以避免 Iter04 fine-tune 的严重崩溃。full sliding accuracy 从 `81.776%` 恢复到约 `89%-90%`。
2. 训练脚本内部 sampled validation 仍不能可靠选择最终模型；full sliding checkpoint selection 是必要的。
3. 这些改动没有突破当前平台。最佳 `90.108%` 仍低于 Iter04 state head 的 `90.198%`，更远低于 `95%` 目标。

## 工程瓶颈

长 segment 训练和 full sliding evaluation 都很慢。原因是当前 `WindowSequenceDataset` 对一个 512/1024ms segment 内的每个 1ms window 都重新从 HDF5 取窗口并 voxelize；而每个 window 又有 400ms context，重叠计算极大。

后续若继续 sequence fine-tune，需要先做 sequence/context cache 或直接构造连续时间张量，避免重复 voxelization。

## 下一步建议

不建议继续在当前 `train_sequence_scnn.py` 的逐 window 重复 voxelization 结构上堆更多 long-segment fine-tune。更合理的下一步是：

1. 做 sequence-level score/cache 训练：先缓存每条 sequence 的 SNN feature/logit 序列，再训练更强但仍轻量的 temporal adaptor，并做 per-sequence calibration。
2. 若要 end-to-end 微调，先重写数据管线为连续 sequence tensor/cache，使 512/1024ms 训练可承受，然后用 full sliding 指标驱动 early stopping。
3. 检查 label 或数据本身的跨序列差异：当前多个路线都停在 `90%` 左右，更像是部分 sequence 的 score 分布偏移，而不是单纯训练片段长度问题。
