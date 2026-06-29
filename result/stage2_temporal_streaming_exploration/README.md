# Stage 2 Temporal / Streaming 探索记录

日期：2026-06-26

分支：`explore-temporal-streaming`

目标：在 20 ms 独立 window 分类最高约 `80.76%` natural accuracy、`72.17%` balanced accuracy 的基础上，探索更长 temporal context、sequence-level smoothing、slip onset 分析和真正 streaming SNN，尝试把检测精度推到 `90%` 以上。

## 当前结论

本轮最有效的提升来自更长 temporal context。

| 最好结果 | 模型 | 指标 |
|---|---|---|
| window-level natural 100k | `ctx500_frame_cnn_v1` | Accuracy `88.72%`，balanced acc `84.46%`，ROC-AUC `91.15%` |
| window-level balanced 100k | `ctx500_frame_cnn_v1` | Accuracy `85.18%`，F1 `84.14%`，ROC-AUC `91.24%` |
| 轻量 SNN natural 100k | `ctx300_lite_scnn_v1` | Accuracy `85.44%`，balanced acc `78.69%`，ROC-AUC `86.36%` |
| 轻量 SNN balanced 100k | `ctx300_lite_scnn_v1` | Accuracy `80.09%`，F1 `77.81%`，ROC-AUC `86.57%` |
| sequence quick check | `ctx500_frame_cnn_v1` + MA/EMA | 16 条 sequence 上最高 accuracy `89.16%` |
| streaming SNN | `stream_lite_t256_last_v1` | Segment validation accuracy `72.66%` |

目前还没有在稳定、全局的 window-level 指标上达到 `90%`。最接近的是 `ctx500_frame_cnn_v1` 的 natural 100k `88.72%` 和 16 条 sequence 平滑后的 `89.16%`。8 条 sequence 子集可以超过 `95%`，但该子集明显偏容易，不能作为全验证集结论。

## 已实现代码

- 数据层支持按 `context_ms` 从原始 HDF5 事件流动态截取更长上下文。
- `train_lite_scnn.py` 支持 `--context-ms`、`--time-bins`、`--time-steps` 和 checkpoint 兼容恢复。
- `evaluate_lite_scnn.py` 支持长上下文 checkpoint 的 100k random/balanced 评估。
- `evaluate_sequence_smoothing.py` 支持完整 sequence 的 raw、MA、EMA、连续触发和 onset bucket 分析。
- `TacSpikeStreamingLiteSCNN` 与 `train_streaming_scnn.py` 支持每 1 ms 输入、状态延续和 truncated BPTT。
- 新增远程复现实验脚本：
  - `scripts/train/run_stage2_temporal_streaming_exploration.sh`
  - `scripts/train/launch_stage2_temporal_streaming_batch.sh`
  - `scripts/train/evaluate_stage2_temporal_streaming_runs.sh`

## 结果文件

- `01_temporal_context.md`：50/100/200/300/500/1000 ms 上下文实验，包含 FrameCNN upper-bound 与 Lite-SCNN 对照。
- `02_sequence_smoothing.md`：ctx100 完整 32 条 sequence 和 ctx500 quick check 的平滑结果。
- `03_onset_analysis.md`：transition 附近窗口的错误分布和 onset delay。
- `04_streaming_snn.md`：stateful streaming SNN 的 t128 all-step 和 t256 last-step 实验。
- `remote_summaries/`：远程训练/评估 JSON 备份。

## 关键判断

1. 20 ms window 信息量不足，增加上下文是必要的。
2. 500 ms FrameCNN 是当前最强 upper-bound；1000 ms 没有继续提高，说明不是越长越好。
3. 轻量 LIF-SCNN 能从长上下文受益，但仍落后 FrameCNN，后续 SNN 结构需要更好的时间信息保留或读出。
4. Sequence smoothing 能小幅提高实用指标，但不能可靠地把全局结果推过 `90%`。
5. Onset 附近仍然困难，50 ms 内 accuracy 约 `60%`，但边界窗口占比低，不是全部误差来源。
6. 当前 naive streaming BPTT 明显弱于 sliding-window 长上下文，不适合直接作为主路线替代方案。

## 后续建议

下一步如果继续追求 `90%`，优先级建议如下：

1. 以 `ctx500_frame_cnn_v1` 作为 upper-bound 参照，做完整 validation sequence 的离线缓存版评估，避免动态 500 ms voxelize 过慢。
2. 设计能显式保留 300-500 ms 历史的轻量 LIF-SCNN，例如更强 temporal readout、后半段 readout、或 teacher-student distillation。
3. 对 validation sequence 按物体/材料/实验批次分组，定位低分 sequence，而不是继续只看总体均值。
4. Streaming 作为后续目标保留，但应先从 sliding-window 模型蒸馏或状态缓存推理开始，不建议继续从零训练 naive streaming。
