# Stage 2 Temporal / Streaming 精度探索计划

日期：2026-06-26

分支：`explore-temporal-streaming`

目标：在阶段 2 原始 20 ms 独立 window 分类约 `80.8%` natural accuracy、`72.2%` balanced accuracy 的基础上，探索更长 temporal context、sequence-level smoothing、slip onset 分析和真正 streaming 推理，目标尝试把实用检测精度推到 `90%` 以上。

## 背景判断

上一轮 `result/stage2_accuracy_exploration/README.md` 的结论是：

- 普通 CNN upper-bound 在 20 ms window 上也只有约 `80.76%` natural accuracy。
- Lite-SCNN 长训、输入放大、更高空间分辨率和更深 LIF-SCNN 都没有本质突破。
- 当前瓶颈更可能来自 20 ms 独立窗口的信息量不足、标签边界模糊和逐窗口评估方式。

因此本轮不优先继续堆单窗口模型，而是优先改变时间上下文和评估方式。

## 探索项

### A. 更长 temporal context

目的：验证单个 20 ms window 信息不足的假设。

实现方式：

- 不重新生成数据集，直接从原 HDF5 的 `events/t,x,y,p` 中按当前窗口 `t_label` 回看更长时间。
- 输入从 `[20, 2, 32, 32]` 扩展到 `[50, 2, 32, 32]`、`[100, 2, 32, 32]`。
- 标签仍使用当前 `t_label` 对应的 `label/slip`。

优先实验：

| Run | 模型 | 输入 | 目的 |
|---|---|---|---|
| `ctx50_frame_cnn_v1` | FrameCNN | 50 ms | 检查 50 ms 上下文 upper-bound |
| `ctx100_frame_cnn_v1` | FrameCNN | 100 ms | 检查 100 ms 上下文 upper-bound |
| `ctx50_lite_scnn_v1` | Lite-SCNN | 50 ms | 检查轻量 SNN 是否能吃到上下文收益 |
| `ctx100_lite_scnn_v1` | Lite-SCNN | 100 ms | 检查更长 SNN 状态是否提升 |

记录文件：`01_temporal_context.md`

### B. Sequence-level smoothing

目的：验证单窗口输出抖动是否拉低实用检测精度。

实现方式：

- 使用已训练 checkpoint，对完整 validation sequence 按原始 1 ms stride 逐窗口输出 score。
- 对 score 做 moving average / EMA / 连续 K 个窗口投票。
- 同时报告 raw window metrics 和 smoothing 后 metrics。

优先配置：

- moving average：`5/10/20/50 ms`
- EMA：`alpha=0.1/0.2/0.4`
- 连续触发：`K=3/5/10`

记录文件：`02_sequence_smoothing.md`

### C. Slip onset 分析

目的：判断错误是否集中在滑移开始/结束边界附近。

实现方式：

- 对每个 sequence 的 `label/slip` 找 transition。
- 将窗口按距离最近 transition 的时间分桶：`0-10 ms`、`10-20 ms`、`20-50 ms`、`>50 ms`。
- 报告各桶 accuracy / balanced accuracy / F1 / ROC-AUC。
- 评估去掉 transition 周围不确定区间后的 upper-bound。

记录文件：`03_onset_analysis.md`

### D. Streaming SNN

目的：实现之前讨论的真正 streaming 方案，即每 1 ms 输入新的 event bin，模型状态跨 window 延续，而不是每次重新输入完整 20/50/100 ms window。

实现方式：

- 先训练仍采用 truncated BPTT：从连续 sequence 中截取 `T=64/128/256 ms` 片段，逐 ms 输入单帧 event bin。
- 模型每 1 ms 输出一次 score，loss 可使用最后一步或全序列平均。
- 推理时按完整 sequence streaming 运行，LIF 状态持续更新。

优先模型：

- `StreamingLiteSCNN`：复用 Lite-SCNN 的 conv/LIF/fc 结构，但支持状态输入输出和 per-step logits。
- 输出层优先使用非发放 logit readout，不强制最后一层 fire；如果 SNN hidden 已经足够事件驱动，最后分类层保留连续值更利于训练。

训练策略：

- 初始使用 BPTT / truncated BPTT，不做纯 STDP。
- 先测 `T=64`，若有效再扩展到 `T=128/256`。
- 使用 balanced 片段采样，避免 no-slip 多数类主导。

记录文件：`04_streaming_snn.md`

## 成功标准

本轮目标是尝试达到：

- 100k natural validation accuracy >= `90%`
- 或 sequence-level smoothed accuracy / balanced accuracy >= `90%`

如果仍未达到，需要明确指出是：

- 更长上下文仍不可分；
- smoothing 只能减少抖动但不能改变 ROC-AUC；
- onset 边界占错分比例过高；
- streaming BPTT 是否实际提高了 sequence-level 检测。

## 执行顺序

1. 实现动态 temporal context 数据读取和训练参数。
2. 跑 `ctx50/ctx100 FrameCNN`，先判断数据可分性上限是否明显提高。
3. 跑 `ctx50/ctx100 Lite-SCNN`，判断 SNN 是否能吃到上下文收益。
4. 对最好的 checkpoint 做完整 sequence smoothing 和 onset 分析。
5. 实现并训练 streaming SNN，对比 window 模型的 sequence-level 结果。

## 风险与注意事项

- 100 ms 输入会增大 I/O 和显存开销，先使用 `spatial_pool=4` 和 AMP。
- 动态长上下文会从 events 流重新 voxelize，训练速度可能低于原 20 ms。
- 90% 目标不保证可达；若 CNN upper-bound 仍明显低于 90%，则说明主要瓶颈不是 SNN 训练策略。

## 执行后状态

状态：本轮探索已完成。

实际执行范围超过原始 50/100 ms 计划，额外补充了：

- `ctx200/ctx300/ctx500/ctx1000_frame_cnn_v1`
- `ctx200/ctx300_lite_scnn_v1`
- `stream_lite_t128_v1`
- `stream_lite_t256_last_v1`
- `ctx500_frame_cnn_v1` 的 8 条和 16 条 sequence smoothing quick check

最强 window-level 结果为 `ctx500_frame_cnn_v1`：

- 100k natural accuracy：`88.72%`
- 100k balanced accuracy：`85.18%`
- balanced 100k ROC-AUC：`91.24%`

最强轻量 SNN 结果为 `ctx300_lite_scnn_v1`：

- 100k natural accuracy：`85.44%`
- 100k balanced accuracy：`80.09%`
- balanced 100k ROC-AUC：`86.57%`

最强 sequence quick check 为 `ctx500_frame_cnn_v1` + MA/EMA：

- 16 条 sequence 上最高 accuracy：`89.16%`
- 8 条容易 sequence 上可达 `95%` 以上，但该结果不能代表全验证集。

本轮未能在稳定全局指标上达到 `90%`。当前判断是：500 ms 左右上下文显著提高可分性，但还不足以可靠突破 90%；naive streaming BPTT 没有实际提高检测精度；后续应优先围绕 500 ms 上下文做轻量 SNN 结构/蒸馏和分组数据诊断。
