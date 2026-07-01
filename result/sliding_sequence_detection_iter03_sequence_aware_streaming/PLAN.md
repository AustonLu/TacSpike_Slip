# Sliding Sequence Detection Iter03 计划：Sequence-Aware / Streaming 训练

日期：2026-07-01

分支：`sliding`

## 背景

Iter02 证明 `400ms / 100 bins` 是当前最合理的 sliding-window context，sequence accuracy `89.714%`，几乎追平 500ms，同时 false alarm/min 和 p95 delay 更好。但 targeted retraining 失败，说明简单加容量、label smoothing、margin loss 仍然是 window-level 训练思路，不能解决连续状态检测目标。

本轮目标是把训练目标从独立 window classification 转向连续状态检测。长期目标是逼近 `95%` 综合精度，但本轮首先要验证 sequence-aware/streaming 训练是否能缩小与 95% 的差距。

## 关键判断

已有 `train_streaming_scnn.py` 支持每 1ms 输入新事件 bin、LIF 状态跨时间延续和 truncated BPTT。但历史 naive streaming 结果较弱，原因可能是：

- 每步只有 1ms 输入，状态学习难度大。
- loss 没有处理 onset/offset 标签噪声。
- 没有显式惩罚状态抖动或 false alarm run。
- 没有完整 sequence-level 评估闭环。

因此本轮不只是“跑 streaming”，而是加入连续状态检测目标。

## 探索项

### 01 改造 Streaming 训练目标

在 `scripts/train/train_streaming_scnn.py` 上扩展：

- transition ignore band：onset/offset 附近若干 ms 不计入 CE。
- temporal smoothness loss：惩罚相邻 score 的大幅变化。
- flip penalty：用相邻预测概率差近似惩罚状态频繁切换。
- positive/negative class weighting：处理 segment 内类别不平衡。
- warmup steps：片段前若干 ms 只用于建立状态，不计入 supervised loss。

保持 LIF，不使用 IAF。

### 02 Streaming 训练实验

先跑 3 个 streaming 变体：

| run id | segment steps | loss |
|---|---:|---|
| `stream_t400_all_ignore25_smooth_v1` | 400 | all valid steps CE + ignore25 + smooth |
| `stream_t400_tail200_ignore25_smooth_v1` | 400 | 只监督后 200ms + ignore25 + smooth |
| `stream_t512_tail256_ignore50_smooth_v1` | 512 | 只监督后 256ms + ignore50 + smooth |

训练期指标只作为参考，最终以完整 validation sequence sliding/streaming 指标为准。

### 03 Streaming Sequence-Level 评估

新增或扩展评估脚本，对完整 validation sequence 逐 ms streaming 推理：

- 每条 sequence 开始时 reset LIF state。
- 每 1ms 输入新 event bin。
- 输出每 ms score。
- 使用与 Iter02 相同的 causal MA/EMA/debounce 指标。

报告：

- accuracy
- balanced accuracy
- F1
- segment recall
- missed slip segments
- false alarm runs/min
- p95 delay
- prediction switches

### 04 若 Streaming 不足，做 Sliding-Window Sequence-Aware 训练

如果 streaming 从零训练仍明显低于 `400ms` sliding baseline，则补做一个 400ms sliding-window sequence-aware 训练：

- 使用 `400ms / 100 bins`
- 训练样本按 transition distance 采样或加权
- loss 降低 onset/offset 附近硬标签权重
- 目标仍然在 sequence-level 指标上超过 Iter02 的 `89.714%`

### 05 总结与迭代

若任一方案达到或接近 95%，继续围绕该方向扩大验证集；若未达到，记录瓶颈并给出下一轮可执行策略。

## 成功标准

最低成功标准：

- 跑通 sequence-aware/streaming 训练。
- 跑通完整 sequence-level streaming 评估。
- 明确 streaming 是否优于 Iter02 的 `400ms` sliding baseline。

目标成功标准：

- sequence-level accuracy 达到 `95%`。
- 如果未达 95%，至少超过 Iter02 best `89.733%` 或明确证明当前 streaming 方向不成立。

## 远程路径

- 远程项目：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- 数据集：`/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0`
- 日志根目录：`/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter03`

## 本轮输出

- `01_streaming_training_objective.md`
- `02_streaming_training.md`
- `03_streaming_sequence_eval.md`
- `04_sliding_sequence_aware_fallback.md`
- `05_decision.md`
- `SUMMARY.md`
- `remote_summaries/`
