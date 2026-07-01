# Sliding Sequence Detection Iter01 计划

日期：2026-07-01

分支：`sliding`

## 背景

之前的 90% 结果来自固定随机 window 上的 score ensemble。这个口径能反映分类可分性，但实际滑移检测是持续运行的序列任务：模型每 1ms 接收新数据、输出当前 slip 状态，评价应考虑连续 slip 段、误报、漏报和检测延迟，而不仅是随机抽样 window 的逐点 accuracy。

本轮从“独立 window 分类”转向“连续滑动序列检测评估”。优先按数据集最初的 window size 进行检测；如果指标或行为不理想，再用当前最强的 500ms 上下文模型复核。

## 本轮目标

建立一个可复用的 sequence/sliding detection 评估口径，报告：

- 逐窗口 accuracy / balanced accuracy / F1 / ROC-AUC / PR-AUC
- causal smoothing / EMA / hysteresis 后的序列状态指标
- onset 检测数量、漏检数量、检测延迟
- false alarm 次数和 no-slip 段误报率
- per-sequence 指标，找出 hard sequences

本轮不优先追求新的训练方法，先确认评价协议和已有模型在连续序列上的真实行为。

## 探索项

### 01 原始 window size sliding 评估

先使用数据集最初 window size 对应的模型或最接近的 checkpoint，做完整 validation sequence 上的滑动检测评估。

关注问题：

- 逐点 window accuracy 是否和随机 100k 指标一致。
- causal smoothing / hysteresis 是否能减少抖动。
- onset delay 是否可接受。
- false alarm 是否集中在少数 sequence。

### 02 500ms 上下文 sliding 评估

若 01 的表现不足，使用目前最强的 500ms / 100-bin LIF-SNN 或 Iter10 best5 ensemble 做同样评估。

关注问题：

- 长上下文是否显著减少 false alarm。
- 是否牺牲 onset latency。
- best5 ensemble 的随机 window 优势是否能转化成连续序列检测优势。

### 03 指标与后续建议

整理两类口径：

- window-level strict 指标
- sequence-level detection 指标

给出下一步是否应训练真正 streaming LIF-SNN、是否需要 sequence-level voxel cache、是否需要 onset/offset 软标签。

## 成功标准

本轮成功标准不是必须达到 95%，而是建立可靠评价协议并给出可行动结论：

- 能在完整或足够大的 validation sequence 子集上跑通。
- 能报告 onset delay / missed onset / false alarm。
- 能判断原始 window size 与 500ms 上下文哪个更适合滑动检测。

## 远程路径

- 远程项目：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- 数据集：`/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0`
- 日志根目录：`/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter01`

## 本轮输出

- `01_original_window_sliding.md`
- `02_ctx500_sliding.md`
- `03_decision.md`
- `SUMMARY.md`
- `remote_summaries/`
