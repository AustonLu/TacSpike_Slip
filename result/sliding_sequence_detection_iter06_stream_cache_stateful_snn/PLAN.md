# Sliding Sequence Detection Iter06 计划：Stream Cache 与 Stateful SNN

## 0. 本轮状态

当前分支：`sliding`

本轮只写计划，不改训练代码、不构造数据、不启动远程实验。待用户 review 通过后，再按本文档执行。

上一轮 Iter05 的最佳结果：

```text
probe_seg1024_no_smooth/best.pt
strict accuracy = 90.108%
balanced accuracy = 88.044%
F1 = 81.934%
segment recall = 95.0%
delay p95 = 678.7 ms
```

上一轮结论是：长 segment、去掉 smoothness/flip penalty、full sliding checkpoint selection 能避免 fine-tune 崩溃，但仍没有突破约 `90%` 的平台。因此，本轮不再继续堆当前的重复 window fine-tune，而是把训练目标改成面向连续序列的 stream/state detection。

## 1. 本轮核心假设

### 假设 A：当前平台主要来自训练目标和在线任务不一致

当前模型虽然用 sliding validation 评估，但训练仍大量依赖 window 或短 segment。真实任务是每 1 ms 持续运行的状态检测，模型应学习连续状态，而不是独立窗口分类。

预期改进：

- 降低窗口边界噪声导致的误判。
- 改善 slip onset 附近的状态连续性。
- 提高 full-sequence strict accuracy、balanced accuracy 和 F1。

### 假设 B：需要从原始 HDF5 派生 stream cache，但不应重新生成重复 window 数据集

当前 400/500 ms context 的训练和评估会重复 voxelize 大量重叠窗口，效率很低。新的派生数据应以 sequence 为单位缓存 1 ms event bin，而不是缓存每个重叠 window。

预期改进：

- 显著降低数据读取和 voxelization 成本。
- 支持不同 chunk length、BPTT length、context length 的快速实验。
- 为真正 stateful streaming SNN 训练提供统一输入格式。

### 假设 C：最后一层使用 non-spiking membrane/logit readout 可能比 spike-count head 更适合连续检测

连续状态检测需要稳定可校准的 slip score。最后一层不一定必须发放 spike；使用 membrane/logit readout 可以直接做 threshold、EMA 和 debounce。

预期改进：

- 降低输出量化带来的阈值敏感性。
- 改善 ROC/PR threshold selection。
- 更容易控制 false alarm 和 detection delay。

## 2. 本轮成功标准

主要目标：

```text
validation full-sequence strict accuracy >= 95.0%
```

阶段性可接受目标：

```text
strict accuracy > Iter04 best 90.198%
balanced accuracy >= 89.0%
F1 >= 83.0%
segment recall >= 95.0%
delay p95 不显著劣于 Iter04 best 的 237.5 ms
```

如果 strict accuracy 没有达到 95%，也需要明确回答：

- 是 stream cache/data pipeline 的问题；
- 是 stateful SNN 结构不足；
- 是训练目标或 onset 标签噪声问题；
- 还是数据集存在跨 sequence 分布偏移，导致 95% 目标在当前标签定义下不稳定。

## 3. 本轮目录结构

本轮结果目录：

```text
result/sliding_sequence_detection_iter06_stream_cache_stateful_snn/
```

计划执行后应包含：

```text
PLAN.md
01_stream_cache_design_and_sanity.md
02_stateful_scnn_training.md
03_onset_mask_and_soft_label.md
04_readout_and_postprocess.md
05_error_analysis_and_calibration.md
SUMMARY.md
remote_summaries/
```

其中本次仅创建 `PLAN.md`。其余文件在实验执行时逐项补充。

## 4. 探索项 1：构造面向 stream 的 sequence cache

### 目标

从原始 TacSpike HDF5 数据派生 sequence-level stream cache。该 cache 不是重新生成重叠 window，而是每条 sequence 只保存连续 1 ms event bin 和对应状态标签。

### 建议格式

每条 sequence 保存为一个压缩文件，例如 `.npz` 或 `.h5`：

```text
event_bins: [T_ms, 2, 32, 32]
labels:     [T_ms]
valid_mask: [T_ms]
onset_mask: [T_ms]
meta:
  sequence_id
  split
  length_ms
  slip_ratio
  original_h5_path
```

其中：

- `event_bins`：从原始 events 按 1 ms bin 聚合，先做 4x4 spatial sum pooling，将 `128x128` 降到 `32x32`。
- `labels`：每 1 ms 的 no-slip/slip 状态。
- `valid_mask`：排除无标签、边界不确定或超出有效范围的时刻。
- `onset_mask`：标记 slip onset 附近区域，用于 ignore loss 或 soft label。

### 需要验证的 sanity check

- cache sequence 数量与 manifest split 一致。
- 每条 sequence 的时间长度与原始 HDF5 window/label 对齐。
- cache 的 slip ratio 与原始统计接近。
- 随机抽样可视化 1 ms event bin 和标签轨迹。
- 随机抽样对比原始动态 voxelize 与 cache 切片结果，确认数值一致。

### 预期输出

```text
01_stream_cache_design_and_sanity.md
remote_summaries/stream_cache_sanity.json
```

## 5. 探索项 2：训练真正的 stateful streaming SNN

### 目标

训练时按连续 chunk 输入模型，LIF 状态在 chunk 内保留，只在 sequence 边界 reset。模型每 1 ms 输出一次 slip score。

### 基础输入

```text
x: [B, L, 2, 32, 32]
y: [B, L]
mask: [B, L]
```

候选 chunk length：

```text
L = 256, 384, 512
```

首轮优先：

```text
L = 384
```

原因：500 ms 已经较长且训练慢，384 ms 在上下文和效率之间更均衡；如果 384 ms 不足，再回到 512 ms。

### 候选模型

主模型：

```text
Input [B, L, 2, 32, 32]
Conv(2, 32, k=5, s=1, p=2) + LIF
Conv(32, 64, k=3, s=2, p=1) + LIF
AdaptiveAvgPool2d(4x4)
Linear(64*4*4, 128) + LIF
Linear(128, 1) non-spiking membrane/logit readout
Output [B, L]
```

保守对照：

```text
Conv width 16/32, hidden 64
```

增强对照：

```text
Conv width 32/64, hidden 256
dropout 0.1
```

### 训练方法

```text
optimizer: AdamW
lr: 1e-4, 3e-4
weight_decay: 1e-4
loss: masked BCEWithLogitsLoss
class balance: positive weight or balanced chunk sampling
BPTT: truncated BPTT over chunk length L
state reset: sequence boundary only
gradient_clip: 1.0
mixed precision: enabled if stable
```

### checkpoint 选择

不能只用 sampled validation。每个候选 checkpoint 必须通过 full sequence sliding/stream validation 选择。

### 预期输出

```text
02_stateful_scnn_training.md
remote_summaries/stateful_scnn_*_summary.json
remote_summaries/stateful_scnn_*_eval.json
```

## 6. 探索项 3：处理 onset 标签噪声

### 目标

slip onset 附近的标签边界可能存在标注或物理过渡不确定性。直接用 hard label 训练可能惩罚合理的提前/滞后响应。本项探索 onset mask 和 soft label。

### 候选设置

```text
ignore_around_onset = 0 ms, 30 ms, 50 ms
soft_label_ramp = none, 50 ms
```

优先顺序：

1. `ignore_around_onset = 30 ms`
2. `ignore_around_onset = 50 ms`
3. `soft_label_ramp = 50 ms`

### 判断标准

- 如果 strict accuracy 提升但 delay 明显变差，说明模型变得过度保守。
- 如果 balanced accuracy/F1 提升且 delay 不恶化，说明 onset 噪声处理有效。
- 如果指标无变化，说明主要瓶颈不在 onset 边界。

### 预期输出

```text
03_onset_mask_and_soft_label.md
remote_summaries/onset_mask_*_summary.json
remote_summaries/onset_mask_*_eval.json
```

## 7. 探索项 4：readout 与在线后处理

### 目标

对 stateful SNN 的每 1 ms slip score 做因果后处理，寻找 accuracy、false alarm 和 delay 的折中。

### 候选 readout

```text
non-spiking logit readout
output membrane readout
spike-count readout
```

首选：

```text
non-spiking logit readout
```

### 候选后处理

```text
raw score threshold
EMA alpha = 0.02, 0.05, 0.1
moving average window = 20, 50, 80, 100 ms
debounce-on-k = 2, 3, 5
debounce-off-k = 10, 20, 30
threshold grid selected on validation set
```

### 判断标准

本项不能只追求 strict accuracy，也要同时报告：

- balanced accuracy
- F1
- false alarm
- segment recall
- detection delay p50/p95

### 预期输出

```text
04_readout_and_postprocess.md
remote_summaries/readout_postprocess_*_eval.json
```

## 8. 探索项 5：错误分析与 per-sequence calibration

### 目标

如果 stateful SNN 仍卡在 90% 左右，需要判断是否存在跨 sequence 分布偏移，或少数 sequence 主导错误。

### 分析内容

- 每条 sequence 的 accuracy、balanced accuracy、F1。
- 每条 sequence 的 slip ratio、event count、空 bin 比例。
- 错误最多的 sequence 列表。
- false positive 主要发生在 slip 前、slip 后还是全程 no-slip。
- false negative 主要发生在 onset 附近还是持续 slip 段。
- score 分布是否存在 sequence-wise bias。

### 可尝试的轻量 calibration

只在 validation 上探索，不直接使用 test：

```text
global threshold
split-level threshold
per-sequence normalization using pre-slip baseline
causal running z-score
```

注意：per-sequence calibration 必须保持因果，不能使用未来标签或完整 sequence 的全局统计。

### 预期输出

```text
05_error_analysis_and_calibration.md
remote_summaries/per_sequence_error_table.csv
remote_summaries/calibration_eval.json
```

## 9. 执行顺序

review 通过后，建议按以下顺序执行：

1. 实现并验证 stream cache。
2. 用 stream cache 复现当前 best model 的 full sequence evaluation，确认 cache 没有改变指标定义。
3. 训练 `L=384` 的 stateful SNN，使用 non-spiking logit readout。
4. 对 `L=256/512` 做上下文长度对照。
5. 加入 onset ignore/soft label。
6. 做 readout/postprocess sweep。
7. 做 per-sequence error analysis 和必要的 causal calibration。
8. 写 `SUMMARY.md`，提交并 push。

## 10. 风险和停止条件

### 风险 1：stream cache 与原始 sliding window 指标不一致

处理方式：先做 cache sanity check，并用同一 checkpoint 对比 cache evaluation 与原始 evaluation。

### 风险 2：stateful SNN 训练不稳定

处理方式：

- 先使用较小 LR。
- 使用 gradient clipping。
- 从现有 SCNN checkpoint 初始化卷积层。
- 必要时先冻结 conv backbone，只训练 temporal/readout 层。

### 风险 3：更长 BPTT 带来显存和速度压力

处理方式：

- 从 `L=256/384` 开始。
- 使用 batch size 1 或 2。
- 使用 AMP。
- 优先优化 cache 和 DataLoader，而不是直接扩大模型。

### 停止条件

本轮满足任一条件即可总结：

- 达到 `strict accuracy >= 95.0%`。
- 明确超过 Iter04 best，且进一步提升需要新的数据/标签策略。
- 多个 stateful 设置仍停在约 `90%`，并通过错误分析证明主要瓶颈来自数据/标签或跨 sequence 分布偏移。

## 11. 预期最终总结格式

本轮结束时 `SUMMARY.md` 至少包含：

- 最佳 run id 和 checkpoint。
- full-sequence validation 指标。
- 与 Iter04/Iter05 best 的对比。
- 是否达到 95%。
- stream cache 是否与原始 evaluation 一致。
- stateful SNN 是否优于 window/segment 训练。
- onset mask/soft label 是否有效。
- readout/postprocess 的最佳配置。
- 下一轮建议。
