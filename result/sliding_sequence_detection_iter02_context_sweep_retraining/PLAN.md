# Sliding Sequence Detection Iter02 计划：Context Sweep 与重训练

日期：2026-07-01

分支：`sliding`

## 背景

Iter01 证明原始 window size 在连续序列检测上不稳定，而 500ms best5 SNN ensemble 加 causal MA50/debounce 可以达到 `89.78%` sequence-level accuracy。用户认为 500ms 可能过长，希望先探索更短 window/context 是否足够，再根据结果重新训练，以期达到更高综合精度。

本轮目标从单纯后处理搜索转为：确定合理 temporal context，并训练更适合连续滑移检测的 SNN。

## 本轮假设

1. 20ms 过短，误报 run 与状态抖动严重。
2. 500ms 有效但可能不是最优部署长度；`200-400ms` 可能在稳定性和计算量之间更合适。
3. 若使用相同模型容量、相同训练策略，不同 context 的 sequence-level 指标可以揭示主要收益来自上下文长度还是训练方法。
4. 若最佳 context 仍未稳定超过 90%，下一步应采用 sequence-aware 重训练，而不是只调后处理阈值。

## 探索项

### 01 Context Sweep 训练

训练同构 `time_channel_scnn`，context 分别为：

- `100ms / 50 bins`
- `200ms / 50 bins`
- `300ms / 75 bins`
- `400ms / 100 bins`
- `500ms / 100 bins`

初始训练预算：

- model width：32
- hidden dim：256
- sampling：random
- class weight：none
- ignore transition：50ms
- epochs：8
- train samples/epoch：60000
- val samples：20000
- scheduler：cosine
- AMP：开启

使用相同预算是为了先比较 context 本身，不把训练预算差异混入结论。

### 02 Context Sweep 序列评估

对 01 的每个 checkpoint 做 sequence-level sliding detection：

- validation split，固定 seed，选择 16 条完整 sequence
- score transform：raw
- smoothing：causal MA `3,5,10,20,50`
- EMA：`0.1,0.2,0.4`
- debounce：on `2,3,5`，off `2,3,5,10`
- threshold grid：101
- 输出每个 context 的 score cache 和 sliding detection JSON

主要指标：

- accuracy
- balanced accuracy
- F1
- segment recall
- false alarm runs/min
- p95 delay
- prediction switches

### 03 针对性重训练

根据 02 的结果选择最有希望的 context，优先考虑：

- 达到或接近 500ms accuracy 的最短 context
- 或在 false alarm 与 delay 间最平衡的 context

重训练策略：

- 加大 epoch 和 train samples
- 保持 SNN LIF，不改成 IAF
- 尝试 sequence-aware 的 window 训练近似：ignore/降权 transition band、label smoothing、margin loss、适度增大 capacity
- 重点目标是 sequence-level accuracy `>=90%`，并降低 false alarm 与 missed segments

### 04 总结与下一轮建议

整理 context sweep 与重训练结果，回答：

- 是否存在明显短于 500ms 的可用 context？
- 90% 是否能由更短 context 达到？
- 未达到 95% 的主要瓶颈是 context、训练目标、模型容量还是标签定义？
- 下一轮是否进入真正 streaming LIF-SNN。

## 成功标准

最低成功标准：

- 跑通 5 个 context 的训练和同口径 sequence-level 评估。
- 明确推荐一个后续 context 长度。
- 给出重训练结果和是否超过 `90%` 的判断。

目标成功标准：

- sequence-level accuracy `>=90%`。
- 同时 false alarm runs/min 不高于 Iter01 500ms best5 的 `0.325` 太多。
- 若未达到 95%，必须解释主要原因和下一步验证路线。

## 远程路径

- 远程项目：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- 数据集：`/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0`
- 日志根目录：`/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter02`

## 本轮输出

- `01_context_sweep_training.md`
- `02_context_sweep_sequence_eval.md`
- `03_targeted_retraining.md`
- `04_decision.md`
- `SUMMARY.md`
- `remote_summaries/`
