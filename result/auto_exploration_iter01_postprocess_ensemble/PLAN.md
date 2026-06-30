# Auto Exploration Iter01: 后处理与集成验证计划

日期：2026-06-29

分支：`auto-snn-accuracy-exploration`

## 背景

上一轮最好 SNN：

- `ctx500_tb100_wide2_scnn_distill_v1`
  - 100k random tuned accuracy：`87.58%`
  - 100k balanced tuned accuracy：`83.50%`
  - 100k balanced ROC-AUC：`89.60%`
- `ctx500_tb100_wide2_scnn_ignore50_v1`
  - 100k random tuned accuracy：`87.68%`
  - 100k balanced tuned accuracy：`83.22%`
  - 100k balanced ROC-AUC：`89.46%`

还没有达到 `90%`。本轮先不重新训练，优先验证现有模型的输出是否可以通过 sequence smoothing、阈值校准或 score ensemble 提升到 `90%`。如果不行，下一轮再进入模型结构和训练目标改造。

## 假设

1. 当前模型的 window-level score 已经有一定可分性，错误可能部分来自逐窗抖动。
2. sequence-level smoothing 可能提升自然序列上的 accuracy，但可能牺牲 onset 延迟。
3. `distill` 和 `ignore50` 模型错误模式可能不完全一致，score ensemble 可能提高 random/natural accuracy。
4. 如果后处理和集成都不能接近 `90%`，说明主要瓶颈仍在模型表示、标签噪声或数据上限，而不是阈值。

## 探索项

### 01. 完整 sequence smoothing

对 `ctx500_tb100_wide2_scnn_distill_v1` 和 `ctx500_tb100_wide2_scnn_ignore50_v1` 在完整 validation sequences 上评估：

- raw
- causal moving average
- EMA
- consecutive trigger
- onset bucket
- detection delay

记录文件：`01_sequence_smoothing.md`

### 02. Score ensemble

实现并评估现有 checkpoint 的 score averaging：

- distill + ignore50
- distill + ignore50 + deep distill
- 与单模型对比 100k random / balanced

记录文件：`02_score_ensemble.md`

### 03. 阈值与类别先验分析

比较：

- default threshold
- accuracy-tuned threshold
- balanced-accuracy-tuned threshold
- random vs balanced sampling 的阈值偏移

记录文件：`03_threshold_prior.md`

## 成功标准

优先目标：

- 100k random validation accuracy >= `90%`
- 或 100k balanced validation accuracy >= `90%`

次级目标：

- 明确 smoothing/ensemble 是否可以稳定超过单模型 `87.68%` random tuned accuracy。
- 判断下一轮是否应继续做后处理，还是回到训练目标/结构改造。

## 预计输出

- 本轮所有远程 JSON 保存到 `remote_summaries/`。
- 每个探索项一个 markdown。
- `SUMMARY.md` 总结本轮结论、原因和下一轮建议。
