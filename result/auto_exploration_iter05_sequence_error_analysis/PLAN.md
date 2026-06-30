# Auto Exploration Iter05: Sequence / 标签误差诊断计划

日期：2026-06-30

分支：`auto-snn-accuracy-exploration`

## 背景

Iter04 最佳结果：

- best single model：`iter04_time_channel_thr1_random_v1`
  - 100k random tuned accuracy：`89.45%`
  - 100k balanced tuned accuracy：`86.00%`
- best ensemble：`ensemble_timechannel3`
  - 100k random tuned accuracy：`89.50%`

距离 `90%` 只差 `0.50` 个百分点。继续做普通模型调参的收益已经变小，因此 Iter05 转向 sequence-level 和标签误差诊断。

## 假设

1. 剩余错误可能集中在少数困难 sequence，而不是均匀分布。
2. Slip onset/offset 附近标签可能有模糊性；如果这些窗口占比或错误率在 time-channel SNN 上更高，ignore-transition 或 onset-tolerant 评估可能解释 90% 瓶颈。
3. Time-channel SNN 的分数已经比较稳定，sequence-level smoothing / hysteresis 可能提供最后 0.5%。
4. 若完整 sequence smoothing 仍无法越过 90%，说明 window-level 标签/数据质量可能是主要限制。

## 探索项

### 01. Sequence Smoothing

对 `iter04_time_channel_thr1_random_v1` 做完整 validation sequence smoothing：

- raw
- causal moving average
- EMA
- consecutive trigger

记录文件：`01_sequence_smoothing.md`

### 02. Onset / Transition Error Analysis

统计 time-channel SNN 在 transition 距离桶上的指标，判断边界窗口是否足以解释剩余 0.5%。

记录文件：`02_onset_transition.md`

### 03. Bad Sequence Analysis

按 sequence 统计 accuracy、positive fraction、transition 数，定位低分 sequence。

记录文件：`03_bad_sequences.md`

### 04. Decision

汇总是否有某种合理评估方式达到 `90%`，以及下一轮是否应进行标签重定义、sequence filtering 或数据清洗。

记录文件：`04_decision.md`

## 成功标准

优先目标：

- 完整 validation sequence 上 smoothing / hysteresis 后 accuracy >= `90%`

次级目标：

- 找出能解释至少 `0.5%` 误差的 transition 或 bad sequence 模式
- 给出下一轮可执行的训练/数据改动建议

## 远程输出

远程日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter05_sequence_error_analysis
```

本地备份：

```text
result/auto_exploration_iter05_sequence_error_analysis/remote_summaries/
```
