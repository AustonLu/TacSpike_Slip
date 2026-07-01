# Sliding Sequence Detection Iter04 计划：连续状态目标

日期：2026-07-01

## 背景

Iter02/Iter03 的最好 strict sequence accuracy 约为 `89.9%`。目前最优方法仍是 `400ms` sliding-window SNN score，再做 moving average / debounce。Iter03 已证明：

- 纯 1ms streaming LIF-SNN 从零训练不足，达不到 window SNN 的水平。
- sequence-aware fine-tune 只带来小幅提升。
- 简单状态解码和 ensemble 不能突破 `90%`。

因此本轮不再只把每个 window 当作独立分类样本，而是把任务明确建模为连续 slip state 检测。

## 成功标准

主标准仍沿用严格口径：

- validation selected-sequence strict per-window sequence accuracy 达到或超过 `95%`。

同时必须记录：

- balanced accuracy / F1；
- segment recall；
- false alarms per minute；
- onset delay p95；
- worst sequences。

若未达到 `95%`，需要判断瓶颈来自模型容量、状态解码、标签边界还是数据/指标上限。

## 实验项

1. 标签与可达上限审计
   - 统计 validation 序列的状态段、transition 周边窗口占比、不同 transition tolerance 下的 oracle/smoothed upper-bound。
   - 目标：判断 95% 是否主要受边界定义、长延迟或少数坏序列限制。

2. 连续状态后处理扩展
   - 在现有最优 SNN score cache 上搜索更丰富的状态机参数：
     - causal / centered moving average；
     - asymmetric onset/offset tolerance；
     - minimum slip duration；
     - gap fill；
     - Viterbi switch cost。
   - 目标：确认不改模型时是否还能突破 `90%`，以及是否存在 95% 可达空间。

3. 轻量序列状态头
   - 冻结或半冻结 Iter02/Iter03 SNN score extractor，训练一个小型 causal temporal state head。
   - 输入：一个或多个 SNN score 序列及其低维动态特征。
   - 输出：每 1ms 的连续 slip state。
   - 目标：用极轻量的序列模型替代手工 MA/debounce，验证“状态目标”是否能显著提升。

4. 状态目标 SNN fine-tune
   - 从当前最优 `ctx400` / Iter03 sequence fine-tune checkpoint 继续训练。
   - 强化连续状态损失、transition ignore/tolerance、smoothness/flip penalty。
   - 目标：确认 end-to-end SNN 微调是否优于单独状态头。

## 本轮输出

目录：

```text
result/sliding_sequence_detection_iter04_state_objective/
```

至少包含：

- `PLAN.md`
- `01_label_and_upper_bound_audit.md`
- `02_state_postprocess_search.md`
- `03_temporal_state_head.md`
- `04_state_objective_finetune.md`
- `SUMMARY.md`
- `remote_summaries/`

## 远程执行

默认远程配置：

```text
ssh -J fics jiajunlu@192.168.68.198
PROJECT_DIR=/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2
DATA_ROOT=/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0
LOG_ROOT=/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter04
```

## 提交策略

按 workflow，本轮完成后提交并 push 到 GitHub。提交范围只包含本轮代码、脚本和结果文档，不纳入与本轮无关的已有未提交文件。
