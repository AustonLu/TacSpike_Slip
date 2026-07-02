# Sliding Sequence Detection Iter05 计划：长序列状态微调

日期：2026-07-02

## 背景

Iter04 将任务显式改为连续 slip state detection。最佳结果是三路 causal temporal state head：

```text
strict accuracy = 90.198%
balanced accuracy = 88.503%
F1 = 82.288%
segment recall = 95.0%
```

但直接状态目标 SNN fine-tune 失败，统一 sliding validation 只有 `81.776%`。失败现象是 recall 和 segment recall 大幅下降，模型过于保守。

## 本轮假设

Iter04 fine-tune 失败不是说明“上游 SNN extractor 不可微调”，而是训练目标和验证选择方式不合适：

1. `segment_windows=64` 只有 64ms，明显短于当前 `400ms` context 和连续滑移状态变化尺度。
2. `smoothness_weight=0.001` 与 `flip_penalty_weight=0.01` 可能让模型过度偏向不切换，从而漏检 slip。
3. 用短片段 `valid_accuracy` 选 checkpoint 会偏向多数类/保守预测，不能代表 full sliding sequence strict accuracy。

## 实验项

### 01 长 segment 微调

对比：

- `512ms` segment；
- `1024ms` segment。

二者都从 Iter03 最优 `seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt` 初始化。

### 02 去除或降低平滑/切换惩罚

对比：

- no smooth/no flip：`smoothness_weight=0`，`flip_penalty_weight=0`；
- low smooth/no flip：`smoothness_weight=0.0001`，`flip_penalty_weight=0`。

### 03 用 full sliding 口径选 checkpoint

训练时保存每个 epoch checkpoint。训练完成后：

- 先评估各 run 的 `best.pt`；
- 如果 `best.pt` 不理想，再评估前 4 个 epoch checkpoint；
- 最终按 16 条 validation selected sequence 的 full sliding strict accuracy、balanced accuracy、F1 和事件级指标判断，而不是按训练脚本内部短片段 validation。

## 运行配置

远端默认：

```text
ssh -J fics jiajunlu@192.168.68.198
PROJECT_DIR=/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2
DATA_ROOT=/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0
LOG_ROOT=/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter05
```

训练 run：

| run id | segment | smooth | flip | transition ignore | best metric |
|---|---:|---:|---:|---:|---|
| `ft_ctx400_seg512_no_smooth` | 512 | 0 | 0 | 0 | valid_balanced_accuracy |
| `ft_ctx400_seg512_low_smooth` | 512 | 0.0001 | 0 | 0 | valid_balanced_accuracy |
| `ft_ctx400_seg1024_no_smooth` | 1024 | 0 | 0 | 0 | valid_balanced_accuracy |
| `ft_ctx400_seg1024_ignore50_no_smooth` | 1024 | 0 | 0 | 50 | valid_balanced_accuracy |

## 成功标准

主目标仍为：

```text
validation selected sequence strict accuracy >= 95%
```

阶段性有效信号：

- 超过 Iter04 最佳 `90.198%`；
- segment recall 保持 `>=95%`；
- onset delay p95 不劣于约 `237ms`；
- false alarm runs/min 不显著恶化。

## 输出

```text
result/sliding_sequence_detection_iter05_long_segment_finetune/
```

包含：

- `PLAN.md`
- `01_long_segment_training.md`
- `02_low_penalty_training.md`
- `03_full_sliding_checkpoint_selection.md`
- `SUMMARY.md`
- `remote_summaries/`
