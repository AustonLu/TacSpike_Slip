# Auto Exploration Iter02: 训练目标与采样策略计划

日期：2026-06-30

分支：`auto-snn-accuracy-exploration`

## 背景

Iter01 的结论是后处理和 score ensemble 有小幅收益，但仍未达到 `90%`：

- Ensemble 3 random tuned accuracy：`88.08%`
- Ensemble 3 balanced tuned accuracy：`84.14%`
- Ensemble 3 ROC-AUC：约 `90.20%`

因此 Iter02 回到训练过程，重点检查采样策略、focal loss、margin regularization 和 random sampling 蒸馏是否能提高 score separability。

## 假设

1. 之前主模型主要用 balanced sampling 训练，可能没有直接优化自然分布下的 accuracy。
2. 当前 false positive / false negative tradeoff 对阈值敏感，random sampling 可能改善自然分布指标。
3. Focal loss 或 margin loss 可能提升 hard examples 的 score margin，从而提高 tuned accuracy 和 ROC-AUC。
4. 如果这些训练目标仍不能接近 `90%`，瓶颈更可能来自输入表示、SNN 时间信息保留能力或标签定义。

## 探索项

### 01. Random sampling full training

训练 `ctx500_tb100_wide2_scnn`，采样改为 `random`，不使用 class weight。目标是直接优化自然分布。

记录文件：`01_random_sampling.md`

### 02. Random sampling + distillation

同样使用 random sampling，但加入 CNN teacher distillation。目标是检查 teacher 是否能在自然分布训练中提升 SNN。

记录文件：`02_random_distill.md`

### 03. Focal loss / margin loss quick

在 balanced sampling 下快速测试：

- focal loss, gamma=2
- CE + margin regularization
- focal loss + distillation

记录文件：`03_focal_margin.md`

### 04. 最佳候选 100k 评估

对 Iter02 中最好的 checkpoint 做 100k random 和 balanced validation 评估，并与 Iter01/历史最好结果对比。

记录文件：`04_eval_best.md`

## 成功标准

优先目标：

- 100k random validation accuracy >= `90%`
- 或 100k balanced validation accuracy >= `90%`

次级目标：

- 超过 Iter01 Ensemble 3 的 random tuned `88.08%`
- 超过 Iter01 Ensemble 3 的 balanced tuned `84.14%`
- ROC-AUC 明显超过 `90.20%`

## 远程输出

远程日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter02_training_objectives
```

本地备份：

```text
result/auto_exploration_iter02_training_objectives/remote_summaries/
```
