# Auto Exploration Iter04: Time-Channel SNN 精修计划

日期：2026-06-30

分支：`auto-snn-accuracy-exploration`

## 背景

Iter03 找到当前最强 SNN：

```text
iter03_time_channel_scnn_thr1_full_v1
```

指标：

- 100k random tuned accuracy：`88.98%`
- 100k balanced tuned accuracy：`85.45%`
- 100k random ROC-AUC：`90.24%`
- 100k balanced ROC-AUC：`90.35%`

距离 `90%` random tuned accuracy 只差约 `1.02` 个百分点。Iter04 不再大幅换结构，而是围绕 time-channel LIF-SCNN 做针对性精修。

## 假设

1. Time-channel SNN 已经接近目标，剩余差距可能来自 sampling 和 calibration。
2. Iter03 使用 balanced sampling，可能不是 natural random accuracy 的最优训练目标。
3. Distillation alpha 和正则化会影响 score separability 与阈值处的 recall/specificity tradeoff。
4. Time-channel SNN 与 wide3 / previous SNN 错误可能有互补性，SNN-only ensemble 可能把 random tuned 推过 `90%`。

## 探索项

### 01. Sampling / Prior Refinement

训练 time-channel SNN 的 random sampling 版本，检查是否提升 100k random accuracy。

候选：

- `iter04_time_channel_thr1_random_v1`
- `iter04_time_channel_thr1_random_distill_v1`

记录文件：`01_sampling_prior.md`

### 02. Distillation Alpha / Regularization

在 balanced sampling 下扫描蒸馏和正则化：

- `distill_alpha=0.1`
- `distill_alpha=0.5`
- `dropout=0.2`
- `label_smoothing=0.03`

记录文件：`02_distill_regularization.md`

### 03. SNN-only Ensemble

对 Iter03 最强 time-channel SNN、Iter03 wide3、历史 wide2/distill/ignore50 做 score ensemble。目标是利用模型互补性冲击 `90%` random tuned。

记录文件：`03_snn_ensemble.md`

### 04. Best 100k Evaluation

对最强单模型或 ensemble 做 100k random/balanced validation。若 quick/full 训练没有超过 Iter03，则只评估 ensemble。

记录文件：`04_best_eval.md`

## 成功标准

优先目标：

- 100k random validation tuned accuracy >= `90%`
- 或 100k balanced validation tuned accuracy >= `90%`

次级目标：

- 超过 Iter03 random tuned `88.98%`
- 超过 Iter03 balanced tuned `85.45%`
- ROC-AUC 超过 `91%`

## 远程输出

远程日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter04_time_channel_refine
```

本地备份：

```text
result/auto_exploration_iter04_time_channel_refine/remote_summaries/
```
