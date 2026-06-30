# 01. Random Sampling Full Training

状态：完成

## 目的

检查 balanced sampling 是否和最终关注的 natural accuracy 存在目标不一致。Iter02 训练一个 `ctx500_tb100_wide2_scnn`，使用 `random` sampling 和无 class weight，直接拟合自然分布。

## 配置

- 模型：`lite_scnn`
- 输入：`context_ms=500`，`time_bins=100`
- SNN：LIF，`threshold=0.1`
- 宽度：conv `32/64`，hidden `256`
- readout：`logit_mean`
- readout 起点：后半段，`readout_start_frac=0.5`
- loss：CE
- sampling：`random`
- epoch：`8`
- 每 epoch 训练样本：`60000`
- validation：`20000` random samples

远程 run：

```text
iter02_random_wide2_v1
```

## 20k Validation 结果

| Run | Best epoch | Accuracy | Balanced acc | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `iter02_random_wide2_v1` | 8 | 86.97% | 80.33% | 73.93% | 88.95% | 81.49% |

混淆矩阵：

- TP：`3697`
- TN：`13696`
- FP：`608`
- FN：`1999`

## 观察

Random sampling 明显把模型推向自然分布 accuracy：20k random validation 达到 `86.97%`，接近之前 SNN 单模型最好 100k random tuned `87.68%`，但没有超过。

主要问题是 recall 仍偏低，20k validation recall 约 `64.91%`，false negative 仍然多。这说明 random sampling 虽然减少了 false positive，但牺牲了 slip 类召回，balanced accuracy 没有同步提升。

## 结论

Random sampling 是有用的自然分布优化方式，但不是突破 `90%` 的充分条件。它没有改善 score separability 到足以超过 Iter01 ensemble。
