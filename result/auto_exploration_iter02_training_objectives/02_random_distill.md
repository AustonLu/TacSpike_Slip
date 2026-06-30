# 02. Random Sampling + Distillation

状态：完成

## 目的

在 random sampling 训练上加入 CNN teacher distillation，检查 soft label 是否能提升 SNN 的 score ranking 和 recall。

## 配置

- 学生模型：`ctx500_tb100_wide2_scnn`
- teacher：`ctx500_tb100_frame_cnn_teacher_v1`
- sampling：`random`
- loss：`0.5 * CE + 0.5 * KL(student, teacher)`
- temperature：`2.0`
- epoch：`8`
- 每 epoch 训练样本：`60000`
- validation：`20000` random samples

远程 run：

```text
iter02_random_wide2_distill_v1
```

## 20k Validation 结果

| Run | Best epoch | Accuracy | Balanced acc | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `iter02_random_wide2_distill_v1` | 8 | 87.08% | 81.95% | 75.53% | 88.85% | 80.92% |

混淆矩阵：

- TP：`3989`
- TN：`13427`
- FP：`877`
- FN：`1707`

## 观察

相比纯 random CE：

- Accuracy：`86.97% -> 87.08%`，小幅提升。
- Balanced accuracy：`80.33% -> 81.95%`，提升更明显。
- Recall：`64.91% -> 70.03%`，distillation 让模型更愿意预测 slip。
- ROC-AUC：`88.95% -> 88.85%`，没有提升。

蒸馏改善了 operating point，但没有提高整体排序能力。也就是说，在阈值可调的 100k 评估上，它未必会优于纯 random CE。

## 结论

Random + distillation 对 recall 有帮助，但不是 Iter02 的最佳 100k tuned accuracy 方案。它提示 teacher 可以改善分类偏置，但当前 teacher-student 设置没有充分提升 score separability。
