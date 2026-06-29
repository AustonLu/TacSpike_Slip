# CNN Teacher Distillation 实验

状态：完成

目的：用 CNN teacher 的 soft score 约束 SNN，检查是否比硬标签 CE 更接近 CNN upper-bound。

## Teacher

本轮主线使用 `500 ms / 100 bins`，因此训练了输入时间维度匹配的 teacher：

```text
ctx500_tb100_frame_cnn_teacher_v1
```

Teacher 结果：

| Run | 参数量 | Best epoch | 20k balanced val acc | ROC-AUC |
|---|---:|---:|---:|---:|
| `ctx500_tb100_frame_cnn_teacher_v1` | 560,450 | 7 | 85.32% | 91.31% |

蒸馏损失：

```text
loss = (1 - alpha) * CE + alpha * KL(student / T, teacher / T) * T^2
alpha = 0.5
T = 2.0
```

## SNN Distillation 结果

| Run | 参数量 | Best epoch | 20k balanced val acc | ROC-AUC |
|---|---:|---:|---:|---:|
| `ctx500_tb100_wide2_scnn_distill_quick_v1` | 282,688 | 4 | 79.96% | 85.95% |
| `ctx500_tb100_wide2_scnn_distill_v1` | 282,688 | 6 | 84.07% | 90.32% |
| `ctx500_tb100_deep_scnn_distill_quick_v1` | 619,522 | 4 | 82.48% | 88.33% |
| `ctx500_tb100_deep_scnn_distill_v1` | 619,522 | 7 | 83.74% | 90.77% |

## 100k 复评

| Run | Sampling | Default acc | Tuned acc | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| `ctx500_tb100_wide2_scnn_distill_v1` | random | 86.99% | 87.58% | 89.65% | 82.32% |
| `ctx500_tb100_wide2_scnn_distill_v1` | balanced | 83.46% | 83.50% | 89.60% | 90.65% |

## 结论

蒸馏对完整训练有小幅收益：`wide2 distill` 的 20k balanced val `84.07%` 高于 `wide2 ignore50` 的 `83.61%`，100k balanced 也达到本轮最高 `83.50%`。但是 teacher 自身也只有 `85.32%` balanced val，且 SNN 蒸馏后仍未达到 `90%`，说明 teacher-student 不是单独突破瓶颈的手段。
