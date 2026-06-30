# 04. 最佳候选 100k 评估

状态：完成

## 目的

用统一的 100k random / balanced validation 口径复评 Iter02 中最有希望的两个候选：

- `iter02_random_wide2_v1`
- `iter02_random_wide2_distill_v1`

阈值按 accuracy 搜索，以便和前几轮 tuned accuracy 对齐。

## 结果

| Run | Sampling | Default acc | Tuned acc | Balanced acc at tuned | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| `iter02_random_wide2_v1` | random 100k | 87.19% | 87.46% | 82.35% | 88.88% | 81.95% |
| `iter02_random_wide2_v1` | balanced 100k | 80.64% | 83.28% | 83.28% | 88.89% | 90.35% |
| `iter02_random_wide2_distill_v1` | random 100k | 87.17% | 87.22% | 81.69% | 88.81% | 81.33% |
| `iter02_random_wide2_distill_v1` | balanced 100k | 82.07% | 82.79% | 82.79% | 88.81% | 90.06% |

## 对比历史最好

| 方法 | Random tuned acc | Balanced tuned acc | ROC-AUC |
|---|---:|---:|---:|
| Iter01 Ensemble 3 | 88.08% | 84.14% | 约 90.20% |
| Stage2 best SNN single | 87.68% | 83.50% | 约 89.60% |
| Iter02 best | 87.46% | 83.28% | 88.89% |

## 观察

Iter02 的 best 100k random tuned accuracy 是 `87.46%`，比 Iter01 ensemble 低 `0.62` 个百分点，也略低于此前 best SNN single 的 `87.68%`。

Random sampling 可以把默认阈值下的 natural accuracy 训练得更自然，但 ROC-AUC 没有提高，说明 ranking/separability 没有增强。Balanced 100k 指标也没有超过前一轮。

## 结论

Iter02 没有达到 `90%`，也没有刷新当前最佳。下一轮不应继续只围绕 focal/margin/random sampling 小幅调参，而应尝试更直接的输入表示或 SNN 结构改动。
