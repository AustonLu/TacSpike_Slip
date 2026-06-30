# 02. Score Ensemble

状态：完成

## 目的

验证不同 SNN checkpoint 的错误模式是否互补。如果互补，平均 slip-vs-no-slip margin 应该提升 100k random 或 balanced validation accuracy。

## 实现

新增评估脚本：

```bash
scripts/train/evaluate_checkpoint_ensemble.py
```

做法：

1. 对同一批采样 index 分别运行多个 checkpoint。
2. 取每个模型的 `score = logit_slip - logit_no_slip`。
3. 对 score 求平均。
4. 计算 default threshold 和 tuned threshold 指标。

## 配置

Ensemble 2：

- `ctx500_tb100_wide2_scnn_distill_v1`
- `ctx500_tb100_wide2_scnn_ignore50_v1`

Ensemble 3：

- `ctx500_tb100_wide2_scnn_distill_v1`
- `ctx500_tb100_wide2_scnn_ignore50_v1`
- `ctx500_tb100_deep_scnn_distill_v1`

每项评估：

- split：`val`
- samples：`100000`
- sampling：`random` 或 `balanced`
- tuned threshold metric：`accuracy`

## 结果

| 模型 | Sampling | Default acc | Tuned acc | Default balanced acc | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Ensemble 2 | random | 87.51% | 87.76% | 83.57% | 89.84% |
| Ensemble 2 | balanced | 83.54% | 83.54% | 83.54% | 89.75% |
| Ensemble 3 | random | 87.84% | 88.08% | 83.82% | 90.27% |
| Ensemble 3 | balanced | 83.80% | 84.14% | 83.80% | 90.20% |

对比上一轮最好单模型：

| 模型 | Random tuned acc | Balanced tuned acc | Balanced ROC-AUC |
|---|---:|---:|---:|
| `wide2_distill` | 87.58% | 83.50% | 89.60% |
| `wide2_ignore50` | 87.68% | 83.22% | 89.46% |
| Ensemble 3 | 88.08% | 84.14% | 90.20% |

## 结论

Ensemble 有稳定但有限的提升：

- random tuned accuracy 从 `87.68%` 提升到 `88.08%`
- balanced tuned accuracy 从 `83.50%` 提升到 `84.14%`
- balanced ROC-AUC 从 `89.60%` 提升到 `90.20%`

但仍未达到 `90%` accuracy。说明不同模型确实有一定互补性，但互补不足以突破目标。下一轮可以把 ensemble 作为诊断工具，而不是最终方案；更关键的是训练出分数可分性更强的单模型。
