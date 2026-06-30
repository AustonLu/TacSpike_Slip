# Auto Exploration Iter01 总结

日期：2026-06-30

## 本轮目标

验证在不重新训练模型的情况下，后处理和 score ensemble 是否可以把 SNN accuracy 推到 `90%`。

## 最佳结果

本轮最佳是 Ensemble 3：

| 评估 | Default acc | Tuned acc | ROC-AUC |
|---|---:|---:|---:|
| 100k random | 87.84% | 88.08% | 90.27% |
| 100k balanced | 83.80% | 84.14% | 90.20% |

对比上一轮最好单模型：

- random tuned accuracy：`87.68%` -> `88.08%`
- balanced tuned accuracy：`83.50%` -> `84.14%`
- balanced ROC-AUC：`89.60%` -> `90.20%`

## 是否达到 90%

没有达到。

Ensemble 使 ROC-AUC 超过 `90%`，但 accuracy 仍然停在：

- random：`88.08%`
- balanced：`84.14%`

## 主要原因

1. Sequence smoothing 几乎没有提升，说明错误不是简单的逐窗抖动。
2. 阈值调优只有小幅收益，说明主要瓶颈不是 calibration。
3. Ensemble 有稳定收益，说明不同模型有互补性，但互补不足以弥补 score separability 的缺口。
4. 当前最强 CNN/SNN 都在 85-88% accuracy 区间，说明需要回到训练目标、输入表示或标签定义上改进。

## 下一轮建议

下一轮应进入训练目标和输入表示探索，优先级：

1. 训练 `sampling=random` 或 mixed sampling 的 SNN，而不是只用 balanced sampling，再看 natural accuracy 是否接近 90%。
2. 引入 margin/focal loss 或 class-prior-aware loss，提升 score ranking 与 hard example 区分。
3. 针对 transition 以外的低分 sequence 做分组诊断，找出是否存在特定材料/序列拖低指标。
4. 如果继续 ensemble，应把它作为最终部署补充，而不是主要研究路线。

## 本轮产物

- `01_sequence_smoothing.md`
- `02_score_ensemble.md`
- `03_threshold_prior.md`
- `remote_summaries/`
- 新脚本：
  - `scripts/train/evaluate_checkpoint_ensemble.py`
  - `scripts/train/run_iter01_postprocess_ensemble.sh`
  - `scripts/train/launch_iter01_postprocess_ensemble.sh`
