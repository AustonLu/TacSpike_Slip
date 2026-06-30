# Auto Exploration Iter02 总结

日期：2026-06-30

## 本轮目标

探索训练目标和采样策略是否可以把 SNN accuracy 推到 `90%`，重点包括 random sampling、random + distillation、focal loss 和 margin regularization。

## 最佳结果

本轮最佳是 `iter02_random_wide2_v1`：

| 评估 | Default acc | Tuned acc | ROC-AUC |
|---|---:|---:|---:|
| 100k random | 87.19% | 87.46% | 88.88% |
| 100k balanced | 80.64% | 83.28% | 88.89% |

## 是否达到 90%

没有达到。

本轮没有超过 Iter01 ensemble：

- Iter01 ensemble 100k random tuned：`88.08%`
- Iter02 best 100k random tuned：`87.46%`
- 差距：`-0.62` 个百分点

## 主要结论

1. Random sampling 能提高自然分布默认阈值 accuracy，但会降低 slip recall，balanced accuracy 不占优。
2. Random + distillation 能提高 recall 和 balanced accuracy，但没有提高 ROC-AUC，因此 tuned accuracy 不如纯 random CE。
3. Focal loss 和 margin regularization quick run 没有显示出继续扩大的价值。
4. 当前瓶颈不像是简单的 class prior 或 loss 形式问题，更像是 SNN 对 500 ms / 100 bins 输入的时间信息保留能力不足。

## 下一轮建议

下一轮优先探索输入表示和结构，而不是继续做 focal/margin 小调参：

1. 尝试更接近 CNN teacher 的 time-channel 输入前端，但仍保留 LIF 隐层，检查 SNN 是否主要损失在逐时刻卷积读入阶段。
2. 尝试更宽的 `wide3` 或带 BatchNorm 的 hybrid SCNN，看容量和归一化是否能缩小 CNN/SNN 差距。
3. 继续使用 100k random/balanced 作为最终评估口径；quick run 只作为筛选。
4. 如果新结构仍停在 `88%` 左右，应转向数据/标签诊断或更强 teacher，而不是继续扩大轻量 SNN。

## 本轮产物

- `01_random_sampling.md`
- `02_random_distill.md`
- `03_focal_margin.md`
- `04_eval_best.md`
- `remote_summaries/`
- 新增/修改脚本：
  - `scripts/train/train_lite_scnn.py`
  - `scripts/train/run_iter02_training_objectives.sh`
  - `scripts/train/launch_iter02_training_objectives.sh`
  - `scripts/train/evaluate_iter02_training_objectives.sh`
