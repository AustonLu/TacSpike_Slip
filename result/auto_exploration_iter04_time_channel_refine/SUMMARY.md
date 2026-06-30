# Auto Exploration Iter04 总结

日期：2026-06-30

## 本轮目标

围绕 Iter03 最强 time-channel LIF-SCNN 做精修，尝试把 100k random tuned accuracy 从 `88.98%` 推到 `90%`。

## 最佳结果

本轮最佳是 `ensemble_timechannel3`：

| 指标 | 数值 |
|---|---:|
| 100k random tuned accuracy | 89.50% |
| 100k random default accuracy | 89.36% |
| Balanced acc at tuned threshold | 85.38% |
| ROC-AUC | 91.24% |
| PR-AUC | 84.85% |

最佳单模型是 `iter04_time_channel_thr1_random_v1`：

- 100k random tuned accuracy：`89.45%`
- 100k balanced tuned accuracy：`86.00%`
- ROC-AUC：约 `91.06%`

## 是否达到 90%

没有达到。

但本轮刷新了当前 SNN 最好结果：

- Iter03 best random tuned：`88.98%`
- Iter04 best random tuned：`89.50%`
- 提升：`+0.52` 个百分点
- 距离目标：`0.50` 个百分点

## 主要结论

1. Random sampling 对 natural accuracy 有稳定收益，是本轮最有效单模型改动。
2. Random + distillation 没有超过 random-only，说明 teacher 对自然分布训练的帮助有限。
3. Distillation alpha、dropout、label smoothing 在 balanced sampling 下基本持平，不能补上最后差距。
4. SNN-only ensemble 有小幅收益，但加入结构差异更大的 wide3/wide2 后 accuracy 反而下降。
5. 当前模型分数可分性已经较强，ROC-AUC 超过 `91%`，但 accuracy 卡在 `89.5%`，下一轮应重点分析剩余错误的位置和标签定义。

## 下一轮建议

Iter05 应转向错误诊断和数据/标签层面：

1. 对 `iter04_time_channel_thr1_random_v1` 和 `ensemble_timechannel3` 做 sequence-level error analysis，找出拖低指标的 sequence。
2. 分析错误是否集中在 slip onset / offset 附近，尝试 ignore-transition 或 label delay tolerance 的评估与训练。
3. 尝试 sequence-level smoothing / hysteresis 在 time-channel SNN 上是否能越过 `90%`，但需要用全 validation sequence 而不是少量 easy subset。
4. 若 window-level 独立标签仍卡住，考虑把目标改为 onset-tolerant 或短时连续检测指标，而不是单窗口硬标签 accuracy。

## 本轮产物

- `01_sampling_prior.md`
- `02_distill_regularization.md`
- `03_snn_ensemble.md`
- `04_best_eval.md`
- `remote_summaries/`
- 新增脚本：
  - `scripts/train/run_iter04_time_channel_refine.sh`
  - `scripts/train/launch_iter04_time_channel_refine.sh`
  - `scripts/train/evaluate_iter04_time_channel_refine.sh`
