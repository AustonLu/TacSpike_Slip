# Auto Exploration Iter03 总结

日期：2026-06-30

## 本轮目标

探索输入表示和 SNN 结构是否是当前精度瓶颈，重点比较：

1. time-channel LIF-SCNN
2. temporal-conv LIF-SCNN
3. wide3 sequential LIF-SCNN

## 最佳结果

本轮最佳是 `iter03_time_channel_scnn_thr1_full_v1`：

| 评估 | Default acc | Tuned acc | ROC-AUC |
|---|---:|---:|---:|
| 100k random | 88.52% | 88.98% | 90.24% |
| 100k balanced | 85.38% | 85.45% | 90.35% |

20k balanced validation 最好为 `86.13%`，ROC-AUC `91.04%`。

## 是否达到 90%

没有达到。

但本轮刷新了当前 SNN 最好结果：

- Random tuned：`88.08%` -> `88.98%`
- Balanced tuned：`84.14%` -> `85.45%`

Random tuned 距离 `90%` 还差 `1.02` 个百分点。

## 主要结论

1. 输入表示是关键瓶颈。Time-channel LIF-SCNN 明显优于 sequential wide2/wide3。
2. LIF 阈值尺度很关键。BatchNorm + LIF hidden 下 `threshold=0.1` 容易 firing rate 过高或验证塌缩，`threshold=1.0` 明显更稳定。
3. 3D temporal-conv 有收益，但不如 time-channel 表示，且参数更多。
4. 单纯扩大 sequential SCNN 到 634k 参数没有超过 time-channel SNN，说明容量不是唯一瓶颈。
5. 目前最有希望的主模型应切换为 `time_channel_scnn + threshold=1.0 + distillation`。

## 下一轮建议

Iter04 应继续围绕 time-channel SNN 精修，目标是补上最后约 `1%`：

1. 训练目标：比较 balanced、random、mixed sampling 对 time-channel SNN 的 100k random accuracy 影响。
2. 蒸馏设置：扫描 `distill_alpha=0.1/0.3/0.5/0.7`，检查是否能提升 ROC-AUC 或 natural accuracy。
3. 正则化：测试 dropout、weight decay、label smoothing，避免 time-channel 模型过拟合 balanced validation。
4. Ensemble：用 time-channel SNN 与 wide3 / previous wide2 做 SNN-only score ensemble，检查是否能把 random tuned 推过 `90%`。

## 本轮产物

- `01_time_channel_scnn.md`
- `02_temporal_conv_scnn.md`
- `03_wide3_sequential_scnn.md`
- `04_full_eval.md`
- `remote_summaries/`
- 新增/修改脚本：
  - `src/tacspike/models/hybrid_scnn.py`
  - `src/tacspike/models/__init__.py`
  - `scripts/train/train_lite_scnn.py`
  - `scripts/train/run_iter03_structure_input.sh`
  - `scripts/train/launch_iter03_structure_input.sh`
  - `scripts/train/evaluate_iter03_structure_input.sh`
