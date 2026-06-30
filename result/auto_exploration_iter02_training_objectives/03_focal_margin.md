# 03. Focal Loss / Margin Loss Quick

状态：完成

## 目的

快速检查 focal loss 和 margin regularization 是否能提高 hard examples 的 score margin。为了节省资源，本项使用 balanced sampling quick run。

## 新增实现

在 `scripts/train/train_lite_scnn.py` 中加入：

- `--loss-type {ce,focal}`
- `--focal-gamma`
- `--margin-loss-weight`
- `--margin-value`

训练时可选 focal CE，并可在 CE/focal/distillation loss 后附加 margin regularization。

## Quick 配置

- 模型：`ctx500_tb100_wide2_scnn`
- sampling：`balanced`
- epoch：`4`
- 每 epoch 训练样本：`12000`
- validation：`6000` balanced samples

## 结果

| Run | Loss | Best epoch | Accuracy | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| `iter02_balanced_wide2_focal_quick_v1` | focal gamma=2 | 4 | 79.48% | 86.11% | 86.89% |
| `iter02_balanced_wide2_margin_quick_v1` | CE + margin 0.1 | 4 | 79.55% | 86.55% | 87.57% |
| `iter02_balanced_wide2_focal_distill_quick_v1` | focal + distill | 4 | 79.02% | 86.09% | 87.85% |

## 观察

三组 quick run 都没有超过此前 wide2 quick 的有效区间，也明显低于完整训练模型。Margin quick 的 ROC-AUC 略高于 focal quick，但幅度不足，不值得直接扩大为 full training。

## 结论

Focal loss 和简单 margin regularization 不是当前最优先方向。Iter02 后续只对 random CE 和 random distillation 做 100k 复评。
