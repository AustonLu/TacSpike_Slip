# Stage 2 主模型训练记录

日期：2026-06-25

本阶段按实施计划完成了阶段 2 的前两项工作：训练 `TacSpike-Lite-SCNN-v1` 主模型，并记录训练配置、指标、checkpoint 路径和独立评估结果。消融实验和 paper-style baseline 暂未开展。

## 训练环境

- 远程机器：`ssh -J fics jiajunlu@192.168.68.198`
- 远程项目目录：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- 数据目录：`/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0`
- Python：`/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python`
- GPU：A100，训练时主要使用 `CUDA_VISIBLE_DEVICES=1/2`

## 主模型配置

主模型为轻量 LIF-SCNN，输入为 20 ms strict sliding window：

```text
[B, 20, 2, 32, 32]
Conv(2,16,5) + LIF
Conv(16,32,3,stride=2) + LIF
AdaptiveAvgPool(4x4)
Linear(512,64) + LIF
Linear(64,2)
```

公共训练设置：

- `epochs=10`
- `train_samples_per_epoch=50000`
- `val_samples=20000`
- `batch_size=512`
- `num_workers=8`
- `threshold=0.1`
- `beta=0.85`
- `lr=1e-3`
- `weight_decay=1e-4`
- 参数量：`38304`

代码和指标记录：

- 训练脚本：`scripts/train/train_lite_scnn.py`
- 评估脚本：`scripts/train/evaluate_lite_scnn.py`
- 远程日志根目录：`/lamport/makkapakka/jiajunlu/logs/tacspike_stage2/`
- 本地原始 JSON 备份：`result/stage2_main_training/remote_summaries/`

## 已完成 Run

| Run | 读出 | 采样/权重 | 训练验证口径 | Best epoch | Best val acc | F1 | PR-AUC | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `main_lite_scnn_v1` | output spike count | balanced + inverse-frequency weight | balanced 20k | 6 | 70.01% | 68.41% | 76.68% | 73.73% |
| `main_lite_scnn_v1_membrane` | output LIF membrane | balanced + inverse-frequency weight | balanced 20k | 10 | 65.19% | 69.70% | 79.21% | 76.14% |
| `main_lite_scnn_v1_membrane_noweight` | output LIF membrane | balanced + no weight | balanced 20k | 3 | 68.76% | 69.79% | 78.57% | 75.16% |
| `main_lite_scnn_v1_random_weighted` | output LIF membrane | random + inverse-frequency weight | natural 20k | 1 | 71.24% | 55.82% | 60.76% | 72.92% |
| `main_lite_scnn_v1_logit_mean_v2` | raw logits mean | balanced + no weight | balanced 20k | 10 | 69.42% | 68.99% | 78.86% | 74.61% |

## 100k 独立验证

对较有代表性的 checkpoint 做了 100k validation window 独立评估，并同时记录默认阈值 `0.0` 与按 accuracy 搜索到的最佳阈值。

| Run | 评估采样 | 阈值 | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| `main_lite_scnn_v1` | natural/random 100k | 1.0000 | 78.23% | 68.36% | 54.33% | 59.95% | 72.97% |
| `main_lite_scnn_v1_membrane_noweight` | balanced 100k | 0.1050 | 70.12% | 70.12% | 66.01% | 78.14% | 74.77% |
| `main_lite_scnn_v1_membrane_noweight` | natural/random 100k | 0.3486 | 78.77% | 67.45% | 52.49% | 63.03% | 74.88% |
| `main_lite_scnn_v1_logit_mean_v2` | balanced 100k | 0.1727 | 70.58% | 70.58% | 67.16% | 78.52% | 74.38% |
| `main_lite_scnn_v1_logit_mean_v2` | natural/random 100k | 0.6631 | 79.45% | 68.46% | 54.34% | 63.88% | 74.45% |

当前最高自然分布 accuracy 来自 `main_lite_scnn_v1_logit_mean_v2`，为 `79.45%`。当前最高 balanced validation accuracy 为 `70.58%`。两者都没有达到目标 `94%`。

## 结论

阶段 2 主模型训练和记录已经完成，但当前轻量 LIF-SCNN 配置没有达到论文中约 `94%` 的 accuracy 目标。

需要注意，论文的 `94.33%` 来自不同传感器、不同输入分辨率、不同三分类任务和不同数据采样设置，不能直接视为 TacSpike 当前 20 ms window 二分类任务的可达指标。TacSpike 当前窗口极稀疏，stage 1 统计显示平均每 20 ms window 约 3 个 event，且空窗口占比较高。因此自然分布 accuracy 容易被 no-slip 多数类抬高，balanced accuracy、F1、PR-AUC 和 ROC-AUC 更能反映模型是否真正检测 slip。

当前结果表明：

- 仅靠 `TacSpike-Lite-SCNN-v1` 的 3.8 万参数 LIF-SCNN，在 50k samples/epoch 的快速训练设置下，区分能力大约停在 ROC-AUC `0.74-0.76`。
- `balanced sampler + class weight` 会对 slip 过度补偿，precision/specificity 偏低。
- 关闭 class weight 后 balanced accuracy 稍有改善，但自然分布 accuracy 仍只有约 `78-79%`。
- 将输出层从 spike count / output membrane 改为 raw logits mean 有小幅提升，但不足以接近 94%。

## 后续建议

下一步仍应先在主模型范围内提高可分性，再进入系统性的消融实验：

1. 加大训练样本量和 epoch，例如 `200k-500k samples/epoch`、`30-50 epochs`，确认当前结果不是小样本快速训练上限。
2. 增加 temporal/context 信息，测试 `T=30/50 ms`，因为当前 20 ms window 事件过少。
3. 引入 sequence-level smoothing 和 onset 评估，窗口级 accuracy 可能低估在线滑移检测效果。
4. 评估更强但仍轻量的主模型，例如 `Conv 32/64 + FC128` 或 temporal separable block，再决定 tiny 化。
5. 检查标签定义附近的边界窗口，必要时排除 slip onset 前后不确定区域或加入 label smoothing。
6. 在正式消融前补一个非 SNN upper-bound baseline，用非常小的 ANN/CNN 判断数据本身是否能在 20 ms window 上接近 94%。

## 已知问题

- 远程 `/lamport` 上存在个别坏挂载目录，删除 `main_lite_scnn_v1_logit_mean` 失败并报 `Transport endpoint is not connected`。后续使用了新目录 `main_lite_scnn_v1_logit_mean_v2`，不影响结果。
- 远程 HDF5 抽样统计曾因 I/O 超时未完成；stage 1 已完成过 event count 和 voxel sanity check，本阶段未发现训练脚本层面的明显读取错误。
