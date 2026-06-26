# Stage 2 精度提升探索记录

日期：2026-06-26

本轮目标是在阶段 2 主模型结果未达到论文中约 `94%` accuracy 的背景下，系统排查瓶颈来自 SNN 结构、训练策略，还是当前 TacSpike 20 ms window 任务本身的可分性。

## 版本与数据

- 探索前版本已提交并推送到 GitHub：`feec99f Add stage 2 SNN training pipeline and results`
- 远程训练机器：`ssh -J fics jiajunlu@192.168.68.198`
- 远程项目目录：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- 数据目录：`/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0`
- 远程日志目录：`/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_explore/`
- 本地原始 JSON 备份：`result/stage2_accuracy_exploration/remote_summaries/`

## 探索原则

1. 先用非 SNN 的 `FrameCNN` 做 20 ms window 可分性 upper-bound 检查。如果普通 CNN 也无法接近 94%，则不能简单归因于 SNN 结构过弱。
2. 再测试更强 LIF-SCNN、Lite-SCNN 长训、输入尺度放大、更高空间分辨率等配置，判断是否是训练量、发放率或空间池化造成的欠拟合。
3. 所有实验继续使用当前 HDF5 数据、sequence split 和窗口级指标；100k 评估同时报告自然分布 random sampling 和 balanced sampling。
4. 100k 表中的 accuracy 使用 validation set 上搜索到的最佳 accuracy 阈值，因此代表该 checkpoint 在当前 window 指标下的较乐观结果。

## 阶段 2 原始基线

来自 `result/stage2_main_training/README.md`：

| Run | 模型 | 100k natural acc | 100k balanced acc | ROC-AUC | 说明 |
|---|---|---:|---:|---:|---|
| `main_lite_scnn_v1_logit_mean_v2` | Lite LIF-SCNN, 3.8 万参数 | 79.45% | 70.58% | 74.45% / 74.38% | 阶段 2 主模型最好结果 |

该结果明显低于目标 `94%`，因此本轮继续做上限和训练策略排查。

## 新增代码

- `src/tacspike/models/frame_cnn.py`：非 SNN 小型 CNN upper-bound；输入仍为 `[B,T,C,H,W]`，支持 `time_channels` 和 `sum` 两种时间处理方式。
- `src/tacspike/models/deep_scnn.py`：更宽的 BN + LIF-SCNN，用于排查 Lite-SCNN 容量不足问题。
- `scripts/train/train_lite_scnn.py`：新增 `--model {lite_scnn,deep_scnn,frame_cnn}`、AMP、cosine scheduler、宽度/隐藏维度和时间模式参数。
- `scripts/train/evaluate_lite_scnn.py`：评估脚本按 checkpoint 中保存的 args 自动重建模型。
- `scripts/train/run_stage2_accuracy_exploration.sh`：远程复现实验启动脚本，支持本轮补充的 run id。

## 训练验证集结果

下表为训练过程中保存的 best validation 结果。采样方式随 run 配置不同：大多数为 balanced validation，`cnn_random_weighted_v1` 为自然分布 random validation。

| Run | 目的 | 模型/关键配置 | Best epoch | Val acc | PR-AUC | ROC-AUC | 参数量 |
|---|---|---|---:|---:|---:|---:|---:|
| `cnn_upper_bound_v1` | 检查 20 ms window 是否可由普通 CNN 学好 | `FrameCNN`, time as channels, balanced/no weight | 10 | 71.91% | 80.49% | 76.84% | 514,370 |
| `cnn_random_weighted_v1` | 检查自然分布训练和更大 CNN 是否提高 natural accuracy | `FrameCNN`, width 48, random + inverse-frequency weight | 12 | 80.67% | 67.28% | 77.30% | 1,147,874 |
| `frame_cnn_sum_v1` | 判断 1 ms time-bin 身份是否关键 | `FrameCNN`, temporal sum, balanced/no weight | 7 | 72.48% | 81.23% | 77.52% | 503,426 |
| `deep_scnn_v1` | 判断 Lite-SCNN 是否因容量/BN 不足 | 3 conv + BN + LIF + logit mean | 5 | 67.74% | 74.65% | 72.43% | 356,866 |
| `lite_longtrain_v1` | 判断主 SNN 是否训练量不足 | Lite-SCNN, 15 epoch, 200k samples/epoch, cosine, AMP | 15 | 70.89% | 79.09% | 74.84% | 38,304 |
| `lite_scaled_v1` | 判断输入事件过稀是否导致 LIF 欠发放 | Lite-SCNN, input scale 4.0 | 8 | 69.92% | 79.30% | 75.34% | 38,304 |
| `lite_pool2_v1` | 判断 32x32 空间池化是否损失过多 | Lite-SCNN, spatial pool 2 | 8 | 69.04% | 79.08% | 75.41% | 38,304 |

## 100k Natural Validation

评估方式：`split=val`，`sampling=random`，`samples=100000`，阈值按 accuracy 搜索。

| Run | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC | 阈值 |
|---|---:|---:|---:|---:|---:|---:|
| `cnn_random_weighted_v1` | **80.76%** | 70.24% | 57.58% | 67.26% | **77.35%** | 0.0928 |
| `frame_cnn_sum_v1` | 80.37% | **70.28%** | **57.63%** | 66.60% | 77.04% | 0.9560 |
| `lite_longtrain_v1` | 80.02% | 69.05% | 55.40% | 64.79% | 74.66% | 0.6682 |
| `cnn_upper_bound_v1` | 79.75% | 69.26% | 55.82% | 65.38% | 76.36% | 0.7293 |
| `lite_scaled_v1` | 79.38% | 67.86% | 53.16% | 64.31% | 74.96% | 0.8006 |
| `lite_pool2_v1` | 79.20% | 68.56% | 54.57% | 63.79% | 75.16% | 0.7402 |
| `deep_scnn_v1` | 76.03% | 62.65% | 42.84% | 56.09% | 72.01% | 0.1295 |

本轮最高 natural accuracy 为 `cnn_random_weighted_v1` 的 `80.76%`。Lite-SCNN 长训后达到 `80.02%`，比阶段 2 原始 `79.45%` 略高，但没有本质突破。

## 100k Balanced Validation

评估方式：`split=val`，`sampling=balanced`，`samples=100000`，阈值按 accuracy 搜索。部分早期 run 未补 balanced 100k，因此只列已完成项。

| Run | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC | 阈值 |
|---|---:|---:|---:|---:|---:|---:|
| `cnn_random_weighted_v1` | **72.17%** | **72.17%** | **67.92%** | **80.90%** | **77.35%** | -0.6035 |
| `frame_cnn_sum_v1` | 72.17% | 72.17% | 67.59% | 80.55% | 77.02% | 0.2376 |
| `lite_longtrain_v1` | 71.28% | 71.28% | 67.03% | 79.08% | 74.76% | 0.1560 |
| `lite_scaled_v1` | 70.65% | 70.65% | 67.30% | 78.83% | 74.95% | 0.2083 |
| `lite_pool2_v1` | 70.50% | 70.50% | 67.23% | 78.66% | 75.17% | 0.2425 |

本轮最高 balanced accuracy 为 `72.17%`，由 `cnn_random_weighted_v1` 和 `frame_cnn_sum_v1` 达到；仍远低于 `94%`。

## 关键判断

1. 当前 20 ms window 任务没有显示出接近 94% 的可分性。普通 CNN 的参数量已经达到 50 万到 115 万，100k natural accuracy 仍只到 `80.76%`，balanced accuracy 只到 `72.17%`。
2. Lite-SCNN 的训练量不是主要瓶颈。`lite_longtrain_v1` 将训练量提高到 15 epoch、200k samples/epoch 后，balanced 100k 只从原始 `70.58%` 提升到 `71.28%`。
3. LIF 发放率或输入尺度不是主要瓶颈。`lite_scaled_v1` 放大输入到 4 倍后没有超过原始主模型。
4. 当前 `spatial_pool=4` 不是主要瓶颈。`lite_pool2_v1` 保留更高空间分辨率后没有改善。
5. 更深更宽的 LIF-SCNN 没有自动改善，`deep_scnn_v1` 反而明显变差，说明简单加容量和 BN 不是有效路径。
6. `FrameCNN` 的 `time_channels` 和 `sum` 模式结果接近，说明在当前 20 ms 窗口内，1 ms time-bin 身份没有提供足以突破瓶颈的判别信息。

综合判断：当前精度瓶颈更可能来自窗口级任务定义和数据可分性，而不是现有 Lite-SCNN 训练脚本或轻量模型容量。阶段 1 已显示 TacSpike window 较稀疏，很多 20 ms 窗口事件极少；再叠加 sliding window 标签边界，单个窗口内的 slip/no-slip 可能存在天然模糊。

## 与论文 94% 的关系

文献中的约 `94%` 结果来自不同传感器、不同输入分辨率、不同任务定义和数据采样设置，不能直接作为 TacSpike 当前 20 ms window 二分类任务的必达指标。对本数据集，目前更合理的阶段性基线应是：

- window-level natural accuracy：约 `80-81%`
- window-level balanced accuracy：约 `71-72%`
- ROC-AUC：约 `0.75-0.77`

如果继续只优化当前 20 ms 独立窗口分类，预计很难通过模型/训练技巧直接达到 `94%`。

## 后续建议

1. 优先扩展 temporal context：生成或读取 `50 ms`、`100 ms` 或多窗口上下文输入，再训练同样的 Lite-SCNN/FrameCNN upper-bound。当前单个 20 ms 窗口信息量很可能不足。
2. 引入 streaming/sequence-level 评估：模型仍可先按 sliding window 训练，但在线推理时用状态延续、概率平滑或连续 K 个窗口投票，指标应加入 slip onset latency、segment-level recall 和 false alarm rate。
3. 检查标签边界：对 slip onset 前后若干 ms 的不确定窗口单独统计，必要时做边界窗口剔除、软标签或分段评估。
4. 保留 `lite_longtrain_v1` 作为当前 SNN 主模型改进基线：它仍只有 38k 参数，balanced 100k 为 `71.28%`，适合后续 tiny/streaming 化。
5. 继续使用非 SNN CNN 作为数据可分性上限探针。只有当 CNN upper-bound 在更长上下文或新标签定义下明显上升，再投入更多 SNN 结构搜索才更有意义。

## 复现实验

远程项目同步本地探索代码后，可使用：

```bash
cd /lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2
CUDA_VISIBLE_DEVICES=0 bash scripts/train/run_stage2_accuracy_exploration.sh lite_longtrain_v1
CUDA_VISIBLE_DEVICES=1 bash scripts/train/run_stage2_accuracy_exploration.sh lite_scaled_v1
CUDA_VISIBLE_DEVICES=2 bash scripts/train/run_stage2_accuracy_exploration.sh lite_pool2_v1
CUDA_VISIBLE_DEVICES=3 bash scripts/train/run_stage2_accuracy_exploration.sh frame_cnn_sum_v1
```

本轮其他 run 的训练命令和完整结果保存在远程日志目录及本地 JSON 备份中。

## 已知限制

- `cnn_random_weighted_v1` 原计划 20 epoch，但远程进程在 epoch 15 左右结束；best checkpoint 出现在 epoch 12。其结果已经是本轮最好，但仍远低于目标。
- 远程 `/lamport` 曾出现个别坏挂载目录，读取日志时偶发超时；本轮最终 JSON 已全部拉回本地。
- 本轮没有重新生成更长时间窗口的数据，因此还没有直接验证 `50/100 ms` temporal context 的收益。
