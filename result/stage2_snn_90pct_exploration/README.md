# Stage 2 SNN 90% 精度探索记录

日期：2026-06-29

分支：`explore-temporal-streaming`

目标：在当前分支继续探索 SNN 精度，重点验证 500 ms 上下文、更大 SCNN、CNN teacher distillation、onset 标签噪声处理和 DeepSCNN 结构对照，尝试把 validation accuracy 推到 `90%` 以上。

## 当前结论

本轮没有达到 `90%` accuracy。当前最强 SNN 是：

| 模型 | 评估 | Accuracy | Balanced acc | ROC-AUC | 参数量 |
|---|---|---:|---:|---:|---:|
| `ctx500_tb100_wide2_scnn_distill_v1` | 100k random | 86.99% default / 87.58% tuned | 83.57% default | 89.65% | 282,688 |
| `ctx500_tb100_wide2_scnn_distill_v1` | 100k balanced | 83.46% default / 83.50% tuned | 83.46% default | 89.60% | 282,688 |
| `ctx500_tb100_wide2_scnn_ignore50_v1` | 100k random | 87.48% default / 87.68% tuned | 83.19% default | 89.57% | 282,688 |
| `ctx500_tb100_wide2_scnn_ignore50_v1` | 100k balanced | 83.22% default / 83.22% tuned | 83.22% default | 89.46% | 282,688 |

对照 CNN teacher：

| 模型 | 20k balanced val | ROC-AUC | 参数量 |
|---|---:|---:|---:|
| `ctx500_tb100_frame_cnn_teacher_v1` | 85.32% | 91.31% | 560,450 |

本轮最有效的方向是：

1. 保留 `500 ms` 上下文，但把 SNN 时间步从 `500` 降到 `100`，训练吞吐从约 `115-158 samples/s` 提升到约 `700-1350 samples/s`。
2. 把 Lite-SCNN 加宽到 `32/64 conv channels + 256 hidden`，完整训练后 20k balanced val 提升到 `83-84%`。
3. CNN distillation 在完整训练中有小幅收益：`wide2 distill` 20k balanced val `84.07%`，高于 `wide2 ignore50` 的 `83.61%`。
4. 直接加深为 DeepSCNN 没有稳定超过 Wide2-SCNN：best 20k balanced val `83.74%`，但训练震荡更大、速度更慢。

## 结果文件

- `PLAN.md`：本轮探索计划和实际执行说明。
- `01_ctx500_snn.md`：500 ms / 500-step 和 500 ms / 100-bin SNN 上下文实验。
- `02_wide_scnn.md`：SCNN 加宽实验。
- `03_distillation.md`：CNN teacher distillation 实验。
- `04_onset_noise.md`：transition/onset 标签噪声处理实验。
- `05_deep_scnn.md`：DeepSCNN 结构加深对照。
- `remote_summaries/`：远程训练和 100k 评估 JSON 备份。

## 判断

当前证据更支持“数据/标签与模型形式共同限制”而不是单纯“网络太小”。SNN 加宽、蒸馏、onset 过滤都能提升，但 100k balanced accuracy 仍停在 `83.5%` 左右；同时 100-bin FrameCNN teacher 自身也只有 `85.32%` balanced val，说明即使非脉冲模型在这个输入设定下也没有接近 `90%` accuracy。
