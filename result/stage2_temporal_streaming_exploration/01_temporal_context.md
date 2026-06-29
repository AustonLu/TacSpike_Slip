# 更长 Temporal Context 实验

状态：完成

目的：验证单个 20 ms window 信息量不足的假设，并检查更长上下文对非脉冲 CNN upper-bound 和轻量 LIF-SCNN 的收益。

## 实现

本轮没有重新生成 HDF5 数据集，而是在数据读取层按当前 window 的 `t_label` 向前回看更长时间：

- `context_ms=50/100/200/300/500/1000`
- 输入形状为 `[context_ms, 2, 32, 32]`
- 标签仍使用当前窗口对应的 `label/slip`

训练脚本通过 `--context-ms`、`--time-bins`、`--time-steps` 控制输入长度。所有新增 SNN 实验继续使用 LIF，不使用 IAF。

## 训练验证结果

| Run | 模型 | 输入 | Best epoch | Val acc | PR-AUC | ROC-AUC | 参数量 |
|---|---|---:|---:|---:|---:|---:|---:|
| `ctx50_frame_cnn_v1` | FrameCNN | 50 ms | 7 | 76.38% | 84.75% | 82.09% | 531,650 |
| `ctx100_frame_cnn_v1` | FrameCNN | 100 ms | 7 | 79.57% | 87.86% | 86.08% | 560,450 |
| `ctx200_frame_cnn_v1` | FrameCNN | 200 ms | 6 | 82.80% | 90.26% | 89.37% | 618,050 |
| `ctx300_frame_cnn_v1` | FrameCNN | 300 ms | 6 | 84.27% | 91.32% | 90.86% | 675,650 |
| `ctx500_frame_cnn_v1` | FrameCNN | 500 ms | 6 | **85.64%** | 92.20% | **91.95%** | 790,850 |
| `ctx1000_frame_cnn_v1` | FrameCNN | 1000 ms | 6 | 85.42% | **92.32%** | 91.62% | 1,078,850 |
| `ctx50_lite_scnn_v1` | Lite-SCNN | 50 ms | 8 | 69.92% | 82.69% | 79.67% | 38,304 |
| `ctx100_lite_scnn_v1` | Lite-SCNN | 100 ms | 8 | 74.44% | 85.85% | 83.40% | 38,304 |
| `ctx200_lite_scnn_v1` | Lite-SCNN | 200 ms | 5 | 77.83% | 87.45% | 85.93% | 38,304 |
| `ctx300_lite_scnn_v1` | Lite-SCNN | 300 ms | 6 | **79.36%** | **88.50%** | **87.17%** | 38,304 |

## 100k Natural Validation

评估方式：`split=val`，`sampling=random`，`samples=100000`，阈值按 accuracy 搜索。

| Run | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `ctx500_frame_cnn_v1` | **88.72%** | **84.46%** | **79.05%** | **82.86%** | **91.15%** |
| `ctx300_frame_cnn_v1` | 87.60% | 82.63% | 76.59% | 81.39% | 90.04% |
| `ctx200_frame_cnn_v1` | 86.23% | 80.53% | 73.60% | 79.30% | 88.38% |
| `ctx300_lite_scnn_v1` | 85.44% | 78.69% | 71.17% | 76.87% | 86.36% |
| `ctx200_lite_scnn_v1` | 84.56% | 77.34% | 69.11% | 74.97% | 85.04% |
| `ctx100_frame_cnn_v1` | 84.21% | 76.54% | 67.96% | 75.88% | 85.34% |
| `ctx100_lite_scnn_v1` | 83.62% | 75.73% | 66.66% | 73.12% | 82.77% |
| `ctx50_frame_cnn_v1` | 82.20% | 73.16% | 62.55% | 71.29% | 81.34% |
| `ctx50_lite_scnn_v1` | 81.28% | 72.25% | 60.96% | 68.66% | 79.09% |

## 100k Balanced Validation

评估方式：`split=val`，`sampling=balanced`，`samples=100000`，阈值按 accuracy 搜索。

| Run | Accuracy | Balanced acc | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `ctx500_frame_cnn_v1` | **85.18%** | **85.18%** | **84.14%** | **91.33%** | **91.24%** |
| `ctx300_frame_cnn_v1` | 83.95% | 83.95% | 82.91% | 90.55% | 90.24% |
| `ctx200_frame_cnn_v1` | 82.14% | 82.14% | 81.20% | 89.33% | 88.67% |
| `ctx300_lite_scnn_v1` | 80.09% | 80.09% | 77.81% | 87.69% | 86.57% |
| `ctx100_frame_cnn_v1` | 79.04% | 79.04% | 77.54% | 86.85% | 85.29% |
| `ctx200_lite_scnn_v1` | 78.98% | 78.98% | 76.43% | 86.53% | 85.30% |
| `ctx100_lite_scnn_v1` | 77.36% | 77.36% | 74.31% | 85.06% | 82.90% |
| `ctx50_frame_cnn_v1` | 75.58% | 75.58% | 72.71% | 83.80% | 81.38% |
| `ctx50_lite_scnn_v1` | 73.76% | 73.76% | 69.99% | 81.92% | 79.06% |

## 结论

更长上下文是本轮最有效的提升手段。相对 20 ms 最好结果 natural `80.76%`、balanced `72.17%`，500 ms FrameCNN 提升到 natural `88.72%`、balanced `85.18%`，ROC-AUC 超过 `91%`。

但 500 ms 仍未在 100k window-level accuracy 上达到 `90%`，并且继续加长到 1000 ms 没有继续提升 validation accuracy。这说明当前任务的瓶颈不是单纯“上下文越长越好”，500 ms 附近可能是当前 FrameCNN 配置下较合适的上限探针。

轻量 Lite-SCNN 也吃到了长上下文收益：300 ms SNN 达到 natural `85.44%`、balanced `80.09%`，明显优于 20/100/200 ms SNN。但它仍落后 500 ms FrameCNN 约 5 个 balanced accuracy 点，后续如果继续优化 SNN，应重点研究如何把 FrameCNN 的 500 ms 上下文收益迁移到轻量 LIF 结构中。
