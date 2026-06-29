# Stage 2 SNN 90% 精度探索计划

日期：2026-06-29

分支：`explore-temporal-streaming`

目标：在当前最好轻量 LIF-SNN `ctx300_lite_scnn_v1` 的 100k natural `85.44%`、balanced `80.09%` 基础上，探索是否能把 SNN 推到 `90%` 以上，或至少缩小与 `ctx500_frame_cnn_v1` 的差距。

## 背景

上一轮结论：

- 最好 CNN upper-bound：`ctx500_frame_cnn_v1`
  - natural 100k accuracy：`88.72%`
  - balanced 100k accuracy：`85.18%`
  - balanced ROC-AUC：`91.24%`
- 最好 LIF-SNN：`ctx300_lite_scnn_v1`
  - natural 100k accuracy：`85.44%`
  - balanced accuracy：`80.09%`
  - balanced ROC-AUC：`86.57%`

本轮集中处理 SNN 与 CNN 之间的差距，继续使用 LIF，不使用 IAF。

## 实际探索项

### A. 500 ms LIF-SNN

目的：确认 SNN 是否能直接吃到 `500 ms` 上下文收益。

执行中发现 `500 ms / 500 step` SNN 虽然语义最接近 1 kHz 输入，但训练吞吐只有约 `115-158 samples/s`，且 quick 指标没有提高。因此追加 `500 ms / 100 bins`：仍保留 500 ms 历史，但把 BPTT 长度降到 100。

记录文件：`01_ctx500_snn.md`

### B. 加宽 SCNN

目的：检查 3.8 万参数 Lite-SCNN 是否容量不足。

执行配置：

- `wide`：conv `24/48`，hidden `128`。
- `wide2`：conv `32/64`，hidden `256`。
- readout 使用 `logit_mean`，部分实验只读后半段时间步。

记录文件：`02_wide_scnn.md`

### C. CNN Teacher Distillation

目的：让 SNN 学习 CNN teacher 的 soft score，而不是只学习硬标签。

由于 `500 ms / 100 bins` 的输入时间通道不同，本轮训练了匹配的 teacher：

```text
ctx500_tb100_frame_cnn_teacher_v1
```

记录文件：`03_distillation.md`

### D. Onset 标签噪声处理

目的：减少 label transition 附近模糊硬标签对训练的负面影响。

策略：训练采样时忽略距离 label transition `50 ms` 内的窗口；validation 不忽略任何样本。

记录文件：`04_onset_noise.md`

### E. DeepSCNN 结构对照

目的：检查更深的 LIF-SCNN、BatchNorm 和第三个卷积 LIF block 是否解决容量瓶颈。

记录文件：`05_deep_scnn.md`

## 成功标准

优先看：

- 100k natural validation accuracy >= `90%`
- 或 100k balanced validation accuracy >= `90%`

如果达不到，记录是否至少满足：

- natural accuracy 接近/超过 `ctx500_frame_cnn_v1` 的 `88.72%`
- balanced accuracy 明显超过当前 SNN 的 `80.09%`
- ROC-AUC 明显接近 CNN 的 `91.24%`

## 执行结论

本轮没有达到 `90%` accuracy。当前最好 SNN：

- `ctx500_tb100_wide2_scnn_distill_v1`
  - 100k random accuracy：default `86.99%`，tuned `87.58%`
  - 100k balanced accuracy：default `83.46%`，tuned `83.50%`
  - 100k balanced ROC-AUC：`89.60%`
- `ctx500_tb100_wide2_scnn_ignore50_v1`
  - 100k random accuracy：default `87.48%`，tuned `87.68%`
  - 100k balanced accuracy：`83.22%`
  - 100k balanced ROC-AUC：`89.46%`

相比上一轮 `ctx300_lite_scnn_v1` 的 balanced `80.09%`，本轮最好 100k balanced accuracy 提升到 `83.50%`，ROC-AUC 从 `86.57%` 提升到 `89.60%`。但它仍低于 `ctx500_frame_cnn_v1` 的 balanced `85.18%`，也未达到 `90%`。
