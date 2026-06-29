# 500 ms LIF-SCNN 实验

状态：完成

目的：确认 SNN 是否能直接利用 `500 ms` 上下文，并比较 `500 step` 与 `100 bins` 两种时间离散方式。

## 配置

- 输入历史：`context_ms=500`
- 模型：Lite-SCNN / 加宽 SCNN，均为 LIF
- readout：`logit_mean`
- 训练采样：balanced
- quick run：4 epoch，10k train/epoch，5k val

## 500 ms / 500 step quick 结果

| Run | 参数量 | Val acc | ROC-AUC | 吞吐 |
|---|---:|---:|---:|---:|
| `ctx500_lite_scnn_quick_v1` | 38,304 | 76.72% | 84.38% | 158.0 samples/s |
| `ctx500_lite_scnn_tail_quick_v1` | 38,304 | 73.34% | 82.28% | 153.7 samples/s |
| `ctx500_wide_scnn_tail_quick_v1` | 110,128 | 75.36% | 84.17% | 128.9 samples/s |
| `ctx500_wide_scnn_distill_quick_v1` | 110,128 | 76.88% | 84.97% | 115.3 samples/s |

结论：直接用 `500 step` BPTT 没有带来收益，且训练明显变慢。后半段 readout 在这个设置下降低了指标。

## 500 ms / 100 bins quick 结果

| Run | 参数量 | Val acc | ROC-AUC | 吞吐 |
|---|---:|---:|---:|---:|
| `ctx500_tb100_lite_scnn_quick_v1` | 38,304 | 76.30% | 86.19% | 1351.4 samples/s |
| `ctx500_tb100_wide_scnn_quick_v1` | 110,128 | 76.94% | 86.02% | 984.3 samples/s |
| `ctx500_tb100_wide2_scnn_quick_v1` | 282,688 | 79.14% | 86.42% | 727.0 samples/s |

`100 bins` 保留了 500 ms 上下文，但把 BPTT 长度从 500 降到 100，吞吐提升约 5-10 倍，并且 ROC-AUC 略有提升。因此后续主线改用 `500 ms / 100 bins`。

## 结论

长上下文本身仍然重要，但对 SNN 来说不应机械使用 1 ms 一个 BPTT step。当前更合理的工程折中是：输入窗口仍覆盖 500 ms，模型用 100 个较粗时间 bin 处理历史事件。
