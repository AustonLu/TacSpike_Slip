# 01 原始 Window Size Sliding 评估

日期：2026-07-01

## 配置

- 远程 run id：`original20_lite_longtrain_val32`
- checkpoint：`/lamport/makkapakka/jiajunlu/logs/tacspike_stage2_explore/lite_longtrain_v1/best.pt`
- 数据：validation split，随机选取 32 条完整 sequence
- 上下文：checkpoint 原始 window size
- 后处理搜索：
  - 原始 score
  - causal moving average：`3,5,10,20,50`
  - causal EMA：`0.1,0.2,0.4`
  - debounce：on `2,3,5`，off `2,3,5,10`
- 结果文件：
  - `remote_summaries/original20_lite_longtrain_val32_sliding_detection.json`
  - `remote_summaries/original20_lite_longtrain_val32_run.out`

## 最佳结果

最佳方法为 `ma_50_debounce_on2_off10`。

| 指标 | 数值 |
|---|---:|
| selected sequences | 32 |
| total windows | 671348 |
| positive fraction | 0.3111 |
| accuracy | 0.7960 |
| balanced accuracy | 0.7066 |
| F1 | 0.5891 |
| ROC-AUC | 0.7066 |
| PR-AUC | 0.5843 |
| true slip segments | 45 |
| detected slip segments | 41 |
| missed slip segments | 4 |
| segment recall | 0.9111 |
| median delay | 0 ms |
| p95 delay | 2768 ms |
| false alarm runs | 478 |
| false alarm runs/min | 42.72 |
| false positive window fraction | 0.0569 |
| prediction switches | 2185 |
| label transitions | 60 |

## 判断

原始 window size 在连续序列上不够稳定。它能检测到大部分 slip 段，segment recall 达到 91.1%，但是误报 run 很多，预测状态抖动严重：`prediction_switches=2185`，而真实 `label_transitions=60`。这说明该模型在独立窗口分类口径下有一定可分性，但在真实 1 kHz 连续检测中，短上下文不足以提供稳定状态。

本项未达到 90% accuracy，也不适合作为后续主路线。后续应优先考察更长上下文或真正 sequence/streaming 训练。
