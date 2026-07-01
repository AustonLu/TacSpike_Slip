# 02 500ms 上下文 Sliding 评估

日期：2026-07-01

## 配置

- 远程 run id：`ctx500_best5_val16`
- 数据：validation split，随机选取 16 条完整 sequence
- total windows：369291
- 模型：Iter10 中表现最好的 5 个 500ms LIF-SNN checkpoint 进行 score ensemble
- ensemble：
  - `iter06_time_channel_random_ignore100_v1`
  - `iter06_time_channel_random_ignore50_v1`
  - `iter07_time_channel_w48_h384_ignore50_v1`
  - `iter09_tw_near20_mid100_v1`
  - `iter09_tw_near50_mid100_smooth02_v1`
- score transform：per-model `zscore`
- ensemble weights：mean
- 后处理搜索：
  - 原始 ensemble score
  - causal moving average：`3,5,10,20,50`
  - causal EMA：`0.1,0.2,0.4`
  - debounce：on `2,3,5`，off `2,3,5,10`
- 结果文件：
  - `remote_summaries/ctx500_best5_val16_sliding_detection.json`
  - `remote_summaries/ctx500_best5_val16_run.out`

## 最佳结果

最佳方法为 `ma_50_debounce_on5_off10`。

| 指标 | 数值 |
|---|---:|
| selected sequences | 16 |
| total windows | 369291 |
| positive fraction | 0.2684 |
| accuracy | 0.8978 |
| balanced accuracy | 0.8719 |
| F1 | 0.8108 |
| ROC-AUC | 0.8719 |
| PR-AUC | 0.8245 |
| true slip segments | 20 |
| detected slip segments | 18 |
| missed slip segments | 2 |
| segment recall | 0.9000 |
| mean delay | 64.44 ms |
| median delay | 0 ms |
| p95 delay | 302.85 ms |
| max delay | 744 ms |
| false alarm runs | 2 |
| false alarm runs/min | 0.325 |
| false positive window fraction | 0.0722 |
| prediction switches | 32 |
| label transitions | 29 |

## 与原始 Window 的差异

500ms 上下文显著改善了连续状态稳定性：

| 项目 | 原始 window | 500ms best5 |
|---|---:|---:|
| accuracy | 0.7960 | 0.8978 |
| balanced accuracy | 0.7066 | 0.8719 |
| F1 | 0.5891 | 0.8108 |
| segment recall | 0.9111 | 0.9000 |
| false alarm runs/min | 42.72 | 0.325 |
| prediction switches | 2185 | 32 |

500ms best5 的主要收益不是把 segment recall 大幅提高，而是把连续序列中的误报 run 和状态抖动压低到可接受范围。它距离 90% accuracy 只差约 0.22 个百分点。

## 失败样本

最差 sequence 主要表现为两类：

| sequence_id | windows | accuracy | true segments | missed | false alarms | FP fraction |
|---|---:|---:|---:|---:|---:|---:|
| `flat_batch_4_52` | 18481 | 0.4902 | 1 | 1 | 0 | 0.0000 |
| `sharp_batch_2_74` | 7681 | 0.6918 | 1 | 1 | 0 | 0.0000 |
| `sharp_batch_2_108` | 23981 | 0.7888 | 2 | 0 | 0 | 0.1556 |
| `sphere_batch_1_127` | 27347 | 0.7943 | 2 | 0 | 0 | 0.2214 |
| `sphere_batch_2_134` | 24747 | 0.8051 | 3 | 0 | 0 | 0.2075 |

前两条是漏检导致 accuracy 很低；后三条主要是 slip 已检测到，但 no-slip 段仍有较长 false positive window。后续如果要稳定超过 90%，不能只调阈值，还需要处理 sequence 内的类别转移和 onset/offset 标签噪声。

## 阈值网格复核

额外运行 `ctx500_best5_val16_threshgrid`，只保留最有效的 `ma_50` 后处理，并对 debounce threshold 做 302 个候选搜索，同时输出 score cache。

结果仍为 `ma_50_debounce_on5_off10`，accuracy `0.897801`，阈值仍选择原始 base threshold `0.630487`。这说明当前 0.22 个百分点差距不是简单靠更细 threshold/debounce 搜索能补上。

缓存文件：

- `remote_summaries/ctx500_best5_val16_threshgrid_score_cache.npz`

该 cache 仅 5.9 MB，包含本次 16 条 sequence、5 个 checkpoint 的 raw score matrix、labels、global indices 和 sequence offsets。后续同一批 sequence 的后处理搜索可以直接读 cache，不必重新动态构造 500ms window。
