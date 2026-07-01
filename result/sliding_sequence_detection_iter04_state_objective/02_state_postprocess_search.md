# 02 连续状态后处理扩展

## 目的

本实验先不改变 SNN score extractor，只在已有 score 序列上搜索更强的连续状态后处理，判断 `95%` strict accuracy 是否可以仅靠 moving average、debounce、gap fill、min duration 等状态机参数达成。

## 输入

- 主参考：Iter03 sequence fine-tune 的 `ctx400` SNN score cache。
- 对照：Iter02 `ctx400` score cache。
- 评估口径：validation selected 16 sequences，逐 1ms window strict accuracy。

## 结果

本地 smoke search 复现出的最佳后处理为：

```text
subset_1_causal_ma100_thr_deb_on5_off20_gap0_minon0_minoff0
```

关键指标：

| 指标 | 数值 |
|---|---:|
| strict accuracy | 89.902% |
| balanced accuracy | 87.766% |
| F1 | 81.549% |
| precision | 80.005% |
| recall | 83.154% |
| specificity | 92.378% |
| segment recall | 95.0% |
| false alarm runs/min | 1.787 |
| onset delay p95 | 231.3 ms |
| missed slip segments | 1 / 20 |

作为参考，远端重新生成的两个单模型 validation cache：

| 模型/score | 最佳方法 | strict accuracy | segment recall | onset delay p95 |
|---|---|---:|---:|---:|
| Iter02 ctx400 | `ma_100` | 89.773% | 90.0% | 284.1 ms |
| Iter03 seq-ft ctx400 | `ma_100_debounce_on5_off20` | 89.888% | 95.0% | 231.2 ms |

## 失败点

远端 `postprocess_ctx400_seqft` 作业由于依赖路径问题未产出完整 JSON，因此本节采用本地 smoke search 与远端单模型 cache 结果共同记录。该问题不影响主要结论，因为后处理搜索的最优水平仍停在约 `89.9%`。

## 判断

手工状态机后处理可以提高事件级表现，特别是 segment recall 和 onset delay，但 strict accuracy 仍被大段 false positive / false negative 限制。仅靠 MA/debounce/gap fill 不能把结果从 `89.9%` 推到 `95%`。
