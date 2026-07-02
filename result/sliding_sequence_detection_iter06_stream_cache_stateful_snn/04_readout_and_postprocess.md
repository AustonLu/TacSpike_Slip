# 04 Readout 与全序列后处理评估

本轮评估使用完整 validation sequence 的连续输出。模型每 1 ms 输出一次 raw slip score，然后在 sequence 内做 causal moving average、EMA 和 debounce 搜索。

评估脚本：`scripts/train/evaluate_stream_cache_scnn.py`

评估方式：

- 每条 sequence 开始时 reset LIF state；
- 同一 sequence 内按 `chunk_steps=2048` 分块推理，chunk 间传递 LIF state；
- 在 16 条 validation sequence 上搜索后处理参数；
- 主排序指标为 full-sequence strict accuracy。

## 最佳后处理结果

| run | best method | strict acc | balanced acc | F1 | precision | recall | event recall | delay p95 | false alarms/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stream_l256_wide` | `ma_150_debounce_on8_off50` | 87.595% | 82.570% | 75.629% | 79.983% | 71.724% | 85.0% | 365.0 ms | 11.373 |
| `stream_l384_wide` | `ma_150_debounce_on2_off50` | 87.682% | 82.864% | 75.946% | 79.778% | 72.466% | 85.0% | 148.0 ms | 10.398 |
| `stream_l512_wide` | `ma_150_debounce_on3_off50` | 87.012% | 81.975% | 74.608% | 78.477% | 71.103% | 85.0% | 310.2 ms | 10.723 |
| `stream_l384_ignore30` | `ma_150_debounce_on2_off50` | 87.876% | 82.978% | 76.221% | 80.460% | 72.405% | 85.0% | 312.4 ms | 9.586 |

注：`stream_l512_wide` 已使用 epoch 8 最终 best checkpoint 补评估。

## 与上一轮 window-based SNN 对比

上一轮最优结果：

```text
probe_seg1024_no_smooth/best.pt
strict accuracy = 90.108%
balanced accuracy = 88.044%
F1 = 81.934%
segment recall = 95.0%
delay p95 = 678.7 ms
```

本轮最佳结果目前是 `stream_l384_ignore30`：

```text
strict accuracy = 87.876%
balanced accuracy = 82.978%
F1 = 76.221%
segment recall = 85.0%
delay p95 = 312.4 ms
```

因此本轮 stateful streaming SNN 相比上一轮 window-based SNN：

- strict accuracy 低约 `2.23%`；
- balanced accuracy 低约 `5.07%`；
- F1 低约 `5.71%`；
- segment recall 低 `10%`；
- delay p95 更短，但这是以漏检更多 slip segment 为代价。

## 判断

后处理能把 sampled validation 约 68% 的逐毫秒判别提升到约 88% strict accuracy，但仍不能达到 90%/95% 目标。说明输出 score 里有可用的时序信息，但 raw streaming SNN 的 score 分离度不够，后处理主要是在平滑噪声，而不是创造新的判别证据。
