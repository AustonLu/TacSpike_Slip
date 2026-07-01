# 03. Streaming 完整序列评估

## 评价口径

对 validation split 固定 16 条完整 sequence 做逐 ms streaming 推理，并在完整 score 序列上搜索 MA / EMA / debounce。

| run | best method | accuracy | balanced acc | F1 | segment recall | missed | false alarms/min | FP window frac | p95 delay |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stream_t400_all` | `ma_50_debounce_on2_off10` | 0.852975 | 0.776586 | 0.690692 | 1.00 | 0 | 32.007 | 0.0585 | 1130.8ms |
| `stream_t400_tail200` | `ma_50_debounce_on2_off10` | 0.852515 | 0.782341 | 0.696586 | 0.90 | 2 | 36.232 | 0.0662 | 546.2ms |
| `stream_t512_tail256` | `ma_50_debounce_on5_off2` | 0.856980 | 0.795456 | 0.713203 | 1.00 | 0 | 43.218 | 0.0717 | 416.8ms |

## 结论

纯 1ms streaming LIF-SNN 没有接近 Iter02 的 400ms sliding baseline：

- Iter02 `ctx400` sequence accuracy：0.89714。
- 本轮最好 streaming sequence accuracy：0.85698。
- false alarm runs/min 也显著更差。

因此，当前轻量 streaming SNN 从零训练不是通往 95% 的有效方向。后续应优先保留 400ms sliding-window 表征，再让输出层或后处理满足连续状态约束。
