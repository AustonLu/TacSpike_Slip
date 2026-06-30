# Auto Exploration Iter07 总结

日期：2026-06-30

## 本轮目标

通过扩大 time-channel LIF-SCNN 容量和增强训练强度，把 strict 100k random tuned accuracy 推到 `>=90%`。

## 当前状态

已完成 quick 容量实验与标准 100k random/balanced/transition bucket 评估。由于容量扩张明显低于 Iter06，未继续启动 long/lr5e4 分支。

## 最佳结果

| 配置 | 100k random tuned accuracy | ROC-AUC | >100 ms tuned |
|---|---:|---:|---:|
| `iter07_time_channel_w48_h384_ignore50_v1` | 89.447% | 90.563% | 90.035% |
| `iter07_time_channel_w64_h512_ignore50_v1` | 88.711% | 88.510% | 89.305% |

对比 Iter06 最佳单模型：

| 配置 | 100k random tuned accuracy | ROC-AUC |
|---|---:|---:|
| `iter06_time_channel_random_ignore50_v1` | 89.631% | 91.828% |

## 是否达到 90%

没有。

## 结论

1. time-channel LIF-SCNN 单纯扩大到 `width=48/64` 没有改善 strict accuracy。
2. `width=64/hidden=512` 泛化明显下降，容量不是主要瓶颈。
3. 当前最可靠模型仍是 Iter06 的 `width=32/hidden=256 + random + ignore50`。
4. 下一步不应继续盲目加宽 backbone，应显式处理 transition/onset/offset 或做 sequence score cache smoothing。

## 下一轮建议

Iter08 建议转向：

1. 轻量 sequence score cache + smoothing/hysteresis，在完整 validation sequence 上评估。
2. transition/onset-aware training：增加 transition 辅助标签或对 transition 附近样本单独建权重，而不是简单过滤。
3. 若继续模型侧探索，应优先改 loss/label 结构，而不是增加卷积宽度。
