# 01 capacity scaling 记录

## 计划

比较更大的 time-channel LIF-SCNN：

| run_id | width | hidden | epochs | train samples/epoch | 备注 |
|---|---:|---:|---:|---:|---|
| `iter07_time_channel_w48_h384_ignore50_v1` | 48 | 384 | 10 | 70000 | 中等扩容 |
| `iter07_time_channel_w64_h512_ignore50_v1` | 64 | 512 | 10 | 60000 | 大模型，降低 batch size |

## 待记录

| run_id | best epoch | 20k val acc | 100k random tuned | ROC-AUC | 结论 |
|---|---:|---:|---:|---:|---|
| `iter07_time_channel_w48_h384_ignore50_v1` | 9 | 88.66% | 89.447% | 90.563% | 不如 Iter06 w32/hidden256 |
| `iter07_time_channel_w64_h512_ignore50_v1` | 10 | 87.89% | 88.711% | 88.510% | 明显过拟合/泛化下降 |

## 观察

1. 单纯扩大 time-channel LIF-SCNN 容量没有改善 strict 100k random accuracy。
2. `width=48/hidden=384` 训练集指标继续上升，但 validation 和 100k random 指标低于 Iter06 `ignore50`。
3. `width=64/hidden=512` 的 ROC-AUC 明显下降，说明容量过大或 batch size 下降导致泛化变差。
4. 因容量扩张方向劣化，本轮不继续启动 long/lr5e4 变体，避免把算力投入已不占优的模型族。
