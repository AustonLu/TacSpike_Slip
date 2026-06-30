# 02 transition-aware 训练记录

## 计划

训练四个模型：

| run_id | 训练采样 | 标签处理 |
|---|---|---|
| `iter06_time_channel_random_ignore50_v1` | random + ignore 50 ms | hard label |
| `iter06_time_channel_random_ignore100_v1` | random + ignore 100 ms | hard label |
| `iter06_time_channel_random_ignore150_v1` | random + ignore 150 ms | hard label |
| `iter06_time_channel_random_ignore100_smooth03_v1` | random + ignore 100 ms | label smoothing 0.03 |

## 待记录

训练结束后记录：

| run_id | best epoch | val accuracy during training | 100k random tuned | 100k balanced tuned | 备注 |
|---|---:|---:|---:|---:|---|
| `iter06_time_channel_random_ignore50_v1` | 10 | 89.27% | 89.631% | 86.349% | 本轮最佳单模型；ROC-AUC 91.828% |
| `iter06_time_channel_random_ignore100_v1` | 8 | 89.195% | 89.447% | 86.141% | strict 指标未超过 ignore50 |
| `iter06_time_channel_random_ignore150_v1` | 10 | 88.805% | 89.553% | 86.207% | 过滤更宽未带来收益 |
| `iter06_time_channel_random_ignore100_smooth03_v1` | 8 | 89.14% | 89.463% | 85.997% | label smoothing 没有改善 strict accuracy |

## 观察

1. `ignore50` 是最好的单模型，100k random tuned accuracy 达到 `89.631%`，相对 Iter04 最强单模型 `89.45%` 有小幅提升。
2. `ignore100/150` 没有进一步提升 strict 指标，说明过滤过宽会损失一部分边界附近的有效 hard samples。
3. label smoothing 对本轮 time-channel LIF-SCNN 没有明显收益。
4. ROC-AUC 最高为 `ignore50` 的 `91.828%`，分数可分性较 Iter04 有提升，但 accuracy 仍卡在阈值附近。
