# 01 长 segment 状态微调

## 目的

验证 Iter04 状态目标微调失败是否主要由 `segment_windows=64` 太短导致。本轮将 sequence training segment 增大到 `512ms` 和 `1024ms`，保持 `context_ms=400`、`time_bins=100`，从 Iter03 最优 checkpoint 初始化。

## 配置

共同配置：

```text
init_checkpoint = seq_ft_ctx400_s32_transition_mix_lr1e4_v1/best.pt
model = time_channel_scnn
context_ms = 400
time_bins = 100
lr = 1e-5
epochs = 8
sampling = transition_mix
best_metric = valid_balanced_accuracy
save_epoch_checkpoints = true
```

训练 run：

| run | segment | train segments/epoch | val segments |
|---|---:|---:|---:|
| `ft_ctx400_seg512_no_smooth` | 512 | 1200 | 400 |
| `ft_ctx400_seg512_low_smooth` | 512 | 1200 | 400 |
| `ft_ctx400_seg1024_no_smooth` | 1024 | 600 | 200 |
| `ft_ctx400_seg1024_ignore50_no_smooth` | 1024 | 600 | 200 |

## 训练期指标

训练脚本内部短片段 validation 的 best balanced accuracy：

| run | best epoch | sampled valid balanced accuracy | sampled valid F1 |
|---|---:|---:|---:|
| `ft_ctx400_seg512_no_smooth` | 4 | 75.800% | 73.999% |
| `ft_ctx400_seg512_low_smooth` | 4 | 75.772% | 73.964% |
| `ft_ctx400_seg1024_no_smooth` | 5 | 79.939% | 78.680% |
| `ft_ctx400_seg1024_ignore50_no_smooth` | 5 | 80.955% | 79.639% |

## Full Sliding 结果

对各 run 的 `best.pt` 使用完整 selected validation sequence 做 sliding evaluation：

| run | strict accuracy | balanced accuracy | F1 | segment recall | delay p95 |
|---|---:|---:|---:|---:|---:|
| `ft_ctx400_seg512_no_smooth` | 88.878% | 86.040% | 79.409% | 100.0% | 1013.4 ms |
| `ft_ctx400_seg512_low_smooth` | 88.892% | 86.024% | 79.413% | 100.0% | 1016.7 ms |
| `ft_ctx400_seg1024_no_smooth` | 89.650% | 87.151% | 80.915% | 90.0% | 291.6 ms |
| `ft_ctx400_seg1024_ignore50_no_smooth` | 89.703% | 85.978% | 80.246% | 85.0% | 354.2 ms |

## 判断

长 segment 明显避免了 Iter04 状态目标微调的严重崩溃：Iter04 fine-tune full sliding 只有 `81.776%`，本轮 best.pt 可恢复到约 `89.7%`。但这仍低于 Iter04 三路状态头的 `90.198%`，也低于原 Iter03 + 后处理约 `89.9%` 的稳定水平。

1024ms segment 比 512ms 更稳，512ms 虽然 segment recall 到 100%，但 delay p95 超过 1 秒，说明它通过更晚、更长的 slip 状态维持换来了召回，不适合当前目标。
