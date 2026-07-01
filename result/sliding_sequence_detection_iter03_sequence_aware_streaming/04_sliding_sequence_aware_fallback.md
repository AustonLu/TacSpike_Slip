# 04. Sliding-Window Sequence-Aware Fallback

## 目的

在纯 1ms streaming 失败后，回到 Iter02 确认最有效的 `400ms / 100 bins` sliding-window 表征，但训练目标从独立窗口分类改成连续状态输出。

## 新增脚本

新增 `scripts/train/train_sequence_scnn.py`：

- 输入仍是 `time_channel_scnn` 的 400ms sliding window。
- batch 由同一 sequence 的连续窗口片段组成。
- loss 对 `[B, S]` 连续输出计算 CE。
- 可加入同标签相邻 score smoothness 和 flip penalty。
- 支持从已有 checkpoint 初始化。

## 结果

| run | 初始化 | sampling | lr | sampled valid acc | sampled valid ROC-AUC | 完整 sequence best accuracy |
|---|---|---|---:|---:|---:|---:|
| `seq_ctx400_s32_transition_mix_smooth_v1` | 从零 | transition mix | 1e-3 | 0.518156 | 0.426216 | 未评估，训练退化 |
| `seq_ft_ctx400_s32_transition_mix_lr1e4_v1` | Iter02 `ctx400` | transition mix | 1e-4 | 0.739719 | 0.794648 | 0.899023 |
| `seq_ft_ctx400_s32_random_lr5e5_v1` | Iter02 `ctx400` | random | 5e-5 | 0.869594 | 0.892701 | 0.895297 |

完整 sequence 评价：

| run | best method | accuracy | balanced acc | F1 | segment recall | missed | false alarms/min | p95 delay |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| transition-mix fine-tune | `ma_100_debounce_on5_off20` | 0.899023 | 0.877657 | 0.815493 | 0.95 | 1 | 1.787 | 231.3ms |
| random fine-tune | `ma_100_debounce_on5_off20` | 0.895297 | 0.869657 | 0.806736 | 0.90 | 2 | 0.812 | 283.0ms |

## 判断

从零 sequence-aware 训练失败，说明连续片段 loss 自身不能替代 400ms window 表征学习。

从 Iter02 checkpoint 初始化后，transition-mix fine-tune 能把 sequence accuracy 从 Iter02 `0.89714` 小幅推到 `0.89902`，但仍未达到 90%，更远低于 95%。random fine-tune 的 sampled validation 更自然，但完整 sequence 指标反而低一些。
