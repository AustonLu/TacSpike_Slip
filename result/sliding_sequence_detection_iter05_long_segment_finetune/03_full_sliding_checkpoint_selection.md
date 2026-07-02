# 03 Full Sliding Checkpoint Selection

## 目的

验证训练脚本内部的 sampled validation 是否能代表最终 selected-sequence full sliding 指标；如果不能，则使用每个 epoch checkpoint 做 full sliding 选择。

## 评估内容

本轮训练已开启：

```text
--save-epoch-checkpoints
```

额外评估：

- `ft_ctx400_seg512_no_smooth` 的 epoch 1-4；
- `ft_ctx400_seg1024_no_smooth` 的 epoch 1-4；
- 两个小规模 probe run。

## 结果排序

| checkpoint | strict accuracy | balanced accuracy | F1 | segment recall | delay p95 |
|---|---:|---:|---:|---:|---:|
| `probe_seg1024_no_smooth/best.pt` | 90.108% | 88.044% | 81.934% | 95.0% | 678.7 ms |
| `probe_seg512_no_smooth/best.pt` | 89.966% | 87.647% | 81.552% | 95.0% | 485.6 ms |
| `ft_ctx400_seg1024_ignore50_no_smooth/best.pt` | 89.703% | 85.978% | 80.246% | 85.0% | 354.2 ms |
| `ft_ctx400_seg1024_no_smooth/best.pt` | 89.650% | 87.151% | 80.915% | 90.0% | 291.6 ms |
| `ft_ctx400_seg1024_no_smooth/epoch_001.pt` | 89.616% | 87.658% | 81.176% | 95.0% | 652.0 ms |
| `ft_ctx400_seg512_no_smooth/epoch_001.pt` | 89.480% | 86.905% | 80.583% | 95.0% | 628.6 ms |
| `ft_ctx400_seg512_no_smooth/epoch_002.pt` | 89.280% | 87.009% | 80.433% | 95.0% | 550.5 ms |
| `ft_ctx400_seg512_no_smooth/epoch_004.pt` | 88.878% | 86.040% | 79.409% | 100.0% | 1013.4 ms |

## 判断

Full sliding checkpoint selection 是必要的，但本轮没有找到超过 Iter04 最佳 `90.198%` 的 checkpoint。

训练脚本内部 sampled validation 与最终 full sliding 口径仍然不一致。例如 `ft_ctx400_seg1024_ignore50_no_smooth` 的 sampled valid balanced accuracy 达到 `80.955%`，但 full sliding strict accuracy 只有 `89.703%`，且 segment recall 只有 `85%`。

小规模 probe 的 `1024ms/no-smooth` 达到 `90.108%`，接近 Iter04 最佳，但 delay p95 为 `678.7ms`，事件级响应更慢。这说明轻微 fine-tune 可能保留原模型能力，但更长训练会漂移，不能稳定突破平台。
