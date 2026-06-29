# Streaming SNN 实验

状态：首轮完成

目的：把模型从逐窗独立分类推进到真正的 streaming 形式，使用连续输入和状态延续。

## 实现

新增模型：

```text
TacSpikeStreamingLiteSCNN
```

结构复用 Lite-SCNN 的 conv + LIF + fc hidden，但支持每 1 ms 输入一个 event bin，并在时间上保留 LIF 状态。输出层使用连续 logits，不让最后一层强制 fire。

新增训练脚本：

```bash
scripts/train/train_streaming_scnn.py
```

训练方式：

- truncated BPTT
- 从连续 sequence 中采样固定长度片段
- 每个时间步输出 logits
- 默认 `loss_mode=all`，即对片段内每个 ms 都计算 cross entropy

## 实验结果

### `stream_lite_t128_v1`

命令摘要：

```bash
python scripts/train/train_streaming_scnn.py \
  --segment-steps 128 \
  --epochs 5 \
  --train-segments-per-epoch 20000 \
  --val-segments 10000 \
  --batch-size 256 \
  --sampling balanced \
  --loss-mode all \
  --threshold 0.1 \
  --beta 0.85 \
  --scheduler cosine \
  --amp
```

结果：

| Run | Segment | Loss | Best epoch | Val acc | Balanced acc | F1 | PR-AUC | ROC-AUC | 参数量 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `stream_lite_t128_v1` | 128 ms | all steps | 5 | 70.20% | 70.10% | 67.67% | 76.55% | 74.55% | 38,306 |

### `stream_lite_t256_last_v1`

目的：测试首轮 streaming 失败是否主要来自 `loss_mode=all` 对过渡时间步惩罚过强，因此改为 256 ms 片段并只在最后一步计算 loss。

命令摘要：

```bash
python scripts/train/train_streaming_scnn.py \
  --segment-steps 256 \
  --epochs 6 \
  --train-segments-per-epoch 30000 \
  --val-segments 10000 \
  --batch-size 128 \
  --sampling balanced \
  --loss-mode last \
  --threshold 0.1 \
  --hidden-dim 64 \
  --scheduler cosine \
  --amp
```

| Run | Segment | Loss | Best epoch | Val acc | Balanced acc | F1 | PR-AUC | ROC-AUC | 参数量 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `stream_lite_t256_last_v1` | 256 ms | last step | 6 | 72.66% | 72.41% | 69.18% | 79.99% | 77.87% | 38,306 |

## 结论

当前最直接的 stateful streaming + truncated BPTT 没有提高精度，反而明显低于 100 ms sliding-window Lite-SCNN：

- `ctx100_lite_scnn_v1`：100k natural accuracy `83.62%`，balanced accuracy `77.36%`
- `stream_lite_t128_v1`：segment validation accuracy `70.20%`
- `stream_lite_t256_last_v1`：segment validation accuracy `72.66%`

`loss_mode=last` 和更长 segment 能带来小幅提升，但仍明显低于 sliding-window 长上下文模型：

- `ctx300_lite_scnn_v1`：100k natural accuracy `85.44%`，balanced accuracy `80.09%`
- `ctx500_frame_cnn_v1`：100k natural accuracy `88.72%`，balanced accuracy `85.18%`

这说明 streaming 不能简单理解为“保留状态就会更准”。当前失败可能来自以下原因：

1. `loss_mode=all` 对 slip onset 前后的边界时间步惩罚过强；改为 `loss_mode=last` 后有改善，但不是主要瓶颈。
2. streaming 输入每步只有 1 ms event，而 sliding-window 模型每次显式看到 300-500 ms 历史；虽然 LIF 状态能积累历史，但当前结构/阈值/泄漏可能不足以保留可判别信息。
3. 训练片段按末端标签 balanced，但片段内部仍含大量过渡和混合标签，训练目标比 window 训练更难。
4. 还没有做与 streaming 形式匹配的输出平滑和 segment-level loss。

后续如果继续 streaming，应优先尝试：

- 只对后半段计算 loss，而不是全序列或最后一步二选一；
- 更长片段 `512 ms`，并用更低时间分辨率或缓存 voxel 避免训练过慢；
- 增大 hidden 或加入轻量 temporal readout；
- 对 streaming 输出再做 EMA/MA 平滑；
- 先用 `ctx100_lite_scnn_v1` 做 streaming-like 状态蒸馏，而不是从头训练。
