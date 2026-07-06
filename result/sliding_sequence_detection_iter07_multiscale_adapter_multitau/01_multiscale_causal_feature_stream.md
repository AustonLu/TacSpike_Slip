# 01 多尺度因果特征流探索结果

## 目标

本项探索验证：在 stateful streaming SNN 中，显式提供多尺度因果历史特征，是否能缓解 1 ms raw event bin 过稀疏的问题，并提升连续滑移状态检测效果。

## 实现

新增 `feature_mode=multiscale`，在读取 stream cache 时对每个 1 ms 时刻构造多个因果累积窗口：

```text
windows = 1, 20, 50, 100, 200, 400 ms
feature_w(t) = sum event_bin[t-w+1:t]
```

双极性输入从 `[L, 2, 32, 32]` 扩展为 `[L, 12, 32, 32]`，仍然保持每 1 ms 输出一次检测分数，不引入未来信息。

本轮训练两种归一化：

- `multiscale_l384_sqrt`: `feature_w / sqrt(w)`
- `multiscale_l384_mean`: `feature_w / w`

公共配置：

```text
segment_steps = 384
transition_ignore_steps = 30
model = TacSpikeStreamingLiteSCNN
conv1/conv2/hidden = 32/64/128
epochs = 8
parameter_count = 159,362
```

## 训练结果

| run | best epoch | valid accuracy | valid balanced accuracy | valid F1 | valid ROC-AUC | valid PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| multiscale_l384_sqrt | 5 | 77.40% | 77.45% | 77.88% | 83.92% | 84.47% |
| multiscale_l384_mean | 8 | 72.85% | 72.99% | 72.78% | 77.36% | 78.20% |

`sqrt` 明显优于 `mean`。原因很可能是 `mean` 对 200/400 ms 长窗口压得过小，削弱了长时间状态证据；`sqrt` 在抑制长窗口数值爆炸的同时保留了足够幅值。

## 最终序列评估

最终使用训练后的 best checkpoint 重新做 validation 16 条 sequence 的连续检测评估：

| run | best postprocess | accuracy | balanced accuracy | F1 | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| multiscale_l384_sqrt | ma_200 + debounce on2/off50 | 88.57% | 82.42% | 76.46% | 85.48% | 69.16% |
| multiscale_l384_mean | ma_200 + debounce on8/off50 | 86.55% | 81.44% | 73.75% | 77.45% | 70.39% |

## 结论

多尺度因果输入是本轮最有效的 SNN 改进。相对 Iter06 raw 1 ms streaming SNN 的约 87.88% strict accuracy，`multiscale_l384_sqrt` 将最终 sequence-level accuracy 提升到 88.57%，balanced accuracy 到 82.42%，但仍未达到 90%/95% 目标。

关键判断：

- 显式 20-400 ms 历史特征确实有帮助，说明 raw 1 ms 输入过稀疏是重要瓶颈。
- 训练端 balanced accuracy 只有 77.45%，说明不是单纯后处理问题，前端特征和训练目标仍不足。
- recall 只有 69.16%，模型仍偏保守，当前 debounced state detection 为了减少误报牺牲了滑移召回。

## 下一步建议

优先保留 `sqrt` 多尺度输入，并继续尝试：

1. 对多尺度特征加轻量 temporal convolution / depthwise temporal mixing，再输入 SNN。
2. 将训练目标从逐点 BCE 改成 state/segment-aware loss，提高连续 slip 段召回。
3. 为 slip recall 加权或采用 focal/asymmetric loss，避免后处理阶段只能靠牺牲 recall 控误报。
