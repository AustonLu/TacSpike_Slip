# 05 误差分析与校准

本轮没有做完整 per-material/per-shape 的细粒度错误归因，主要完成了 sequence-level stream cache、stateful SNN 训练和 full-sequence 后处理评估。这里记录从已有指标能支持的判断。

## 现象

1. full-sequence strict accuracy 最高约 87.9%，低于上一轮 window-based SNN 约 90.1%，更低于 95% 目标。
2. sampled validation balanced accuracy 最高约 68.7%，说明原始逐毫秒 score 的判别边界较弱。
3. 后处理后 precision 约 80%，recall 约 72%，event recall 只有 85%，仍有真实 slip segment 被漏检。
4. `stream_l384_ignore30` 的 false alarms/min 最低，说明忽略 onset 附近 loss 有助于减少一部分抖动，但不能显著提高 recall。
5. `stream_l512_wide` 没有因为更长 BPTT 更好，整体 strict accuracy、balanced accuracy 和 F1 都低于 384 ms 配置，说明简单拉长 segment 不等于模型能有效保留长上下文。

## 可能原因

### 原因 A：1 ms 原始输入过稀疏

DVS 视触觉事件在 1 ms bin 内的信息量很少，当前 SCNN 需要仅依靠 LIF membrane 自己累计证据。这个累计能力可能弱于显式 400/500 ms window 特征。

### 原因 B：训练和真实 streaming 仍不完全一致

评估时 state 可以贯穿整条 sequence；训练时 state 只贯穿采样 segment，并在 segment 起点 reset。虽然 segment 长度达到 256/384/512 ms，但模型没有经历完整 sequence 的长期状态漂移、稳定接触段和多次 transition。

### 原因 C：后处理指标和训练 loss 不一致

训练优化逐毫秒 BCE，最终指标却依赖 moving average、debounce、false alarm rate、event recall 和 delay。当前 loss 没有直接约束 false alarm run、漏检 segment 或延迟。

### 原因 D：模型缺少强 feature extractor

上一轮较强结果来自重叠 window/context。当前模型直接吃 1 ms bin，容量虽然加到 151k 参数，但没有显式 temporal convolution、delay line、multi-timescale synapse 或 window feature memory。

## 校准结论

这轮结果不支持“只用当前 Lite stateful SCNN + BPTT 拉长 + 后处理”达到 95%。目前最可信的方向是保留 stream cache，但不要让模型只看单个 1 ms bin；应让输入或模型内部显式形成多尺度历史证据。

## 下一步建议

1. 构造 feature stream：从 1 ms cache 在线形成 50/100/200/400 ms 的累计 event features，输入轻量 SNN 或 SNN temporal adapter。
2. 使用 window-based 强模型产生 score cache，再训练一个轻量 state adapter 做连续状态检测，作为从 window 模型到 streaming 模型的桥接。
3. 尝试 multi-timescale SNN：并联不同 beta/threshold 的 LIF 分支，或加入 causal temporal convolution，再输出 1 ms score。
4. 训练目标加入 sequence-level surrogate loss：对 false alarm runs、missed segment、delay 设计可微近似或阶段性选择 full-sequence checkpoint。
5. 扩大 full validation 覆盖范围。当前后处理搜索只用 16 条 validation sequence，下一轮应至少补一版全 val sequence 评估，避免 16 条样本上的后处理参数偶然性。
