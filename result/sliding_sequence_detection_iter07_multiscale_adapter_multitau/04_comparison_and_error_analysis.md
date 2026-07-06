# 04 对比与误差分析

## 核心对比

| 方向 | run | 参数量 | valid balanced accuracy | sequence accuracy | sequence balanced accuracy | sequence F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 多尺度因果输入 | multiscale_l384_sqrt | 159,362 | 77.45% | 88.57% | 82.42% | 76.46% | 本轮最佳 SNN |
| 多尺度因果输入 | multiscale_l384_mean | 159,362 | 72.99% | 86.55% | 81.44% | 73.75% | 归一化过强 |
| 多时间常数 SNN | multitau_l384_ignore30 | 114,918 | 70.28% | 87.70% | 81.89% | 75.16% | 有帮助但不够 |
| score adapter | adapter_iter04_ctx400 | adapter only | 64.45% | 86.06% | 77.23% | 69.13% | 前端 score 有信息但后端不够 |
| score adapter | adapter_iter04_best5 | adapter only | 69.44% | 95.57% | 67.98% | 52.89% | accuracy 虚高，recall 过低 |

## 主要发现

### 1. 最高 accuracy 不等于最佳滑移检测

`adapter_iter04_best5` 的 sequence accuracy 达到 95.57%，但 balanced accuracy 只有 67.98%，recall 只有 35.97%。这说明它主要靠极保守的触发策略获得高 accuracy，不满足滑移检测需求。

因此本项目后续不能只看 accuracy，应至少同时报告：

- balanced accuracy
- F1
- precision / recall
- event recall 或 onset recall
- detection delay
- false alarm

### 2. 多尺度输入是最有价值的结构改进

`multiscale_l384_sqrt` 在训练端和最终 sequence-level 评估都最好：

```text
valid balanced accuracy = 77.45%
sequence accuracy = 88.57%
sequence balanced accuracy = 82.42%
sequence F1 = 76.46%
```

这说明为每个 1 ms 时刻显式提供 20-400 ms 因果历史证据，比单纯依赖 LIF membrane 自己记忆更可靠。

### 3. 目前仍未超过 90%，更未接近 95%

本轮最佳有效 SNN 仍停留在 88.57% sequence accuracy。主要原因不是训练没跑够，而是：

- recall 仍偏低，best SNN recall 约 69%；
- 训练端 validation balanced accuracy 只有 77.45%，模型自身判别能力不足；
- smoothing/debounce 能提高稳定性，但会进一步推高 precision、降低 recall；
- 数据集中 slip/no-slip 状态边界可能存在标签噪声或 onset 对齐不确定，逐点 BCE 不适合直接优化连续状态检测。

### 4. multi-tau 的收益小于预期

multi-tau raw SNN 的 sequence accuracy 达到 87.70%，接近 multi-scale sqrt，但训练端 valid balanced accuracy 只有 70.28%。说明多时间常数能改善后处理后的状态连续性，但没有解决 sparse input 表征瓶颈。

## 对下一轮的判断

本轮结果支持以下路线优先级：

1. 主线继续使用 `multiscale_l384_sqrt`，不要回到纯 raw 1 ms input。
2. 优先改训练目标，而不是继续小幅调后处理。
3. 将模型优化目标转为连续状态检测，例如 segment/state-aware loss、onset-tolerant label、recall-sensitive loss。
4. 如果继续加结构，应在 multi-scale 前提下加入 temporal mixing，而不是只增加 LIF branch。

## 建议的下一轮实验

1. **state-aware training objective**
   - 对连续 slip 段做 segment-level loss。
   - transition 附近使用 soft label 或 ignore band。
   - 提高 slip recall 权重，避免模型过保守。

2. **multi-scale + temporal convolution**
   - 在 `[L, C, H, W]` 上加入轻量 causal temporal conv。
   - 再送入 SNN，降低 LIF 自行学习长时累计的难度。

3. **onset-tolerant evaluation**
   - 将逐点 accuracy 与事件级 detection 分开。
   - 用 onset delay、miss rate、false alarm rate 评价滑移检测系统。

4. **缓存最终 multiscale feature**
   - 当前每次读 slice 动态构造 1/20/50/100/200/400 ms 特征，训练较慢。
   - 若下一轮继续沿用 multi-scale，应生成固定 feature cache，减少 IO 和重复计算。
