# TacSpike 1kHz DVS 视触觉滑移检测 SNN 实验方案

本文档用于设计一个轻量化 SNN，对 TacSpike 1kHz DVS-like 视触觉事件数据做滑移二分类预处理和检测。方案基于当前数据集、参考论文和截至 2026-06-25 的常用 SNN 框架调研。

## 1. 任务和数据集约束

数据集：[`AustonLu/tacspike-slip-detection-1khz`](https://huggingface.co/datasets/AustonLu/tacspike-slip-detection-1khz)

核心信息：

- 任务：`slip_binary`，二分类，`no-slip` / `slip`。
- 事件来源：`v2e + SuperSloMo` 生成的 1kHz DVS-like 事件。
- 时间分辨率：1 ms。
- 窗口：20 ms，stride 1 ms。
- 默认输入体素：`[20, 2, 128, 128]`，即 `T=20`、双极性、空间分辨率 128x128。
- 数据规模：1559 个 sequence；train/val/test 为 1091/234/234 个 sequence。
- 窗口数量：约 2895 万；train/val/test 为 2029 万 / 417 万 / 449 万。
- 类别比例：slip 约 27.43%，no-slip 约 72.57%。
- 平均每 20 ms window 事件数约 2.99，空窗口约 33.25%。
- HDF5 结构：`events/t,x,y,p`，`windows/t_start,t_end,t_label,event_count`，`label/slip`。

这个数据集非常稀疏，不能只看 accuracy。多数类 no-slip baseline 已经约 72.24%，因此实验必须同时报告 precision、recall、F1、balanced accuracy、PR-AUC 和检测延迟。

## 2. 参考论文要点

参考论文：`docs/ref/A Neuromorphic Incipient Slip Detection System using Papillae Morphology.pdf`，arXiv: [2509.09546](https://arxiv.org/abs/2509.09546)。

论文做法摘要：

- 目标是三分类：no slip、incipient slip、gross slip。
- 传感器为 NeuroTac + 同心 papillae 皮肤结构，外圈先发生局部滑移，中心后滑移。
- 只使用正极性事件，以降低计算负载和响应延迟。
- 原始 640x480 事件先裁剪到 400x400，再用 20x20 非重叠 pooling，下采样到 20x20。
- 每个样本为 30 ms，拆成 30 个 1 ms time step，输入张量为 `(30, 1, 20, 20)`。
- 网络是两层 convolution + 两层 fully-connected，神经元为 IAF。
- 第二个卷积层使用 stride=2 替代 pooling。
- 使用 surrogate gradient 训练，输出层按 spike count / spike rate 判别类别。
- 推理后对最终层 spike count 做滑动窗口平均，论文使用 window length=4，并要求胜出类别 spike count 至少比其他类别高 2。
- 论文测试集三分类 accuracy 为 94.33%；动态重力滑移实验中，平滑后 incipient slip 至少提前 gross slip 360 ms 被检测到。

对 TacSpike 的启发：

- “少层 SCNN + spike count readout + 时序平滑”是合理基线。
- 论文输入已经被强下采样到 20x20；TacSpike 数据是 128x128 双极性，因此直接照搬网络会浪费计算且可能过拟合稀疏噪声。
- 论文只用正极性事件，但 TacSpike 已提供双极性；主实验建议保留双极性，做正极性-only 消融。
- 论文为三分类，TacSpike 当前是二分类；输出层改为 2 个神经元。

## 3. 推荐网络结构

主推模型命名：`TacSpike-Lite-SCNN-v1`。

输入：

```text
x: [B, T=20, C=2, H=128, W=128]
```

预处理：

1. 按 window 动态 voxelize，得到 20 个 1 ms bin。
2. 每个 bin 内同一像素同一 polarity 的事件计数裁剪到 `0/1` 或 `0/2`，首版建议 `clip_max=1`，保持输入为脉冲。
3. 4x4 空间 sum pooling，把 128x128 降到 32x32：

```text
[B, 20, 2, 128, 128] -> [B, 20, 2, 32, 32]
```

主网络：

```text
Input:           [B, T, 2, 32, 32]

Conv1:           Conv2d(2, 16, kernel=5, stride=1, padding=2, bias=False)
Neuron1:         LIF, threshold=1.0, beta=0.75~0.90
Output1:         [B, T, 16, 32, 32]

Conv2:           Conv2d(16, 32, kernel=3, stride=2, padding=1, bias=False)
Neuron2:         LIF, threshold=1.0, beta=0.75~0.90
Output2:         [B, T, 32, 16, 16]

Spatial readout: AdaptiveAvgPool2d(4x4) 或 SumPool2d(4x4)
Flatten:         [B, T, 512]

FC1:             Linear(512, 64, bias=False)
Neuron3:         LIF

FC2:             Linear(64, 2, bias=False)
Output neuron:   LIF readout 或 non-spiking membrane readout

Decision:        对 T=20 的输出 spike count 求和，argmax 得到 no-slip/slip。
```

参数量估算：

- Conv1：约 800。
- Conv2：约 4608。
- FC1：约 32768。
- FC2：约 128。
- 总计约 3.8 万参数，不含可选 BatchNorm/阈值参数。

首版建议保留 `16/32/64` 这个宽度，因为数据虽然稀疏，但空间模式和双极性关系可能比论文 20x20 正极性输入更复杂。若准确率足够，再压缩到 tiny 版本：

```text
Conv1 2->8, Conv2 8->16, AdaptivePool 4x4, FC 256->32, FC 32->2
```

tiny 版本参数量约 1 万，更适合作为最终边缘部署模型。

不建议首版直接使用全分辨率 128x128 的普通卷积。考虑到平均每个 20 ms window 只有约 3 个事件，4x4 pooling 能显著降低地址空间和训练显存，同时保留 1kHz 时序信息。

## 4. 训练框架调研和推荐

常用 SNN 框架：

- [SpikingJelly](https://spikingjelly.readthedocs.io/zh-cn/latest/)：基于 PyTorch 的 SNN 深度学习框架，包含神经元、surrogate gradient、神经形态数据处理、DVS Gesture 示例、训练显存优化、算子/能耗估计，以及向 Lava/Loihi 方向转换的教程。适合本项目作为主训练框架。
- [snnTorch](https://snntorch.readthedocs.io/en/latest/)：基于 PyTorch，API 简单，教程完整，明确支持 surrogate gradient 和 BPTT。适合快速教学、原型和小模型验证。
- [Norse](https://norse.github.io/norse/)：基于 PyTorch，偏研究型，神经元模型丰富，API 较底层，适合做自定义神经动力学。
- [Sinabs](https://sinabs.readthedocs.io/v3.1.3/)：论文使用的框架，支持 IAF/LIF、BPTT、surrogate gradient，并偏向 SynSense Speck/DynapCNN 部署。若后续目标硬件是 Speck/DynapCNN，值得作为第二阶段迁移框架。
- [Lava](https://lava-nc.org/)：Intel 神经形态软件栈，适合 Loihi 部署和 process-based neuromorphic execution。作为训练首选不如 PyTorch 系框架直接。
- [Tonic](https://tonic.readthedocs.io/en/latest/)：不是训练框架，而是事件数据集和变换库，可用于 event transforms、voxel/frame 转换、DVS 数据加载思路。
- BindsNET、Brian2、NEST 等：更偏 SNN 仿真或较早期机器学习原型，不建议作为本项目主线。

推荐：

- 主线训练框架：`PyTorch + SpikingJelly`。
- 原因：TacSpike 是自定义 HDF5 数据，PyTorch DataLoader 最方便；SpikingJelly 与 PyTorch 深度集成，能直接做 surrogate gradient BPTT；后续可做 firing rate、SynOps、能耗估计和硬件迁移预研。
- 复现实验的对照框架：`Sinabs`。如果要严格复现参考论文风格，可用 Sinabs 复刻 20x20 正极性输入和 IAF SCNN。

## 5. 训练方法

推荐训练方法：surrogate gradient + BPTT。

理由：

- 输入只有 20 个 time step，完整 BPTT 的显存和计算压力可控。
- 滑移检测依赖 20 ms 内的时序变化，不宜把时间维直接平均成普通 CNN 输入。
- surrogate gradient 是当前训练深层 SNN 最常用、工程上最稳定的方法。

损失函数：

- 首版：输出层对 20 个 time step 的 spike count 求和，得到 `[B, 2]`，用 class-weighted cross entropy。
- 类别权重：根据训练集比例初始化，例如 no-slip 权重约 `1/0.7257`，slip 权重约 `1/0.2743`，再按验证集 PR-AUC/F1 调整。
- 备选：focal loss，用于减少大量 easy no-slip 窗口的主导效应。
- 可加 firing-rate regularization，约束隐藏层平均发放率，避免模型靠高频放电换准确率。

优化器和超参建议：

```text
optimizer: AdamW
lr: 1e-3 起步，cosine decay 或 ReduceLROnPlateau
weight_decay: 1e-4
batch_size: 128~512，取决于 GPU 显存
epochs: 30~80
gradient_clip: 1.0
beta/tau: beta=0.75~0.90，后续可做可学习 beta
threshold: 1.0
surrogate: ATan 或 sigmoid surrogate
```

采样策略：

- 不建议每个 epoch 直接遍历全部 2029 万训练 window，窗口 stride=1 ms，高度相关且训练成本不必要。
- 建议每个 epoch 从 sequence 中按比例随机采样，例如 20 万到 100 万窗口。
- 对 slip/no-slip 做 balanced sampling，同时保留一部分自然分布 batch，用于避免部署时先验偏移。
- 对 no-slip 类拆分为 empty no-slip 和 non-empty no-slip；至少保证 non-empty no-slip 占 no-slip 样本的一定比例，避免模型学成“有事件就是 slip”。
- sequence split 已无泄漏，训练和评估必须保持 sequence 级隔离。

数据增强：

- 可用：event dropout、polarity dropout、空间平移 1~2 px、轻微空间 jitter、时间 jitter 1 ms。
- 谨慎使用：水平/垂直翻转，只有在触觉坐标和滑移方向标签不相关时才启用。
- 不建议：time reversal，因为会破坏滑移检测的因果方向。

## 6. 推理和在线检测

首版推理采用严格 sliding-window 方式：在线输入维护一个 20 ms rolling event buffer，每 1 ms 更新一次窗口，然后把最近完整 20 ms 事件重新 voxelize 后送入模型。也就是说，模型每 1 ms 输出一次判断，但每次判断都基于最近 20 ms 上下文，而不是只看新来的 1 ms event。

选择 strict sliding window 的原因：

- 与训练样本定义完全一致，便于先验证模型能力和标签对齐。
- 每个窗口内部的 LIF 状态从零初始化，避免不同 sequence 或不同窗口之间的状态泄漏。
- 代价是相邻窗口有大量重复计算，后续部署优化时再改为 stateful streaming。

后续 streaming 版本的目标是：每 1 ms 只输入新到的 event bin，并保留卷积层和 LIF 神经元状态，从而复用前 19 ms 的历史状态。这个版本需要单独校准状态 reset、泄漏时间常数、输出平滑和训练/推理分布差异，因此不作为首版实验范围。

单次分类：

```text
score_c(t) = sum_{tau=t-19}^{t} output_spike_c(tau)
pred(t) = argmax_c score_c(t)
```

平滑：

- 对最近 `N=5~10` 个 1 ms 输出做 causal moving average 或 EMA。
- 首版建议 `N=5`，再测试 `N=1/3/5/10` 的延迟和误报 trade-off。
- 论文使用 window length=4，但它的样本设置是 30 ms；TacSpike 是 1 ms stride，不能机械照搬。

触发规则：

```text
smoothed_score_slip - smoothed_score_no_slip >= margin
```

首版 margin 可设为 1 或 2 个 spike count 等价差值。如果使用 membrane/logit readout，则按验证集 PR 曲线选 threshold，优先控制 false alarm。

检测延迟定义：

- 对每个 sequence 找 label 从 0 到 1 的首次时间作为 ground-truth slip onset。
- 预测首次连续 `K` 次为 slip 的时间作为 detection onset，建议 `K=2~3`。
- latency = detection onset - ground-truth onset，单位 ms。
- 同时统计 onset 前 false alarm 次数或 false alarm rate。

## 7. 实验矩阵

必须完成的主实验：

1. `TacSpike-Lite-SCNN-v1`，双极性输入，4x4 pooling，surrogate BPTT。
2. 论文风格 baseline：正极性-only，空间下采样到 20x20，两层卷积 + 两层全连接，神经元仍使用 LIF。

建议消融：

- polarity：双极性 vs 正极性-only vs 合并极性。
- spatial pooling：2x2、4x4、8x8。
- 时间窗：10 ms、20 ms、30 ms。
- readout：output spike count vs output membrane potential。
- 平滑：N=1、3、5、10。
- class imbalance：class weight vs balanced sampler vs focal loss。
- 模型宽度：tiny 8/16/32 vs lite 16/32/64。
- 推理形态：strict 20 ms sliding window 作为首版；stateful streaming 只做后续优化对照。

核心评价指标：

- 窗口级：accuracy、balanced accuracy、precision、recall、F1、ROC-AUC、PR-AUC、confusion matrix。
- sequence 级：slip onset detection latency、false alarm rate、miss rate、平均提前/滞后时间。
- 计算代价：参数量、平均 firing rate、SynOps/window、推理时间/window、GPU/CPU 端吞吐。

## 8. 实现路线

阶段 1：数据读取和 sanity check

- 写 `TacSpikeH5Dataset`，从 `manifest_sequences.csv` 和 HDF5 中按 split 读取 sequence。
- 使用 `events/t,x,y,p` 和 `windows/t_start,t_end` 动态 voxelize。
- 写一个 `inspect_sample.py`，随机画出 20 ms event raster / pooled voxel，确认标签和窗口对齐。

阶段 2：训练主模型和核心消融

- 训练 `TacSpike-Lite-SCNN-v1` 主模型：双极性输入、4x4 pooling、20 ms window、LIF、spike count readout。
- 每个 epoch 记录窗口级指标、类别分布、隐藏层 firing rate、参数量和平均 spike count。
- 完成核心消融实验：polarity、spatial pooling、时间窗、readout、平滑窗口、class imbalance 策略和模型宽度。
- 训练论文风格 baseline：正极性-only、20x20 输入、两层卷积 + 两层全连接、LIF，用于判断 TacSpike 输入分辨率和双极性信息是否带来增益。
- 使用 validation set 选择主模型和阈值，test set 只用于最终报告。

近期输出物：

- 可复现实验配置文件。
- 主模型训练日志和 checkpoint。
- 消融实验汇总表。
- 窗口级 test 指标和 confusion matrix。

远期目标：

- 做 sequence-level 在线评估：对 val/test sequence 按时间顺序 strict sliding-window 推理，加平滑和触发规则，输出每个 sequence 的检测时刻、延迟和误报。
- 从 strict sliding-window 推理迁移到 stateful streaming：每次只输入新 1 ms event bin，保留 LIF 状态，并和 strict sliding-window 的输出、延迟、误报率做对齐验证。
- 在 v1 准确率足够后训练 tiny 版本，与 streaming 优化并列作为部署前轻量化方向。
- 做 quantization-aware 或权重离散化预研。
- 用 SpikingJelly 的算子/能耗估计先做 SynOps 统计。
- 如果目标硬件偏 SynSense，迁移到 Sinabs/DynapCNN；如果目标偏 Loihi，再评估 Lava/NIR 路径。

## 9. 当前建议结论

首个可执行实验不要直接复现论文的 20x20 正极性配置作为最终模型，而应把它作为 baseline。TacSpike 数据集的默认输入是 `[20,2,128,128]`，并且极度稀疏，因此主线模型应保留双极性、先做事件空间 pooling，再用两层轻量 SCNN 提取时空模式。

推荐主线是：

```text
4x4 event sum pooling
-> Conv(2,16,5) + LIF
-> Conv(16,32,3,stride=2) + LIF
-> AdaptivePool(4x4)
-> FC(512,64) + LIF
-> FC(64,2)
-> 20 ms spike count readout
-> 5 ms causal smoothing + margin/K-consecutive trigger
```

训练用 `PyTorch + SpikingJelly`，方法用 surrogate gradient BPTT。这个组合在实现难度、训练稳定性、后续能耗统计和硬件迁移之间最均衡。
