# Sliding Sequence Detection Iter07 计划：Multi-scale Feature、Score Adapter 与 Multi-timescale SNN

## 0. 本轮目标

本轮继续在 `sliding` 分支推进 sequence-level slip detection。Iter06 已证明 sparse stream cache 工程可用，但纯 `1 ms raw bin -> Lite stateful SCNN` 的路线只能达到约 `87.876%` full-sequence strict accuracy，低于上一轮 window-based SNN 的约 `90.108%`，也明显低于 `95%` 目标。

本轮按 `docs/plan/snn_slip_detection_experiment_plan.md` 的实验路线执行，重点探索三个方向：

1. **multi-scale causal feature stream**：每 1 ms 仍输出一次，但输入显式包含多个因果时间尺度的事件累计特征。
2. **score cache + temporal/state adapter**：用已有较强 window/sequence score 作为前端，再训练轻量后端做连续状态检测。
3. **multi-timescale SNN**：在 SNN 内部显式使用不同 LIF 时间常数，提升对短时事件变化和长时 slip 状态的同时建模能力。

## 1. 成功标准

主目标：

```text
validation full-sequence strict accuracy >= 95.0%
```

阶段性目标：

```text
strict accuracy > 90.108%  # 超过上一轮 window-based SNN 最优
balanced accuracy >= 88.0%
F1 >= 82.0%
event recall >= 95.0%
delay p95 不显著劣于上一轮
```

如果没有达到目标，本轮需要回答：

- 瓶颈是否来自 1 ms raw input 过稀疏；
- 显式多尺度输入是否比纯 LIF memory 更有效；
- score adapter 是否能说明“前端特征强度”是主要瓶颈；
- multi-timescale LIF 是否能改善 recall 和 false alarm trade-off。

## 2. 本轮目录结构

```text
result/sliding_sequence_detection_iter07_multiscale_adapter_multitau/
  PLAN.md
  01_multiscale_causal_feature_stream.md
  02_score_cache_temporal_adapter.md
  03_multitau_snn.md
  04_comparison_and_error_analysis.md
  SUMMARY.md
  remote_summaries/
```

## 3. 探索项 1：multi-scale causal feature stream

### 假设

Iter06 的模型失败，不是因为 sequence cache 错，而是因为单个 1 ms event bin 太稀疏。模型只靠 LIF membrane 自己累计几百毫秒历史证据，学习难度太高。

### 方法

在读取 stream cache 时，对每个时刻构造多个因果累计窗口：

```text
windows = 1,20,50,100,200,400 ms
```

每个尺度只使用当前时刻及过去事件，不使用未来信息：

```text
feature_w(t) = sum_{tau=max(0,t-w+1)}^t event_bin(tau)
```

然后将不同尺度沿 channel 维拼接：

```text
raw:        [B, L, 2, 32, 32]
multiscale: [B, L, 12, 32, 32]
```

首选归一化：

```text
sqrt normalization: feature_w / sqrt(w)
```

备选：

```text
mean normalization: feature_w / w
none normalization: feature_w
```

### 实验配置

优先训练：

```text
run: multiscale_l384_sqrt
segment_steps: 384
feature_windows: 1,20,50,100,200,400
normalization: sqrt
model: TacSpikeStreamingLiteSCNN
conv1/conv2/hidden: 32/64/128
transition_ignore_steps: 30
```

如首个 run 明显优于 Iter06，再补：

```text
multiscale_l384_mean
multiscale_l512_sqrt
```

### 记录文件

```text
01_multiscale_causal_feature_stream.md
remote_summaries/multiscale_*.json
```

## 4. 探索项 2：score cache + temporal/state adapter

### 假设

如果用较强 window-based 模型产生每 1 ms 的 slip score，再训练一个轻量 temporal/state adapter 能显著超过当前 stateful SNN，则说明主要瓶颈在前端特征，而不是 sequence-level 后处理。

### 方法

优先复用已有 full-sequence score cache 或评估脚本输出。如果没有可复用 score cache，则先用当前最强可用 checkpoint 生成 validation/train score cache。

adapter 输入：

```text
score(t), causal moving average scores, delta score, short/long score contrast
```

adapter 候选：

1. logistic/MLP temporal adapter，作为非 SNN 上界；
2. lightweight LIF temporal adapter，作为 SNN-compatible adapter。

首版以快速判别为主：

```text
features = raw score + causal MA(20/50/100/200/400) + score deltas
loss = BCE with transition ignore
selection = full-sequence strict accuracy / balanced accuracy
```

### 记录文件

```text
02_score_cache_temporal_adapter.md
remote_summaries/score_adapter_*.json
```

## 5. 探索项 3：multi-timescale SNN

### 假设

单一 beta 的 LIF 难以同时处理短时事件突变和长时 slip 状态。并联 fast/mid/slow LIF 分支可以提高连续状态检测能力。

### 方法

实现多分支 streaming SNN：

```text
branch_fast: beta ~= 0.60-0.70
branch_mid:  beta ~= 0.85
branch_slow: beta ~= 0.95-0.98
```

每个 branch 使用较小 Lite-SCNN，最终对 logits 做平均或轻量 fusion。

优先配置：

```text
run: multitau_l384_ignore30
input: raw 1 ms bins
segment_steps: 384
branch width: 16/32/64
betas: 0.65,0.85,0.95
transition_ignore_steps: 30
```

如果 multi-scale feature 明显更强，再做：

```text
multiscale_multitau_l384
```

### 记录文件

```text
03_multitau_snn.md
remote_summaries/multitau_*.json
```

## 6. 执行顺序

1. 扩展 stream cache 读取函数，支持 multi-scale causal features。
2. 扩展训练/评估脚本，支持 `feature_mode=raw|multiscale`。
3. 实现 multi-timescale SNN，并让训练/评估脚本可从 checkpoint args 自动恢复模型。
4. 实现 score adapter 脚本，优先复用已有 score cache。
5. 同步代码到远程训练目录。
6. 在 A100 上运行核心实验。
7. 拉回 JSON summary 和 detection result。
8. 按探索项分别写 markdown，并写 `SUMMARY.md`。

## 7. 本轮判断优先级

本轮不追求盲目堆实验数量，而是优先回答路线判断：

1. multi-scale causal feature 是否显著强于 raw 1 ms input；
2. score adapter 是否说明前端特征是主瓶颈；
3. multi-timescale LIF 是否能改善 recall / false alarm / delay trade-off；
4. 如果三者都不能超过 90%，则需要重新审视标签定义、train/val sequence 分布和评价目标。

