# 02 Stateful SCNN 训练记录

本轮目标是把训练目标从独立 sliding window 分类改成连续状态检测：输入为按 sequence cache 读取的 1 ms event bin，模型在一个训练 segment 内保留 LIF state，并每 1 ms 输出一次 slip score。

## 代码与数据

- 分支：`sliding`
- 远程工程目录：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- stream cache：`/lamport/makkapakka/jiajunlu/cache/tacspike_stream_cache_v3_sparse`
- cache 配置：`sparse_sp4_both_clipnone_float16`
- 训练数据：train split 1091 条 sequence，20,293,980 个 1 ms bin，正样本比例 0.273441
- 验证数据：val split 234 条 sequence，4,173,170 个 1 ms bin，正样本比例 0.285752

## 模型

本轮使用 `TacSpikeStreamingLiteSCNN`：

- 输入：`[B, L, 2, 32, 32]`
- LIF conv1：`2 -> 32`，`5x5`
- LIF conv2：`32 -> 64`，`3x3, stride=2`
- adaptive average pooling 到 `4x4`
- LIF FC：`64*4*4 -> 128`
- non-spiking logit head：`128 -> 2`
- 参数量：151,362
- 阈值：`0.1`
- beta：`0.85`
- readout：每 1 ms 输出 `logit_slip - logit_no_slip`

注意：评估时 state 在同一条 sequence 内跨 chunk 传递，并在 sequence 边界 reset；训练时 state 只在采样到的连续 segment 内保留，segment 边界 reset。

## 训练矩阵

| run | segment length | batch | train segments/epoch | val segments | lr | onset ignore | best epoch | best sampled valid balanced acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stream_l256_wide` | 256 ms | 16 | 24,000 | 5,000 | 3e-4 | 0 ms | 8 | 66.924% |
| `stream_l384_wide` | 384 ms | 12 | 18,000 | 4,000 | 3e-4 | 0 ms | 8 | 67.501% |
| `stream_l512_wide` | 512 ms | 8 | 14,000 | 3,000 | 2e-4 | 0 ms | 8 | 66.760% |
| `stream_l384_ignore30` | 384 ms | 12 | 18,000 | 4,000 | 3e-4 | 30 ms | 7 | 68.673% |

## 采样验证集结果

| run | valid acc | valid balanced acc | valid F1 | valid ROC-AUC | valid PR-AUC |
|---|---:|---:|---:|---:|---:|
| `stream_l256_wide` | 66.796% | 66.924% | 67.145% | 71.765% | 74.390% |
| `stream_l384_wide` | 67.344% | 67.501% | 67.065% | 72.268% | 74.367% |
| `stream_l512_wide` | 66.606% | 66.760% | 65.842% | 71.496% | 73.401% |
| `stream_l384_ignore30` | 68.452% | 68.673% | 67.722% | 73.478% | 75.595% |

## 观察

1. sparse stream cache 解决了数据读取瓶颈。训练吞吐在主实验中约 6.4k 至 6.8k items/s，验证吞吐约 13k 至 20k items/s。
2. 增大 BPTT/segment length 到 512 ms 没有提升 sampled validation，反而低于 384 ms。
3. 当前纯 1 ms stateful SNN 的 sampled validation balanced accuracy 只有 66% 至 69%，明显低于之前 window-based SNN 的 full sliding strict accuracy 约 90%。
4. firing rate 没有塌缩到全零，LIF3 平均 firing rate 约 0.21 至 0.22，说明主要问题不是完全不发放，而是状态表征和监督目标不足以学出强判别边界。

## 判断

本轮训练证明了 stream cache + stateful SNN 训练链路可运行，但当前结构和训练方式没有把 1 ms 输入累计成足够强的长期证据。它不是数据读取 bug，也不是简单把 segment length 从 256/384 拉到 512 ms 就能解决的问题。
