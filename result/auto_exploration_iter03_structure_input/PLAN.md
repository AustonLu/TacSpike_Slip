# Auto Exploration Iter03: 结构与输入表示探索计划

日期：2026-06-30

分支：`auto-snn-accuracy-exploration`

## 背景

Iter02 说明 random sampling、distillation、focal loss 和 margin regularization 都没有把 SNN 推到 `90%`。最佳结果为：

- `iter02_random_wide2_v1` 100k random tuned accuracy：`87.46%`
- `iter02_random_wide2_v1` 100k balanced tuned accuracy：`83.28%`
- ROC-AUC：约 `88.89%`

这低于 Iter01 ensemble 的 `88.08%` random tuned，也低于当前最强 CNN upper-bound `ctx500_frame_cnn_v1` 的 `88.72%` random tuned。

## 核心假设

当前 sequential Lite-SCNN 每个 5 ms bin 单独做 2D convolution，然后通过 LIF state 递推。这个设计可能没有很好保留 500 ms / 100 bins 的时间位置身份；而 FrameCNN 把 time bins 直接作为 channel 输入，能显式看到整段时间模式。因此 Iter03 重点检查：

1. SNN 差距是否主要来自输入表示，而不是 loss 或采样。
2. time-channel 输入 + LIF 隐层是否能更接近 CNN teacher。
3. 局部 3D temporal convolution + LIF 是否比逐 bin 2D SCNN 更适合滑移时间模式。
4. 单纯继续加宽 sequential SCNN 是否还有收益。

所有新增 SNN 继续使用 LIF，不使用 IAF。

## 探索项

### 01. Time-Channel LIF-SCNN

新增 `time_channel_scnn`：将输入 `[B, T, C, H, W]` reshape 为 `[B, T*C, H, W]`，使用 CNN 类似的 time-channel 前端，但每个隐层激活改为 LIF surrogate spike。

目标：检查是否能把 CNN 的时间通道表达迁移到 SNN。

记录文件：`01_time_channel_scnn.md`

### 02. Temporal-Conv LIF-SCNN

新增 `temporal_conv_scnn`：使用 3D convolution 在时间和空间上联合提取局部模式，每层后接 LIF。它比 time-channel 更接近事件流时序建模，但保留卷积局部性。

目标：检查 3D temporal feature 是否优于 sequential 2D SCNN。

记录文件：`02_temporal_conv_scnn.md`

### 03. Wide3 Sequential SCNN

在现有 sequential Lite-SCNN 上继续加宽到 conv `48/96`、hidden `384`，验证容量是否仍是主要限制。

目标：如果 wide3 明显优于 wide2，说明模型容量仍有收益；如果不提升，则优先放弃单纯扩宽。

记录文件：`03_wide3_sequential_scnn.md`

### 04. 候选 full training 和 100k 评估

先做 quick run 筛选。如果某个候选的 quick 结果超过历史同类 quick 或接近 `84%+` validation accuracy，则扩大为 full training，并做 100k random/balanced 评估。

记录文件：`04_full_eval.md`

## 成功标准

优先目标：

- 100k random validation tuned accuracy >= `90%`
- 或 100k balanced validation tuned accuracy >= `90%`

次级目标：

- 超过 Iter01 ensemble random tuned `88.08%`
- 超过当前 CNN upper-bound random tuned `88.72%`
- ROC-AUC 超过 `90.20%`

## 远程输出

远程日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter03_structure_input
```

本地备份：

```text
result/auto_exploration_iter03_structure_input/remote_summaries/
```
