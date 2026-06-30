# Auto Exploration Iter08 计划：score cache + sequence smoothing

日期：2026-06-30

## 背景

Iter06 最佳 single model 在 strict 100k random 上为 `89.631%`，weighted ensemble 为 `89.953%`，距离 90% 很近。Iter05/06 都显示 transition/onset/offset 附近窗口是主要误差来源。Iter07 证明单纯扩大 time-channel LIF-SCNN 容量无效。

此前直接做完整 sequence smoothing 时太慢，本轮改成两步：

1. 对完整 validation sequence 顺序缓存 score/label。
2. 在缓存上快速搜索 smoothing/hysteresis/debounce。

## 本轮目标

在完整 validation sequence 上验证轻量 streaming 后处理是否可以把 strict sequence-level window accuracy 推到 `>=90%`。

## 实验清单

1. 对 Iter06 最佳单模型缓存完整 validation score：
   - `iter06_time_channel_random_ignore50_v1`

2. 对 Iter06 互补模型缓存完整 validation score：
   - `iter06_time_channel_random_ignore100_v1`
   - `iter06_time_channel_random_ignore150_v1`
   - `iter06_time_channel_random_ignore100_smooth03_v1`

3. 基于缓存搜索：
   - raw single model
   - Iter06 four ensemble
   - weighted ensemble
   - causal moving average
   - EMA
   - debounce / hysteresis

## 判定

主判定仍看 accuracy 是否 >=90%，但需要注明评估口径：

- 若完整 validation sequence 达到 90%，下一步需要用 100k random 同分布口径或 test split 固定参数复核。
- 若完整 sequence 也未达 90，则 sequence smoothing 不能解决当前 strict window 目标。

## 风险

1. smoothing 可能提高稳定区但牺牲 onset 响应，产生检测延迟。
2. 在 validation 上搜索阈值和 hysteresis 参数存在过拟合风险。
3. 完整 sequence 口径与 100k random 口径不完全等价，需要分开报告。
