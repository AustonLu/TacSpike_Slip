# Auto Exploration Iter07 计划：capacity and training strength

日期：2026-06-30

## 背景

Iter06 最佳 weighted SNN ensemble 达到 `89.953%`，但仍未满足 strict 100k random tuned accuracy `>=90%`。最佳单模型是 `time_channel_scnn + random + ignore50`，100k random tuned accuracy 为 `89.631%`，ROC-AUC 为 `91.828%`。

Iter06 的结果说明：

1. transition-aware training 有效，但简单过滤已经接近收益上限。
2. 后处理和 weighted ensemble 只能逼近 90%，但没有越过。
3. `ignore50` 在训练末期仍有提升迹象，模型容量或训练强度可能仍不足。

## 本轮目标

在保持 LIF-SNN 和 time-channel 输入表示的前提下，提高 strict 100k random tuned accuracy 到 `>=90%`。

## 实验清单

1. `iter07_time_channel_w48_h384_ignore50_v1`
   - `width=48`
   - `hidden=384`
   - `ignore_transition_ms=50`
   - 更大容量，保持 10 epoch。

2. `iter07_time_channel_w64_h512_ignore50_v1`
   - `width=64`
   - `hidden=512`
   - `ignore_transition_ms=50`
   - 检查容量上限，但需要注意显存/速度。

3. `iter07_time_channel_w48_h384_ignore50_long_v1`
   - `width=48`
   - `hidden=384`
   - 15 epoch
   - 每 epoch 90k train samples
   - 检查更充分训练是否推过 90%。

4. `iter07_time_channel_w48_h384_ignore50_lr5e4_v1`
   - `width=48`
   - `hidden=384`
   - `lr=5e-4`
   - 检查较小 learning rate 是否改善后期泛化。

## 评估

每个模型训练后运行：

- 100k random validation，按 accuracy 搜索阈值。
- 100k balanced validation，作为类别平衡诊断。
- transition bucket evaluation，报告 strict 与 >100 ms 指标。

若单模型达到 90%，停止本轮训练扩展并记录为当前主模型候选。

## 风险

1. 扩大模型可能过拟合，balanced 指标可能下降。
2. `width=64` 可能因为 batch size 需要下降而影响训练稳定性。
3. 如果容量提升仍不能过 90%，说明 strict 指标主要需要边界建模，而不是普通容量扩张。
