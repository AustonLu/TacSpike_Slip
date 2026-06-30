# Auto Exploration Iter06 计划：transition-aware training

日期：2026-06-30

## 背景

Iter05 显示当前最强 time-channel LIF-SCNN 在 100k random validation 的严格全窗口 tuned accuracy 为 `89.45%`，但排除距离 slip/no-slip transition 100 ms 以内的窗口后达到 `90.05%`。这说明模型在稳定标签区已经接近或达到 90%，剩余误差很可能集中在 onset/offset 附近的硬标签噪声。

## 本轮目标

1. 修正 `sampling=random` 下 `--ignore-transition-ms` 不生效的问题。
2. 训练 transition-aware time-channel LIF-SCNN，比较 `ignore_transition_ms=50/100/150`。
3. 补一个 `ignore100 + label_smoothing`，检查软化硬标签是否比单纯过滤更稳。
4. 对每个模型报告：
   - strict 100k random tuned accuracy
   - strict 100k balanced tuned accuracy
   - filtered random 指标（>50 ms、>100 ms、>150 ms）
   - ROC-AUC、precision、recall、specificity

## 实验清单

1. `iter06_time_channel_random_ignore50_v1`
   - time-channel LIF-SCNN
   - 500 ms context / 100 time bins
   - random sampling
   - ignore transition 50 ms during training

2. `iter06_time_channel_random_ignore100_v1`
   - 同上，但训练忽略 transition 100 ms。

3. `iter06_time_channel_random_ignore150_v1`
   - 同上，但训练忽略 transition 150 ms。

4. `iter06_time_channel_random_ignore100_smooth03_v1`
   - `ignore100 + label_smoothing=0.03`。

## 判定标准

主指标仍是 strict 100k random tuned accuracy。若达到或超过 `90%`，本轮视为达到目标。

如果 strict 指标仍低于 90%，但 filtered >100 ms 指标稳定超过 90%，则记录为：稳定标签区已达标，严格全窗口指标主要受 transition/onset/offset 标签边界限制。

## 预期风险

1. 过滤 transition 训练可能提升稳定区，但降低边界泛化，strict 指标不一定上升。
2. 过滤过宽可能改变训练分布，导致 recall 或 specificity 失衡。
3. 如果 transition 窗口在验证集中虽然占比小但错误率极高，单独过滤训练可能不足以修复 strict accuracy。
