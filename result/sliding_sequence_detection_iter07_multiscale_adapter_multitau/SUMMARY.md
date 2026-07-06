# Iter07 总结：Multi-scale / Adapter / Multi-tau 探索

## 结论摘要

本轮按照 `docs/plan/snn_slip_detection_experiment_plan.md` 的流程，完成了三条探索：

1. multi-scale causal feature stream；
2. score cache + temporal/state adapter；
3. multi-timescale SNN。

最终有效 SNN 最好结果为：

```text
run: multiscale_l384_sqrt
sequence accuracy: 88.57%
sequence balanced accuracy: 82.42%
sequence F1: 76.46%
precision: 85.48%
recall: 69.16%
```

该结果比 raw 1 ms streaming SNN 有改善，但仍未达到 90%，更未达到 95% 综合目标。

## 最终结果表

| 类别 | run | best postprocess | accuracy | balanced accuracy | F1 | precision | recall |
|---|---|---|---:|---:|---:|---:|---:|
| SNN | multiscale_l384_sqrt | ma_200 + debounce on2/off50 | 88.57% | 82.42% | 76.46% | 85.48% | 69.16% |
| SNN | multiscale_l384_mean | ma_200 + debounce on8/off50 | 86.55% | 81.44% | 73.75% | 77.45% | 70.39% |
| SNN | multitau_l384_ignore30 | ma_200 + debounce on2/off50 | 87.70% | 81.89% | 75.16% | 82.06% | 69.33% |
| Adapter | adapter_iter04_best5 | ma_150 + debounce on2/off50 | 95.57% | 67.98% | 52.89% | 99.89% | 35.97% |
| Adapter | adapter_iter04_ctx400 | raw + debounce on3/off50 | 86.06% | 77.23% | 69.13% | 85.20% | 58.16% |
| Adapter | adapter_iter04_seqft | raw + debounce on5/off50 | 84.72% | 73.79% | 63.80% | 87.54% | 50.19% |

## 关键判断

- **多尺度因果输入是本轮最有效方向。** `sqrt` 归一化显著优于 `mean`，说明长窗口证据需要保留足够幅度。
- **multi-tau 有帮助但不是主瓶颈。** 它在 sequence-level 上接近 multi-scale，但训练端判别力更弱。
- **score adapter 的高 accuracy 可能是虚高。** `adapter_iter04_best5` accuracy 达到 95.57%，但 recall 只有 35.97%，不能作为滑移检测达标结果。
- **当前主要问题是 recall 不足。** 最佳 SNN recall 约 69%，说明模型仍倾向保守触发，滑移段漏检较多。

## 未达标原因

本轮没有达到 90%/95%，主要原因包括：

1. 训练目标仍是逐点 BCE，和连续状态检测目标不完全一致。
2. slip/no-slip transition 附近可能存在标签对齐噪声，硬标签会惩罚合理的提前/延迟检测。
3. smoothing/debounce 提高稳定性时会牺牲 recall。
4. raw event 数据稀疏，多尺度输入虽有帮助，但模型还没有充分利用连续状态结构。

## 下一步建议

下一轮不建议继续只调后处理或单纯加宽模型。建议集中做：

1. **把训练目标改成连续状态检测目标**
   - segment-level / state-aware loss；
   - transition soft label 或 ignore band；
   - recall-sensitive loss。

2. **继续以 `multiscale_l384_sqrt` 为主线**
   - 不再回到纯 raw 1 ms input；
   - 增加轻量 causal temporal conv 或 state head。

3. **重构评价指标**
   - 将 window/state accuracy、balanced accuracy、event recall、delay、false alarm 分开报告；
   - 不把单一 accuracy 作为达标标准。

4. **生成多尺度 feature cache**
   - 当前动态构造多尺度特征训练较慢；
   - 后续如果继续沿用该路线，应缓存 1/20/50/100/200/400 ms causal features。

## 远端输出位置

远程日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter07
```

核心 run：

```text
multiscale_l384_sqrt
multiscale_l384_mean
multitau_l384_ignore30
eval_multiscale_l384_sqrt
eval_multiscale_l384_mean
eval_multitau_l384_ignore30
adapter_iter04_best5
adapter_iter04_ctx400
adapter_iter04_seqft
```
