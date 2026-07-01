# 02 Context Sweep 序列评估

日期：2026-07-01

## 配置

对 01 的每个 checkpoint 做 validation sequence-level sliding detection。

共同配置：

- validation split，固定 seed，16 条完整 sequence
- score transform：raw
- smoothing：causal MA `3,5,10,20,50`
- EMA：`0.1,0.2,0.4`
- debounce：on `2,3,5`，off `2,3,5,10`
- 最终采用 cache-based fast evaluation，debounce threshold grid = 1

说明：第一次评估使用 `debounce-threshold-grid=101`，推理完成并写出 score cache 后，后处理阶段过慢。由于 Iter01 已证明细阈值网格不能提升 500ms best5 结果，本轮改用已生成的 score cache 做轻量后处理评估。

## 结果

| context | best method | accuracy | balanced accuracy | F1 | segment recall | missed | false alarms/min | FP window frac | p95 delay | switches |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100ms | `ma_50_debounce_on2_off10` | 0.87495 | 0.82827 | 0.75743 | 0.900 | 2 | 13.973 | 0.0710 | 195.7ms | 606 |
| 200ms | `ma_50_debounce_on5_off2` | 0.88912 | 0.85932 | 0.79374 | 0.900 | 2 | 7.799 | 0.0763 | 150.3ms | 662 |
| 300ms | `ma_50_debounce_on5_off10` | 0.89496 | 0.86893 | 0.80593 | 0.900 | 2 | 2.112 | 0.0749 | 221.1ms | 116 |
| 400ms | `ma_50_debounce_on5_off10` | 0.89714 | 0.87346 | 0.81099 | 0.900 | 2 | 1.625 | 0.0754 | 264.4ms | 96 |
| 500ms | `ema_0.1_debounce_on3_off10` | 0.89733 | 0.86891 | 0.80849 | 0.900 | 2 | 2.925 | 0.0698 | 356.3ms | 138 |

## 判断

100ms 和 200ms 明显不够，主要问题是 false alarm runs/min 和 prediction switches 仍然太高。

300ms 已经接近 500ms，但 accuracy 低约 0.24 个百分点，false alarm/min 高于 400ms。

400ms 是本轮最好的折中：

- accuracy `0.89714`，仅比 500ms 的 `0.89733` 低 0.019 个百分点。
- balanced accuracy 和 F1 反而高于 500ms。
- false alarm runs/min 为 `1.625`，低于 500ms 的 `2.925`。
- p95 delay `264.4ms`，也短于 500ms 的 `356.3ms`。

因此，如果部署成本和延迟要考虑，400ms 比 500ms 更合理。若要进一步压缩，300ms 是可测试候选，但需要额外训练策略补上最后约 0.2-0.3 个百分点。
