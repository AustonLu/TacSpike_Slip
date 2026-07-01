# 05. 连续状态解码与 Score Cache Ensemble

## 目的

验证是否可以不继续训练模型，而通过更强的连续状态解码或 score ensemble 提升到 90% 以上。

## State Decoder

新增 `scripts/train/evaluate_state_decoder.py`，在已有 score cache 上做二状态 Viterbi 风格解码：

- score 减 threshold 作为状态证据。
- `switch_on_cost` / `switch_off_cost` 控制状态切换代价。
- 每条 sequence 独立解码。

小网格结果：

| 输入 | best method | accuracy | balanced acc | F1 | segment recall | false alarms/min | p95 delay |
|---|---|---:|---:|---:|---:|---:|---:|
| Iter02 ctx400 score cache | `viterbi_ma50_scale1_on0_off32_prior0` | 0.898302 | 0.875060 | 0.813204 | 0.90 | 0.487 | 260.4ms |

## Score Cache Ensemble

新增：

- `scripts/train/evaluate_score_cache_pair.py`
- `scripts/train/evaluate_score_cache_ensemble.py`
- `scripts/train/make_score_cache_subset.py`

结果：

| 输入 | best method | accuracy | balanced acc | F1 | segment recall | false alarms/min | p95 delay |
|---|---|---:|---:|---:|---:|---:|---:|
| Iter02 ctx400 + transition-mix fine-tune | `w1_ma100_debounce_on5_off20` | 0.899023 | 0.877657 | 0.815493 | 0.95 | 1.787 | 231.3ms |
| ctx500 best5 mean + ctx400 + random fine-tune | `subset_0_1_ma100_debounce_on5_off20` | 0.898714 | 0.875382 | 0.813844 | 0.90 | 0.162 | 298.8ms |

## 判断

连续状态解码能减少 false alarm runs 和 switches，但没有显著提高 strict per-window sequence accuracy。跨上下文 ensemble 也没有产生互补突破，最佳仍低于 transition-mix fine-tune 单模型分数。

这说明当前瓶颈不是简单阈值、debounce、HMM 状态代价或 score averaging。
