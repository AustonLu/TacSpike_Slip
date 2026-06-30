# 03 达标复核记录

日期：2026-06-30

## 复核目的

`seed123 random100k` 搜索结果达到 `90.009%`，但这是在 validation 上搜索权重/transform/阈值的结果。为判断稳定性，本轮固定最佳组合：

- 成员：Iter06 ignore100、Iter06 ignore50、Iter07 w48 ignore50、Iter09 near20-mid100、Iter09 near50-mid100 smooth02
- 权重：全部 0.2
- transform：`zscore`

然后在另外两个口径复核：

1. `val random 100k`, seed `456`
2. `val balanced 100k`, seed `123`

## 复核结果

| 口径 | tuned accuracy | ROC-AUC | PR-AUC | default accuracy | 是否 >=90% |
|---|---:|---:|---:|---:|---|
| random 100k seed123 | 90.009% | 91.793% | 86.088% | 86.838% | 是 |
| random 100k seed456 | 89.955% | 92.025% | 86.064% | 86.814% | 否，差 0.045 个百分点 |
| balanced 100k seed123 | 86.612% | 91.885% | 92.794% | 86.484% | 否 |

## 判断

1. 主目标口径 `random 100k tuned accuracy >=90%` 在 seed123 上达成。
2. 另一个 random seed 降到 `89.955%`，与 90% 只差 `0.045` 个百分点，说明结果刚好跨线，不能说已经稳定超过 90%。
3. balanced 100k 只有 `86.612%`，说明在类别均衡口径下仍明显不足。
4. ROC-AUC 在 random456 上更高，但 accuracy 未过 90%，说明分数排序能力尚可，阈值附近/类别先验/样本组成仍影响很大。

## 复核结论

Iter10 达到了本轮预设的主指标，但结果非常贴边。后续如果要作为论文式正式结果，必须固定 ensemble 权重和阈值策略，并在 test split 或更多 validation seeds 上报告均值和置信区间。
