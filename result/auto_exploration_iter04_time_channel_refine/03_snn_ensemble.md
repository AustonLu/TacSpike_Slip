# 03. SNN-only Ensemble

状态：完成

## 目的

检查 time-channel SNN、random-sampling SNN、wide3 sequential SNN 和历史 wide2 SNN 是否存在足够互补性，能否通过 SNN-only score ensemble 推过 `90%` random tuned accuracy。

## Ensemble 配置

评估口径：

- split：`val`
- sampling：`random`
- samples：`100000`
- threshold：按 accuracy 搜索

## 结果

| Ensemble | 成员 | Tuned acc | Balanced acc at tuned | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| `ensemble_timechannel3` | Iter03 time-channel + Iter04 random + Iter04 random-distill | 89.50% | 85.38% | 91.24% | 84.85% |
| `ensemble_timechannel3_wide3` | 上述 + Iter03 wide3 | 89.41% | 85.26% | 91.45% | 84.91% |
| `ensemble_timechannel_wide_all` | 上述 + history wide2 distill/ignore50 | 89.11% | 84.83% | 91.47% | 84.67% |

## 观察

Ensemble 可以小幅提高 ROC-AUC，但 tuned accuracy 最高只到 `89.50%`。加入 wide3 和历史 wide2 后 ROC-AUC 上升但 accuracy 下降，说明这些模型的 score ranking 有互补，但最佳阈值附近的错误没有被有效修正。

## 结论

SNN-only ensemble 仍未达到 `90%`。当前最强结果是 `ensemble_timechannel3` 的 `89.50%`，距离目标还差 `0.50` 个百分点。
