# 05 weighted SNN ensemble 搜索

## 目的

Iter06 单模型最好结果为 `ignore50` 的 100k random tuned accuracy `89.631%`，四模型/五模型平均 ensemble 最高到约 `89.915%`。本步骤检查最后不足 `0.1%` 是否来自分数融合校准，而不是模型识别能力。

## 方法

新增脚本 `scripts/train/evaluate_weighted_ensemble_search.py`：

1. 在同一批 100k random validation 窗口上缓存每个 SNN checkpoint 的 slip-vs-no-slip score。
2. 比较四种 score 标定方式：
   - raw
   - zscore
   - minmax
   - rank_centered
3. 搜索 one-hot、均匀子集、随机 Dirichlet 权重。
4. 对每个候选权重搜索 accuracy 最优阈值。

## 注意

这是 validation 上的权重和阈值联合搜索，只能作为诊断或上界探索。若超过 90%，仍应固定权重后用独立 test split 或另一组随机 validation seed 复核。

## 待记录

| ensemble | transform | tuned accuracy | ROC-AUC | 备注 |
|---|---|---:|---:|---|
| Iter06 four mean | raw mean | 89.897% | 91.561% | 普通均值 ensemble |
| Iter04 + Iter06 five mean | raw mean | 89.915% | 91.610% | 普通均值 ensemble，略优 |
| Iter06 four weighted | zscore | 89.949% | 91.738% | 搜索权重 `[0.3869, 0.2702, 0.2598, 0.0830]` |
| Iter04 + Iter06 five weighted | raw | 89.953% | 91.764% | 本轮最佳，但仍未到 90% |

## 结论

weighted ensemble 最高达到 `89.953%`，距离 90% 只差 `0.047` 个百分点，但仍未满足目标。由于该结果已经使用 validation 权重和阈值搜索，不能继续把它当作最终泛化指标放大解释。下一轮应提升模型本身或显式处理 transition/onset，而不是继续扩大 validation 后处理搜索。
