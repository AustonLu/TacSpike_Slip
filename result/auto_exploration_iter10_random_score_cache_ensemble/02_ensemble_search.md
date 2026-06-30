# 02 cache-based ensemble 搜索记录

日期：2026-06-30

## 搜索设置

输入：

- `iter10_core_snn_seed123_random100k/caches/*.npz`
- 9 个 SNN checkpoint
- `val random 100k`, seed `123`

搜索：

- score transform：`raw`、`zscore`、`minmax`、`rank_centered`
- one-hot、均匀子集、Dirichlet 随机权重
- 按 accuracy 搜索最优阈值
- 两阶段实现：先用快速 accuracy 排序，再对 top candidates 计算完整 ROC-AUC/PR-AUC

## 最佳结果

| rank | tuned accuracy | ROC-AUC | PR-AUC | transform | 成员 |
|---:|---:|---:|---:|---|---|
| 1 | 90.009% | 91.793% | 86.088% | `zscore` | 5 模型均值 |
| 2 | 90.008% | 91.747% | 未列出 | `raw` | 同一 5 模型均值 |
| 3 | 90.006% | 91.793% | 未列出 | `minmax` | 同一 5 模型均值 |
| 4 | 90.005% | 91.847% | 未列出 | `raw` | 9 模型稀疏随机权重 |

最佳 5 模型均值：

| 模型 | 权重 |
|---|---:|
| `iter06_time_channel_random_ignore100_v1` | 0.2 |
| `iter06_time_channel_random_ignore50_v1` | 0.2 |
| `iter07_time_channel_w48_h384_ignore50_v1` | 0.2 |
| `iter09_tw_near20_mid100_v1` | 0.2 |
| `iter09_tw_near50_mid100_smooth02_v1` | 0.2 |

最佳结果的详细指标：

| 指标 | 数值 |
|---|---:|
| tuned accuracy | 90.009% |
| balanced accuracy | 86.268% |
| precision | 86.078% |
| recall | 77.548% |
| specificity | 94.988% |
| F1 | 81.591% |
| ROC-AUC | 91.793% |
| PR-AUC | 86.088% |
| threshold | 0.4734 |

## 与历史最佳对比

| 方法 | random 100k tuned accuracy | ROC-AUC |
|---|---:|---:|
| Iter06 weighted ensemble | 89.953% | 91.764% |
| Iter10 best5 zscore mean | 90.009% | 91.793% |

提升幅度：

- accuracy：+0.056 个百分点
- ROC-AUC：+0.029 个百分点

## 观察

1. 本轮首次在主指标 `val random 100k tuned accuracy` 上超过 90%。
2. 最佳组合是简单 5 模型均值，不依赖复杂随机权重；raw、zscore、minmax 三种 transform 的结果都在 90.006%-90.009%。
3. 提升幅度很小，说明 ensemble/校准已经接近当前 score 空间上界。
4. 最优成员来自 Iter06、Iter07 和 Iter09，说明 transition-aware training、capacity scaling 和 transition-weighted loss 虽然单独不强，但存在少量互补。
