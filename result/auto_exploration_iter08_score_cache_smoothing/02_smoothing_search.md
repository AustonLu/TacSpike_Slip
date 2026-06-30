# 02 smoothing / hysteresis 搜索记录

## 计划

使用 `scripts/train/evaluate_score_cache_smoothing.py` 在缓存上搜索：

- causal moving average
- EMA
- debounce
- hysteresis

评估对象：

1. Iter06 ignore50 single model
2. Iter06 four-model average/weighted ensemble

## 待记录

| 配置 | 方法 | 参数 | accuracy | balanced acc | 备注 |
|---|---|---|---:|---:|---|
| single ignore50 | 未完成 | - | - | - | score cache 未能生成 |
| four ensemble | 未完成 | - | - | - | score cache 未能生成 |
| weighted ensemble | 未完成 | - | - | - | score cache 未能生成 |

## 结论

本轮没有得到 smoothing/hysteresis 的有效 accuracy 结果。阻塞点不是算法搜索本身，而是完整 validation sequence 的 score cache 生成太慢。
