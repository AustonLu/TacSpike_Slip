# 03. 阈值与类别先验分析

状态：完成

## 目的

分析当前模型是否只是阈值或类别先验不匹配。如果 tuned threshold 能大幅提升 accuracy，说明模型校准是主要问题；如果提升很小，说明分数可分性本身不足。

## 观察

### Ensemble 2

| Sampling | Default acc | Tuned acc | Threshold | Default balanced acc | Tuned balanced acc |
|---|---:|---:|---:|---:|---:|
| random | 87.51% | 87.76% | 0.462 | 83.57% | 82.71% |
| balanced | 83.54% | 83.54% | 0.005 | 83.54% | 83.54% |

### Ensemble 3

| Sampling | Default acc | Tuned acc | Threshold | Default balanced acc | Tuned balanced acc |
|---|---:|---:|---:|---:|---:|
| random | 87.84% | 88.08% | 0.453 | 83.82% | 83.04% |
| balanced | 83.80% | 84.14% | -0.295 | 83.80% | 84.14% |

## 判断

1. random sampling 下 tuned threshold 大约为 `0.45`，明显偏向减少 false positive。这符合自然分布中 no-slip 占比较高的情况。
2. balanced sampling 下 threshold 接近 `0` 或略为负，说明模型在平衡分布下的默认阈值校准并不差。
3. tuned threshold 的提升有限：
   - random：最多约 `+0.24%`
   - balanced：最多约 `+0.34%`
4. 因此，主要瓶颈不是阈值选择，而是 score ranking / feature separability。

## 结论

类别先验校准可以带来小幅收益，但不能把 accuracy 推到 `90%`。后续训练中可以保留针对部署分布的阈值校准，但不应把它作为主要提精度方向。
