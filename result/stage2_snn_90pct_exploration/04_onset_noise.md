# Onset 标签噪声处理实验

状态：完成

目的：减少 slip/no-slip transition 附近模糊硬标签对训练的负面影响。

## 实现

数据索引缓存中新增每个窗口到最近 label transition 的距离：

```text
slip_transition_distance
no_slip_transition_distance
```

训练采样新增参数：

```text
--ignore-transition-ms 50
```

该参数只过滤训练集采样，不改变 validation/test 评估。

## 结果

| Run | 参数量 | Best epoch | 20k balanced val acc | ROC-AUC | 备注 |
|---|---:|---:|---:|---:|---|
| `ctx500_tb100_wide2_scnn_quick_v1` | 282,688 | 4 | 79.14% | 86.42% | 无过滤 |
| `ctx500_tb100_wide2_scnn_ignore50_quick_v1` | 282,688 | 4 | 81.10% | 87.32% | 过滤 transition 50 ms |
| `ctx500_tb100_wide2_scnn_ignore50_v1` | 282,688 | 8 | 83.61% | 90.15% | full run |

## 100k 复评

| Run | Sampling | Default acc | Tuned acc | Balanced acc | ROC-AUC |
|---|---|---:|---:|---:|---:|
| `ctx500_tb100_wide2_scnn_ignore50_v1` | random | 87.48% | 87.68% | 83.19% default | 89.57% |
| `ctx500_tb100_wide2_scnn_ignore50_v1` | balanced | 83.22% | 83.22% | 83.22% default | 89.46% |

## 结论

transition 过滤在 quick run 中收益明显，从 `79.14%` 提升到 `81.10%`。完整训练后它达到 `83.61%`，略低于 `wide2 distill` 的 `84.07%`，但 100k random tuned accuracy 为本轮最高 `87.68%`。

这说明 onset/transition 标签噪声确实影响训练，但不是全部瓶颈。过滤 transition 还可能牺牲 onset 附近的即时检测能力，后续如果正式采用，应单独评估 onset delay 和 transition bucket accuracy。
