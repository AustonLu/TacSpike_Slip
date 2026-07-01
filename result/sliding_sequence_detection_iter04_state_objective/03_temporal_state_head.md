# 03 轻量因果序列状态头

## 目的

将独立 window score 转换为连续 slip state 检测：冻结已有 SNN score extractor，只训练一个轻量 causal temporal state head。输入为一个或多个 SNN score 序列及其 MA/EMA 动态特征，输出每 1ms 的 slip state。

## 实现

新增脚本：

- `scripts/train/train_score_state_head.py`
- `scripts/train/evaluate_state_postprocess_search.py`
- `scripts/train/audit_sequence_state_labels.py`

状态头结构：

- causal Conv1d stack；
- 每层只做左 padding，避免未来信息泄露；
- GELU + dropout；
- 1x1 Conv 输出每 1ms logit；
- 训练时可忽略 transition 周边窗口，并加入 smoothness loss。

## 实验配置

### 双路状态头

输入：

- Iter02 `ctx400` score；
- Iter03 sequence fine-tuned `ctx400` score；
- 每路附加 raw、MA、EMA 低维动态特征。

主要参数：

```text
chunk_len=2048
hidden_dim=64
layers=5
kernel_size=9
dropout=0.1
transition_ignore=50
smoothness_weight=0.001
```

结果：

| 指标 | 数值 |
|---|---:|
| strict accuracy | 90.041% |
| balanced accuracy | 87.110% |
| F1 | 81.322% |
| precision | 81.867% |
| recall | 80.784% |
| specificity | 93.437% |
| segment recall | 90.0% |
| false alarm runs/min | 0.650 |
| onset delay p95 | 435.7 ms |
| missed slip segments | 2 / 20 |

最佳后处理：

```text
ma100_debounce_on5_off50
```

### 三路状态头

额外加入历史 `ctx500 best5` ensemble score 特征。训练端使用 `subset_best5_train16_fast/score_cache.npz`，验证端使用已校验对齐的完整 `ctx500_best5_val16_threshgrid_score_cache.npz`。对齐检查结果：

```text
labels_equal=True
indices_equal=True
offsets_equal=True
```

主要参数：

```text
hidden_dim=48
layers=4
kernel_size=9
dropout=0.2
extra_feature_specs=raw,ma:50,ma:100,ema:0.05
```

结果：

| 指标 | 数值 |
|---|---:|
| strict accuracy | 90.198% |
| balanced accuracy | 88.503% |
| F1 | 82.288% |
| precision | 79.880% |
| recall | 84.845% |
| specificity | 92.161% |
| segment recall | 95.0% |
| false alarm runs/min | 0.812 |
| onset delay p95 | 237.5 ms |
| missed slip segments | 1 / 20 |

最佳后处理：

```text
ma100_debounce_on5_off50
```

## 结论

轻量状态头可以把 strict accuracy 从约 `89.9%` 提到 `90.2%`，并改善 segment recall 和 onset delay。但提升幅度很小，远不到 `95%`。这说明瓶颈不主要在手工状态机，而在上游 SNN score 的可分性和跨序列泛化能力。

三路融合比双路略好，尤其将 segment recall 提到 `95%`，但代价是 false positive 增加，strict accuracy 仍停在 `90.2%`。
