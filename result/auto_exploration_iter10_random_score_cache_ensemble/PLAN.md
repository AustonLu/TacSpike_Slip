# Auto Exploration Iter10 计划：固定随机窗口 score cache 与扩展 SNN ensemble

日期：2026-06-30

分支：`auto-snn-accuracy-exploration`

## 背景

当前历史最佳结果是 Iter06 的 weighted SNN ensemble：

- 100k random tuned accuracy：`89.953%`
- ROC-AUC：`91.764%`

它距离 `90%` 只差 `0.047` 个百分点。Iter08 已经发现继续做 ensemble/后处理的主要瓶颈不是算法，而是每次搜索都重复读取 HDF5 并动态 voxelize，导致评估过慢。Iter09 的 transition-weighted loss 没有提升单模型，但这些 checkpoint 可能仍与 Iter04/Iter06 形成少量互补。

本轮改为先在同一套固定 `val random 100k` indices 上缓存多个 LIF-SNN checkpoint 的 score/label，然后在 numpy 中快速搜索权重、score transform、阈值和轻量校准。

## 成功标准

主标准：

- fixed `val random 100k` tuned accuracy `>=90%`

若达到 90%，需要额外复核：

- 另一组 random seed 的 100k validation
- 100k balanced validation
- 说明该结果是 validation ensemble/search 上界，正式实验仍需 test split 或固定权重后复核

## 探索项

### 01 random-window score cache

实现脚本：

- `scripts/train/cache_random_window_scores.py`

目标：

- 对固定 sampling seed 生成同一套 100k validation indices。
- 每个 checkpoint 输出一个 `.npz`，包含 `scores`、`labels`、`ordered_indices`、`original_indices`、`checkpoint_name`。
- 确保不同 checkpoint 的 labels 与 ordered indices 完全一致。

优先纳入的 SNN checkpoint：

- Iter04 random/distill/time-channel 系列
- Iter06 transition-aware 系列
- Iter07 capacity 系列中未明显崩坏的模型
- Iter09 transition-weighted 系列
- Stage2 中较强的 `ctx500_tb100_*` LIF-SCNN/DeepSCNN 对照

### 02 cache-based weighted ensemble search

实现脚本：

- `scripts/train/search_score_cache_ensemble.py`

搜索内容：

- score transform：`raw`、`zscore`、`minmax`、`rank_centered`
- one-hot、均匀子集、Dirichlet 随机权重
- 对 top checkpoint 组合做更多随机搜索
- accuracy 最优阈值

输出：

- `ensemble_search_seed123_random100k.json`
- top-k 权重、成员、transform、accuracy、ROC-AUC、PR-AUC

### 03 达标复核

若 02 中结果 `>=90%`：

- 使用同一组权重在 `seed=456` 的 `val random 100k` 上重建 score cache 并复核。
- 使用同一组权重在 `val balanced 100k` 上重建 score cache 并复核。

若 02 中仍低于 90%：

- 记录最优值和距 90% 差距。
- 判断 ensemble/校准上界是否已耗尽。
- 下一轮转向显式 label 边界建模、sequence-level 状态模型或 dataset/label audit。

## 远端路径

- 远程项目：`/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2`
- 数据集：`/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0`
- 日志根目录：`/lamport/makkapakka/jiajunlu/logs/tacspike_auto_iter10_random_score_cache_ensemble`

## 本轮输出

- `01_random_score_cache.md`
- `02_ensemble_search.md`
- `03_recheck.md`
- `SUMMARY.md`
- `remote_summaries/`
