# 01 random-window score cache 记录

日期：2026-06-30

## 目的

Iter08 发现完整 sequence cache 和扩展 ensemble 搜索太慢，瓶颈在重复 HDF5 读取和动态 voxelize。本轮改为固定一套 `val random 100k` 窗口 indices，每个 checkpoint 只推理一次并缓存 score/label，之后所有 ensemble 搜索都在 numpy 上完成。

## 新增脚本

- `scripts/train/cache_random_window_scores.py`
- `scripts/train/search_score_cache_ensemble.py`
- `scripts/train/run_iter10_random_score_cache_ensemble.sh`
- `scripts/train/launch_iter10_random_score_cache_ensemble.sh`
- `scripts/train/run_iter10_recheck_best_ensemble.sh`

## 缓存设置

核心搜索 run：

- run_id：`iter10_core_snn_seed123_random100k`
- split：`val`
- sampling：`random`
- seed：`123`
- samples：`100000`
- batch size：`96`
- 每个 checkpoint 生成 `.npz`，包含 `scores`、`labels`、`ordered_indices`、`original_indices` 和 checkpoint metadata。

## 纳入 checkpoint

| checkpoint | 来源 | 单模型参考 |
|---|---|---:|
| `iter04_time_channel_thr1_random_v1` | Iter04 | 89.447% random 100k |
| `iter04_time_channel_thr1_random_distill_v1` | Iter04 | 88.975% random 100k |
| `iter06_time_channel_random_ignore50_v1` | Iter06 | 89.631% random 100k |
| `iter06_time_channel_random_ignore100_v1` | Iter06 | 89.447% random 100k |
| `iter06_time_channel_random_ignore150_v1` | Iter06 | 89.553% random 100k |
| `iter06_time_channel_random_ignore100_smooth03_v1` | Iter06 | 89.463% random 100k |
| `iter07_time_channel_w48_h384_ignore50_v1` | Iter07 | 89.447% random 100k |
| `iter09_tw_near20_mid100_v1` | Iter09 | 89.545% random 100k |
| `iter09_tw_near50_mid100_smooth02_v1` | Iter09 | 89.465% random 100k |

## 运行情况

每个 100k score cache 约 40-48 秒完成。9 个 checkpoint 全部缓存成功，随后 ensemble 搜索不再访问 HDF5。

## 工程结论

固定 random-window score cache 解决了 Iter08 的主要工程瓶颈。后续只要 checkpoint 已缓存，ensemble、score transform、阈值搜索可以快速复用同一套 labels/indices，适合作为自动探索内循环。
