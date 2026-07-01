# Sliding Sequence Detection Iter01 总结

日期：2026-07-01

## 本轮完成内容

本轮在 `sliding` 分支建立了连续滑动检测评估脚本 `scripts/train/evaluate_sliding_detection.py`，并完成三项验证：

1. 原始 window size 模型在连续 validation sequence 上的 sliding detection。
2. 500ms best5 SNN ensemble 在连续 validation sequence 上的 sliding detection。
3. 500ms best5 的 debounce threshold 网格复核，并保存 sequence score cache。

## 主要结果

| 实验 | sequences | windows | best method | accuracy | balanced accuracy | F1 | segment recall | false alarms/min | switches |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| original20 | 32 | 671348 | `ma_50_debounce_on2_off10` | 0.7960 | 0.7066 | 0.5891 | 0.9111 | 42.72 | 2185 |
| ctx500 best5 | 16 | 369291 | `ma_50_debounce_on5_off10` | 0.8978 | 0.8719 | 0.8108 | 0.9000 | 0.325 | 32 |
| ctx500 threshgrid | 16 | 369291 | `ma_50_debounce_on5_off10` | 0.8978 | 0.8719 | 0.8108 | 0.9000 | 0.325 | 32 |

## 结论

原始 window size 不适合作为连续滑移检测主路线。它能召回多数 slip 段，但误报 run 和状态抖动太多，sequence-level accuracy 只有 `79.60%`。

500ms 上下文显著改善连续状态稳定性，best5 ensemble 加 causal MA50 和 debounce 后达到 `89.78%`，距离 90% 只差约 0.22 个百分点。更细的 threshold 网格没有继续提升，说明瓶颈不只是阈值或 debounce 参数。

## 下一轮建议

下一轮优先做三件事：

1. 生成完整 validation/test 的 sequence-level score cache 或 500ms voxel cache，避免反复动态构造 500ms window。
2. 扩大 sequence-level 验证，确认 `89.78%` 是否在更多 sequence 和固定阈值下稳定。
3. 进入 sequence-aware 训练或 streaming LIF-SNN：训练目标要显式处理 onset/offset、false positive window 和连续状态稳定性。

本轮未严格达到 90%，但已经把问题从“模型是否有可分性”推进到“如何让长上下文 SNN 在连续序列上稳定泛化并减少 hard sequence 错误”。
