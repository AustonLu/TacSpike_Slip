# 01. Streaming 训练目标改造

## 目的

把原来的逐窗分类目标改成连续状态检测目标，先验证真正 1ms stateful streaming LIF-SNN 是否能接近或超过 Iter02 的 400ms sliding baseline。

## 代码改动

在 `scripts/train/train_streaming_scnn.py` 中新增：

- `transition_ignore_steps`：onset/offset 附近不计入 CE。
- `warmup_steps` 和 `supervise_tail_steps`：允许前段只用于建立状态，或只监督 tail。
- `smoothness_weight`：惩罚同标签相邻 score 大幅跳变。
- `flip_penalty_weight`：惩罚同标签相邻预测概率频繁变化。
- `valid_*` 指标：只在非 transition / 非 warmup 的有效位置统计。
- `best_metric`：支持按 `valid_accuracy` 等指标保存 best checkpoint。

新增 `scripts/train/evaluate_streaming_sequence_detection.py`，用于对完整 validation sequence 做逐 ms streaming 推理。每条 sequence 开始 reset LIF state，之后每 1ms 输入一个 event bin，输出连续 score，再使用与 Iter02 相同的 MA / EMA / debounce 评价口径。

## 判断

这个改动让训练和评估都从独立 window 转到连续状态序列，但仍保持 LIF 神经元，不使用 IAF。
