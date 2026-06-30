# SNN 精度自动化探索 Workflow

日期：2026-06-29

适用范围：后续所有为提升 TacSpike slip detection SNN 精度而进行的自动化探索迭代。

## 固定原则

1. 每一轮探索都必须在 `result/` 下新建独立目录。
2. 每一轮探索开始前，必须先在该目录写 `PLAN.md`，明确假设、实验项、成功标准和预计输出。
3. 每一轮探索中，`PLAN.md` 的每一个探索项都必须有单独 markdown 记录过程和结果。
4. 每一轮探索结束时，必须写 `SUMMARY.md`，总结结果、失败或成功原因，并提出下一轮探索建议。
5. 每一轮探索结束后，必须提交 git 并 push 到 GitHub，保证可回溯。
6. 除非用户另行要求，探索应在当前自动化探索分支上继续迭代，不回退用户已有改动。

## 单轮目录结构

建议命名：

```text
result/auto_exploration_iterNN_<short_topic>/
```

每轮至少包含：

```text
PLAN.md
01_<experiment_name>.md
02_<experiment_name>.md
...
SUMMARY.md
remote_summaries/
```

其中：

- `PLAN.md`：实验前写，记录本轮假设、实验矩阵、成功标准和执行脚本。
- `NN_*.md`：每个探索项单独记录配置、命令、训练过程、指标和判断。
- `SUMMARY.md`：本轮结束写，记录最佳结果、是否达到目标、主要原因和下一轮建议。
- `remote_summaries/`：保存远程训练/评估产生的 summary 和 evaluation JSON。

## 每轮执行步骤

1. 确认工作树状态：

```bash
git status --short --branch
```

2. 新建结果目录并写 `PLAN.md`。
3. 如需改代码，保持改动范围只服务本轮探索，并同步到远程训练目录。
4. 启动远程训练或评估，记录 run id、关键参数、远程路径和 checkpoint 路径。
5. 将远程结果 JSON 拉回本轮 `remote_summaries/`。
6. 为每个探索项写单独 markdown。
7. 写 `SUMMARY.md`，至少包含：

- 本轮最佳指标。
- 与上一轮最佳 SNN/CNN upper-bound 的差距。
- 是否达到 `90%` accuracy。
- 主要失败或成功原因。
- 下一轮优先建议。

8. 本地校验：

```bash
python -m compileall src scripts
git diff --check
```

9. 提交并推送：

```bash
git add <本轮相关文件>
git commit -m "<iteration summary>"
git push
```

## 指标口径

每轮结果必须明确区分以下口径：

- 训练期 validation：通常是 balanced sampling 的 5k 或 20k。
- 100k random validation：近似自然类别分布。
- 100k balanced validation：平衡 slip/no-slip 后的分类能力。
- default threshold：模型原始阈值。
- tuned threshold：在验证集上按 accuracy 或其他指标搜索阈值。
- ROC-AUC / PR-AUC：反映分数可分性，不等同于固定阈值 accuracy。

## 当前长期目标

优先目标：

- 100k random validation accuracy >= `90%`
- 或 100k balanced validation accuracy >= `90%`

若仍未达到，应记录是否至少缩小与 CNN upper-bound 的差距，并说明下一轮为什么选择新的探索方向。
