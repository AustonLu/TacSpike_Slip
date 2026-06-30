# 04 extended ensemble 搜索记录

## 计划

由于 Iter06 weighted SNN ensemble 已达到 `89.953%`，本轮尝试扩大已有 SNN checkpoint 池并增加随机权重搜索次数：

1. `weighted_existing5_trials5000`
   - Iter04 random + Iter06 四个 transition-aware SNN
   - 5000 trials

2. `weighted_extended7_trials2000`
   - Iter03 time-channel + Iter04 random/random-distill + Iter06 四个 transition-aware SNN
   - 2000 trials

## 实际结果

两个任务均启动成功，但评估过程超过 25-55 分钟仍未生成 JSON。原因是每个 checkpoint 都要重新从 HDF5 动态 voxelize 同一批 100k random windows，且多模型 ensemble 会重复此过程，I/O 和 CPU voxelize 成为主要瓶颈。

为避免继续占用远程资源，已停止这两个任务。

## 结论

扩大 ensemble 搜索本身不是不可行，但当前实现没有 score cache，评估效率太低，不适合作为自动探索内循环。若以后继续做 ensemble，应先实现 random-window score cache，而不是反复让每个模型重新读取和 voxelize 数据。
