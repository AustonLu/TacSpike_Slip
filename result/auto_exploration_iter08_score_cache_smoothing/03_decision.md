# 03 决策

## 决策

本轮没有达到 90%。

完整 sequence score cache 和扩展 ensemble 搜索都受到动态 HDF5 读取/voxelize 瓶颈影响，无法作为当前自动探索的快速内循环。

下一轮应回到训练侧，直接显式处理 transition/onset/offset：

1. transition-distance sample weighting，而不是简单 ignore。
2. transition 附近样本使用较低权重或软标签。
3. onset/offset 辅助目标，若工程量可控。
