# 阶段 1 数据读取和 Sanity Check

本文档记录阶段 1 已实现的数据读取接口、检查脚本和远程验证命令。

## 本地代码

- `src/tacspike/data/h5_dataset.py`
  - `TacSpikeH5Dataset`：从 `manifest_sequences.csv` 建立 split 级全局窗口索引。
  - `select_events`：按 `windows/t_start` 和 `windows/t_end` 从 `events/t,x,y,p` 选取窗口事件。
  - `voxelize_events`：动态生成 `[T, C, H, W]` 体素，默认双极性 `[20, 2, 128, 128]`。
  - `spatial_sum_pool`：非重叠空间 sum pooling，默认 4x4，得到 `[20, 2, 32, 32]`。

- `scripts/validate/stage1_check_dataset.py`
  - 轻量检查 train/val/test 的 sequence 数、window 数、随机窗口 shape、label、event_count 对齐。

- `scripts/validate/stage1_inspect_sample.py`
  - 抽取指定或随机窗口，输出 JSON 摘要，并生成 event raster / full map / pooled map 可视化。

## 远程数据路径

已在 `miller` 上验证的数据 release 路径：

```bash
/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0
```

远程项目副本：

```bash
/lamport/makkapakka/jiajunlu/projects/TacSpike_Slip_v2
```

远程 Python 环境：

```bash
/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python
```

## 已执行验证

在 `miller` 上执行：

```bash
cd /lamport/makkapakka/jiajunlu/projects/TacSpike_Slip
DATA=/lamport/makkapakka/jiajunlu/TacSpike_Dataset/releases/hf_v1.0.0_ready/tacspike-slip-detection-1khz-v1.0.0
/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python scripts/validate/stage1_check_dataset.py \
  --data-root "$DATA" \
  --splits train val test \
  --samples-per-split 3 \
  --output-json /lamport/makkapakka/jiajunlu/logs/tacspike_sanity/check_dataset_stage1.json
```

结果摘要：

- train：1091 sequences，20293980 windows。
- val：234 sequences，4173170 windows。
- test：234 sequences，4485259 windows。
- 随机检查样本的 `event_count` 与 `windows/event_count` 一致，`mismatched_event_count=0`。
- 输出体素 shape 为 `[20, 2, 128, 128]`，4x4 pooling 后 shape 为 `[20, 2, 32, 32]`。

生成样本图：

```bash
/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python scripts/validate/stage1_inspect_sample.py \
  --data-root "$DATA" \
  --split train \
  --target-label 1 \
  --seed 7 \
  --output-dir /lamport/makkapakka/jiajunlu/logs/tacspike_sanity/plots

/capsule/home/jiajunlu/miniconda3/envs/a100-torch/bin/python scripts/validate/stage1_inspect_sample.py \
  --data-root "$DATA" \
  --split train \
  --target-label 0 \
  --seed 11 \
  --output-dir /lamport/makkapakka/jiajunlu/logs/tacspike_sanity/plots
```

生成文件：

```bash
/lamport/makkapakka/jiajunlu/logs/tacspike_sanity/check_dataset_stage1.json
/lamport/makkapakka/jiajunlu/logs/tacspike_sanity/plots/train_sphere_batch_2_102_w22357_label1.png
/lamport/makkapakka/jiajunlu/logs/tacspike_sanity/plots/train_sphere_batch_2_25_w4865_label0.png
```

两个 PNG 均为有效图片，尺寸为 `1760 x 1280`。

## 注意事项

- `miller` 的 SSH alias 偶发解析失败时，可以显式使用：

```bash
ssh -J fics jiajunlu@192.168.68.198
```

- FICS 登录节点当前 `/lamport/makkapakka/jiajunlu` 挂载异常，和既有判断一致；后续如走 FICS 作业，应固定到 `makkapakka01` 或 `makkapakka03`。
- 在 `/lamport` 上避免对脚本路径使用 `Path.resolve()`，它可能触发坏挂载或符号链接解析阻塞。当前脚本已使用 `Path.absolute()`。
