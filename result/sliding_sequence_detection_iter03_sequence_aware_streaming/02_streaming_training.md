# 02. 1ms Streaming LIF-SNN 训练记录

## 配置

日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter03
```

训练了 3 个 streaming 变体：

| run | segment steps | 监督策略 | transition ignore | best valid acc | valid ROC-AUC | valid F1 |
|---|---:|---|---:|---:|---:|---:|
| `stream_t400_all_ignore25_smooth_v1` | 400 | all valid steps | 25 | 0.821363 | 0.806451 | 0.615523 |
| `stream_t400_tail200_ignore25_smooth_v1` | 400 | tail 200ms | 25 | 0.821836 | 0.806797 | 0.623149 |
| `stream_t512_tail256_ignore50_smooth_v1` | 512 | tail 256ms | 50 | 0.826544 | 0.812497 | 0.633653 |

## 观察

相比历史 naive streaming，本轮 sequence-aware loss 有提升，但训练期有效窗口准确率仍只有约 82.1-82.7%。这明显低于 Iter02 400ms sliding-window 模型的训练/sequence 表现。

主要问题是 1ms 输入的信息密度太低，模型必须完全依赖 LIF state 长时间累计，而当前轻量 SCNN 没有学到足够可靠的长期状态。
