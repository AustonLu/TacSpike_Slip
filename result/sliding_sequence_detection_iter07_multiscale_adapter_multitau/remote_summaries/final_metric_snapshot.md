# Iter07 Final Metric Snapshot

远程日志根目录：

```text
/lamport/makkapakka/jiajunlu/logs/tacspike_sliding_sequence_detection_iter07
```

本文件记录 2026-07-06 远端最终摘要指标。由于远端 SSH 跳板多次出现 banner exchange timeout，本轮没有把完整 JSON/log 小包稳定拉回本地；完整原始文件仍保留在远端日志目录。

## Training Best Validation

| run | latest epoch | best epoch | valid accuracy | valid balanced accuracy | valid F1 | valid ROC-AUC | valid PR-AUC | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multiscale_l384_sqrt | 8 | 5 | 77.40% | 77.45% | 77.88% | 83.92% | 84.47% | 159,362 |
| multiscale_l384_mean | 8 | 8 | 72.85% | 72.99% | 72.78% | 77.36% | 78.20% | 159,362 |
| multitau_l384_ignore30 | 8 | 7 | 70.13% | 70.28% | 70.01% | 75.85% | 77.67% | 114,918 |

## Final Sequence Evaluation

| run | best method | accuracy | balanced accuracy | F1 | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| eval_multiscale_l384_sqrt | ma_200_debounce_on2_off50 | 88.57% | 82.42% | 76.46% | 85.48% | 69.16% |
| eval_multiscale_l384_mean | ma_200_debounce_on8_off50 | 86.55% | 81.44% | 73.75% | 77.45% | 70.39% |
| eval_multitau_l384_ignore30 | ma_200_debounce_on2_off50 | 87.70% | 81.89% | 75.16% | 82.06% | 69.33% |

## Score Adapter Results

| run | adapter valid accuracy | valid balanced accuracy | valid F1 | valid ROC-AUC | best method | sequence accuracy | sequence balanced accuracy | sequence F1 | precision | recall |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| adapter_iter04_best5 | 95.27% | 69.44% | 53.28% | 81.29% | adapter_ma_150_debounce_on2_off50 | 95.57% | 67.98% | 52.89% | 99.89% | 35.97% |
| adapter_iter04_ctx400 | 80.76% | 64.45% | 44.96% | 90.04% | adapter_raw_debounce_on3_off50 | 86.06% | 77.23% | 69.13% | 85.20% | 58.16% |
| adapter_iter04_seqft | 77.40% | 58.09% | 28.19% | 87.31% | adapter_raw_debounce_on5_off50 | 84.72% | 73.79% | 63.80% | 87.54% | 50.19% |

