# 01 标签与可达上限审计

## 目的

确认 strict per-window sequence accuracy 的 `95%` 目标是否被 transition 标签边界天然限制。

## 已完成本地 smoke audit

使用 Iter03 拉回的 `iter02_ctx400_score_cache.npz` 对 16 条 validation sequence 做标签审计。

关键结果：

- validation windows：`369291`
- positive fraction：`0.26836`
- label transitions：`29`
- slip segments：`20`
- transition 周边窗口占比：
  - `±50ms`：`0.747%`
  - `±100ms`：`1.396%`
  - `±500ms`：`5.909%`

完美检测器如果只是整体延迟，strict accuracy 仍然很高：

| delay | strict accuracy |
|---:|---:|
| 50ms | 99.625% |
| 100ms | 99.305% |
| 300ms | 98.204% |
| 500ms | 97.211% |

## 初步判断

`95%` strict accuracy 理论上不是被 transition 标签边界硬性挡住。当前 `89.9%` 更可能来自 score extractor 对若干 sequence 的大段误判、漏检或提前误报，而不是单纯 onset/offset 几十毫秒误差。

## 远程执行

远程 audit 作业最初因运行目录缺少 `evaluate_state_decoder.py` 依赖失败；已补同步依赖并重启。后续结果拉回后更新本节。
