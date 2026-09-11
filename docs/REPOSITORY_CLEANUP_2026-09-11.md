# 仓库清理记录（2026-09-11）

## 本次已清理

### Tracked 空壳与占位文件

- 删除没有任何函数实现的 `attribution/core.py`、`attribution/utils.py`；
- 删除已有 6 个指标 JSON 后不再需要的
  `experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/metrics/.gitkeep`；
- 移除 `test_BRTD.py` 中指向缺失 attribution 实现的注释 import；
- `test.py --attribution` 继续明确拒绝该缺失功能，避免误报可用。

### 可再生缓存

- 将活动 `2026-09-11` 运行目录以外的 97 个 `__pycache__` 目录移入系统回收站；
- 共清理 265 个左右的 Python bytecode，约 3.1 MB；
- 没有删除任何 `.py` 源码或当前训练的 source snapshot。

### 无有效结果的运行现场

以下目录均未产生可接受的 best checkpoint，且没有 tracked 文档或当前工具引用，已移入
系统回收站：

- `_aborted_launches/2026-09-11_bestval_detach_failure`；
- 2026-08-27 的五次 0–3 epoch FeedbackSTS 中止运行；
- 2026-08-28 的六次未完成 PointCenter 启动（`00-12-23`、`00-15-45`、`00-27-44`、
  `00-29-51`、`00-32-06`、`00-34-04`）；它们均停在第 1 epoch 前后且没有 checkpoint；
- 空的 `_queues/bc_tpro_stage1_exact_logit_2026-09-09`。

上述运行现场加缓存约 144 MB。系统回收站可恢复；清空回收站前不会释放其所在磁盘的
最终空间。

## 明确保留

- 正在运行的 best-validation 队列、七个新运行目录及其未来 checkpoint/metrics；
- 当前 bestval 依赖的旧工具链和 train64/val16 split；
- 官方对齐 smoke 与因验证口径修正而停止的现场；
- 09-09 upstream、09-10 nongate 的历史指标、日志和 checkpoint；
- 两个 `release/` 目录中的比赛 checkpoint、源码快照、训练日志与阈值证据；
- 8 月设计、迁移和故障文档。它们合计很小，但承担历史决策和复现证据，不属于缓存。

## 尚未删除的大体积历史数据

仓库工作区约 4.8 GB，其中 ignored `log/` 约 3.9 GB、`.git` 约 879 MB。最大的历史日志
批次包括：

| 目录 | 大约大小 | 当前处理 |
|---|---:|---|
| `log/sem_seg/2026-09-03` | 883 MB | 保留，包含 29 模型 Noise8/clean 对照和大 checkpoint |
| `log/sem_seg/2026-08-27` | 清理后约 657 MB | 保留正式完成的 FeedbackSTS 运行 |
| `log/sem_seg/2026-08-20` | 437 MB | 保留结构筛选历史 |
| `log/sem_seg/2026-08-13` | 366 MB | 保留 BRTD2/F1-OHEM 历史 |
| `log/sem_seg/2026-08-11` | 285 MB | 保留 BRTD/BRTD2 历史 |

这些目录中的 checkpoint 和 SwanLab backup 仍可能是唯一历史证据。未在没有明确保留规则
和外部备份的情况下整批删除，也没有对 Git 历史执行不可逆改写。

## 活动训练保护

清理前后均核对后台会话 `bc_tpro_bestval_official_20260911`。本次操作没有触碰
`log/sem_seg/2026-09-11/`、当前 queue、训练源码或其 source snapshot。
