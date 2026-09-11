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

### 历史中间 checkpoint

删除清单逐目录应用以下保护规则：保留 `best_model.pth`、`latest_model.pth`、
`early_stopping_best_model.pth`、最高训练轮次、所有已有 `eval_epoch-N.txt` 对应轮次，
以及 tracked 文档、JSON、CSV、release 记录明确选择或引用的轮次。所有 2026-09-11 路径
整段排除。

- 移入回收站：1,078 个未引用 `epoch_N_model.pth`；
- 逻辑大小：1,819,459,290 B，约 1.69 GiB；
- 清理后仍保留 342 个旧 epoch checkpoint，约 615 MB；其中包含非日期审计目录中的
  对齐/中止现场；
- 清理清单中的文件没有活动进程打开，也没有 current bestval 路径。

删除后复核确认：152 个旧日期 checkpoint 目录的最高轮次、180 个已有评测日志对应轮次、
59 个 CSV 上下文目标和 41 个历史选中轮次全部存在；HRMS epoch86 与 PointCenter epoch90
等关键权重也仍通过既有契约校验。

### SwanLab 缓存

- 移入回收站：160 个 `backup.swanlab`，572,234,371 B，约 546 MiB；
- 保留：160 个 `config.yaml`、157 个 `requirements.txt`、138 个
  `swanlab-metadata.json`，以及全部 checkpoint、训练日志和评测日志；
- 其中 39 个来自 09-08～09-10 BC-TPro，21 个来自 09-04 offline 运行。删除不影响模型、
  论文指标或当前训练，但若不从回收站恢复，将不能再从这些本地 backup 精确执行
  `swanlab sync`。

第二阶段共从工作树移除约 2.23 GiB；连同第一阶段，累计约 2.37 GiB。全部通过挂载盘
回收站接口移除，目前仍可恢复。

## 明确保留

- 正在运行的 best-validation 队列、七个新运行目录及其未来 checkpoint/metrics；
- 当前 bestval 依赖的旧工具链和 train64/val16 split；
- 官方对齐 smoke 与因验证口径修正而停止的现场；
- 09-09 upstream、09-10 nongate 的历史指标、日志及其 best/latest/最终或明确引用的
  checkpoint；
- 两个 `release/` 目录中的比赛 checkpoint、源码快照、训练日志与阈值证据；
- 8 月设计、迁移和故障文档。它们合计很小，但承担历史决策和复现证据，不属于缓存。

## 尚未删除的大体积历史数据

深度清理后 ignored `log/` 约 1.7 GB，`.git` 约 879 MB。最大的剩余历史日志
批次包括：

| 目录 | 大约大小 | 当前处理 |
|---|---:|---|
| `log/sem_seg/2026-09-03` | 439 MB | 保留 29 模型 Noise8/clean 的 best/final/已评测证据 |
| `log/sem_seg/2026-08-27` | 329 MB | 保留正式完成的 FeedbackSTS 关键 checkpoint |
| `log/sem_seg/2026-08-20` | 104 MB | 保留结构筛选关键证据 |
| `log/sem_seg/2026-08-13` | 90 MB | 保留 BRTD2/F1-OHEM 关键证据 |
| `log/sem_seg/2026-08-11` | 41 MB | 保留 BRTD/BRTD2 关键证据 |

剩余 checkpoint 均命中保留规则；SwanLab 配置和 metadata 仍在，但大体积 backup 已按
用户确认清理。没有整批删除有效实验目录，也没有对 Git 历史执行不可逆改写。

## 活动训练保护

清理前后均核对后台会话 `bc_tpro_bestval_official_20260911`。本次操作没有触碰
`log/sem_seg/2026-09-11/`、当前 queue、训练源码或其 source snapshot。
