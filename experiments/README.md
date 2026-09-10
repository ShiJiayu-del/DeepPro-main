# 实验索引

此目录只保存版本化实验配置、结果表和分析文档。训练日志、权重、运行状态及失败证据
统一存放于 `../log/sem_seg/`。批次目录名称保持不变，方便已有代码和文档继续引用。

| 日期 | 数据集 | 训练状态 | 结果 | 分析 | 配置及路径 |
|---|---|---|---|---|---|
| 2026-09-01 | NUDT-MIRSDT | 29/29 完成 | [RESULTS](nudt_mirsdt_all_models_2026-09-01/RESULTS.md) | [ANALYSIS](nudt_mirsdt_all_models_2026-09-01/ANALYSIS.md) | [README](nudt_mirsdt_all_models_2026-09-01/README.md)、[manifest](nudt_mirsdt_all_models_2026-09-01/manifest.tsv) |
| 2026-09-03 | Noise8.0_FJY | 29/29 完成 | [RESULTS](nudt_mirsdt_noise8_fjy_all_models_2026-09-03/RESULTS.md) | [ANALYSIS](nudt_mirsdt_noise8_fjy_all_models_2026-09-03/ANALYSIS.md) | [README](nudt_mirsdt_noise8_fjy_all_models_2026-09-03/README.md)、[manifest](nudt_mirsdt_noise8_fjy_all_models_2026-09-03/manifest.tsv) |
| 2026-09-08 | NUDT-MIRSDT Clean -> Noise8 zero-shot | 12/12 完成 | 24 份内部验证指标 | [ANALYSIS](bc_tpro_stage1_2026-09-08/ANALYSIS.md) | [范围更正](bc_tpro_stage1_2026-09-08/SCOPE_CORRECTION_2026-09-09.md) |
| 2026-09-09 | Noise8.0_FJY 同域 BC-TPro（旧 modernized） | 12/12 完成 | [NOISE8_ANALYSIS](bc_tpro_stage1_noise8_2026-09-09/NOISE8_ANALYSIS.md) | 探索结果，不作为官方复现 | [README](bc_tpro_stage1_noise8_2026-09-09/README.md)、[预注册](bc_tpro_stage1_noise8_2026-09-09/EXPERIMENT_PLAN.md) |
| 2026-09-09 | Noise8.0_FJY BC-TPro（TinaLRJ/DeepPro `8fa1a68` 对齐） | 单 seed 修订 B1/C0/C1/C2 4/4 完成；原三 seed 矩阵 6/12 后停止 | [Excel](bc_tpro_stage1_noise8_upstream_2026-09-09/BC_TPRO_STAGE1_SEED47_RESULTS_2026-09-10.xlsx) | C1 综合权衡最好；不采用 AUC 优先 | [README](bc_tpro_stage1_noise8_upstream_2026-09-09/README.md)、[指标修订](bc_tpro_stage1_noise8_upstream_2026-09-09/OFFICIAL_METRIC_AMENDMENT_2026-09-10.md) |
| 2026-09-10 | Noise8.0_FJY BC-TPro 无门控消融 | NG1/NG2/NG3 3/3 完成；复用 B1/C0/C1/C2 seed47 | [RESULTS](bc_tpro_nongate_noise8_seed47_2026-09-10/RESULTS.md)、[Excel](bc_tpro_nongate_noise8_seed47_2026-09-10/NG_EXPERIMENT_RESULTS_2026-09-10.xlsx) | 三个新分支均未综合超过 C1 | [README](bc_tpro_nongate_noise8_seed47_2026-09-10/README.md)、[manifest](bc_tpro_nongate_noise8_seed47_2026-09-10/manifest.tsv)、[CSV](bc_tpro_nongate_noise8_seed47_2026-09-10/results.csv) |

## 文件职责

- `README.md`：数据与训练协议、复现命令。
- `manifest.tsv`：模型清单；`log_dir` 是相对 `log/sem_seg/` 的实际实验路径。
- `results.csv` / `RESULTS.md`：机器可读及可读的训练筛选指标。
- `ANALYSIS.md`：结构对比和稳定性分析。
- `paper_metrics.csv` / `PAPER_METRICS.md`：论文指标汇总（如已生成）。

当前 BC-TPro 单 seed 探索联合考虑 Pd 越高、Fa 越低、AUC27 越高，并报告 Pareto
关系。不同指标发生权衡时不使用单项优先级或未登记的加权分数选出唯一冠军。

2026-09-08 已补齐两套数据各29项论文指标评测，共58/58；FeedbackSTS采用FP32复核结果。
总分析见 [论文指标详细对比](PAPER_COMPARISON_2026-09-08.md)。
分表：[干净数据](nudt_mirsdt_all_models_2026-09-01/PAPER_METRICS.md)、
[Noise8](nudt_mirsdt_noise8_fjy_all_models_2026-09-03/PAPER_METRICS.md)。

## 运行产物位置

`log/sem_seg/_queues/<批次名>/` 保存 `status/`、`launcher_logs/`、`failed_attempts/`、
`paper_metrics/{raw,status,logs}/`、锁和 `PIPELINE_COMPLETE`（如存在）。
评测脚本将原始 JSON 和运行日志写入这里，再将汇总表输出到当前实验目录。

旧 `log/nudt_*/`、`log/archive/` 与 `log/archived_untracked_tools_2026-08-31/`
兼容入口已移除。历史归档为 `log/sem_seg/_archive/<日期>/<用途>/`。
原始日志中的历史路径保持原文，当前访问位置以 manifest 和此索引为准。

2026-09-08 再次检查：本目录没有重复权重或训练日志，无需删除现有结果与配置文件。
调度日志直接位于 `log/sem_seg/_pipeline/`，顶层旧链接已清除。
