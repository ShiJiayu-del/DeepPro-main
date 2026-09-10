# DeepPro / CSIG2026 文档索引

本目录同时保存当前研发说明和历史研究记录。新实验应优先服从当前代码中的安全检查，
再参考本索引标为“当前”的文档；历史文档中的旧服务器路径、预训练流程和 GPU 方案只
用于结果审计，不应直接复制执行。

## 当前主线

比赛提交阶段已经结束。最终 scratch-only Hybrid-RMS 提交 ID `907655` 得分
**91.30**；保留的 checkpoint、源码快照、环境、阈值扫描和历史复现脚本见
[`release/2026-08-29_final_submission_score91.30_scratch/`](../release/2026-08-29_final_submission_score91.30_scratch/README.md)。
比赛结束清理时已删除最终 ZIP、轨迹 TXT、提交校验与相关哈希清单，历史复现脚本不再是
可直接执行的完整提交入口。

当前分支为 `paper-experiments-2026-09-10`。研发主线是在
`NUDT-MIRSDT-Noise8.0_FJY` 上与 TinaLRJ/DeepPro 官方提交 `8fa1a68` 对齐的
BC-TPro 单 seed 实验。B1/C0/C1/C2 与 NG1/NG2/NG3 的 seed47 训练和 internal-val16
评测均已完成；当前综合结论和全部产物见
[无门控实验结果](../experiments/bc_tpro_nongate_noise8_seed47_2026-09-10/RESULTS.md)。
此前的三 seed 预注册矩阵和两套 29 项全模型对比保留为历史筛选证据。

| 文档 | 内容 | 状态 |
|---|---|---|
| [实验改进与优化交接](EXPERIMENT_OPTIMIZATION_HANDOFF_2026-09-10.md) | 当前状态覆盖、完整网络/损失演进、历史实验和官方来源 | 当前首选接手入口 |
| [重要文件与目录说明](IMPORTANT_FILES_GUIDE.md) | 训练、测试、模型、模块、损失、实验、论文指标、发布集与日志目录的逐项说明 | 当前仓库导航入口 |
| [DeepPro 官方仓库对齐](DEEPPRO_OFFICIAL_ALIGNMENT_2026-09-09.md) | 官方提交、模型/loader/训练参数对齐项及有意保留的科研约束 | 当前协议总览 |
| [BC-TPro 无门控结果](../experiments/bc_tpro_nongate_noise8_seed47_2026-09-10/RESULTS.md) | B1/C0/C1/C2/NG1/NG2/NG3 的 seed47 三指标与 Pareto 对比 | 当前结果入口；[Excel](../experiments/bc_tpro_nongate_noise8_seed47_2026-09-10/NG_EXPERIMENT_RESULTS_2026-09-10.xlsx) |
| [Noise8 BC-TPro upstream](../experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/README.md) | 上游语义对齐的 B1/C0/C1/C2 单 seed 对照 | 已完成；[Excel](../experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/BC_TPRO_STAGE1_SEED47_RESULTS_2026-09-10.xlsx) |
| [NUDT-MIRSDT 全模型实验](../experiments/nudt_mirsdt_all_models_2026-09-01/README.md) | 统一数据、训练协议、29 项模型清单、运行与汇总方式 | 历史筛选证据 |
| [模型演进历史快照](MODEL_EVOLUTION_ARCHITECTURE_AND_LOSS_2026-08-26.md) | DeepPro 到 BRTD/Raw-APMD/Hybrid-RMS/FeedbackSTS 前期的历史结构与损失 | 2026-08-26 历史快照；当前状态以新交接为准 |
| [F1 最大化跨领域研究与 PointCenter 决策](F1_MAXIMIZATION_RESEARCH_2026-08-27.md) | SatVideoIRSDT 阶段的证据和候选决策 | 历史研发依据 |
| [Scratch-only 模型改进](SCRATCH_MODEL_IMPROVEMENT_2026-08-25.md) | 非零投影、bandpass、detail 三候选的设计与验收 | 历史实验依据 |
| [网站结果分析](WEBSITE_RESULTS_ANALYSIS_2026-08-25.md) | 网站结果及 scratch-only 决策 | 历史结果依据 |
| [迁移验收](MIGRATION_ACCEPTANCE_2026-08-25.md) | 新服务器路径、环境、数据和运行检查 | 当前运行基线 |

当前不可绕过的规则：

- 新训练只能随机初始化；`base_ckpt`、`spatial_ckpt`、`st_ckpt` 均被拒绝；
- 只允许物理 GPU 0、1、2，GPU3 不进入训练或后处理任务；
- 当前 Noise8 对比只使用 seed47；单 seed 结果不提供训练随机性标准差或显著性结论；
- NUDT/Noise8 检测联合考虑 `Pd@0.5` 越高、`Fa@0.5` 越低和官方 27 阈值 Pd-Fa AUC 越高；使用 Pareto 关系，不采用 AUC 优先或未登记的加权分数；
- 后续结构优先采用无门控的加法残差、固定差分或滤波思路；
- 从 2026-09-09 起，新训练与评测不生成或要求 SHA256、MD5 等文件哈希；历史发布记录中的哈希只作既有审计证据；
- SwanLab 云端异常不能使已经完成的本地训练失效。

## 迁移与完整交接

| 文档 | 内容 | 使用方式 |
|---|---|---|
| [完整迁移交接](MIGRATION_HANDOFF_2026-08-24.md) | 数据、模型、实验、环境和旧服务器完整记录 | 历史事实与故障排查；部分路径和预训练建议已过时 |
| [环境 YAML](environment_sjyPID_2026-08-24.yml) | Conda 环境导出 | 重建环境 |
| [Conda 显式包清单](conda_explicit_sjyPID_2026-08-24.txt) | 精确 Conda 包版本 | 环境审计 |
| [pip freeze](pip_freeze_sjyPID_2026-08-24.txt) | Python 包版本 | 环境审计 |

## 历史结构研究

| 文档 | 内容 | 当前结论 |
|---|---|---|
| [BRTD2 research](brtd2_research.md) | 深层语义适配、3/5/9 帧时域和门控设计 | no-gate 证据影响了 Raw-APMD；旧预训练命令不得执行 |
| [Raw-APMD](raw_apmd.md) | 原始外观、多尺度一/二阶运动与局部对比 | 已成为当前结构母体 |
| [结构优化分析](structure_optimization_2026-08-20.md) | RMS、Channel-RMS、去趋势、多尺度对比 | 促成 Hybrid-RMS；scratch 下扩展模块未成为默认 |
| [第二轮结构候选](structure_round2.md) | 对齐、双向传播、低频净化等八候选 | 保留为候选档案，不是当前执行队列 |
| [实验分析](experiment_analysis_2026-08-20.md) | Raw-APMD 与早期结构的实验复盘 | 历史对照 |

## 对话与决策归档

- [2026-08-12 BRTD 交接](BRTD_CONVERSATION_HANDOFF_2026-08-12.md)
- [2026-08-20 BRTD 交接](BRTD_CONVERSATION_HANDOFF_2026-08-20.md)
- [早期对话交接](CONVERSATION_HANDOFF.md)

这些文件用于追溯当时的假设和决策，不代表当前默认配置。

## 实验发布集

`release/2026-08-29_final_submission_score91.30_scratch/` 是最终提交的历史审计目录。
它保留随机初始化 epoch-86 Hybrid-RMS 权重、网站 91.30 分记录、自适应阈值证据和
已停用脚本；最终 ZIP、轨迹 TXT 和提交校验已删除，不能再称为完整可执行复现包。

`release/2026-08-22_pretrained_vs_scratch_seed47/` 是八组 Hybrid-RMS
pretrained/scratch 对照的历史目录。比赛 ZIP、提交 TXT 和校验清单已删除；剩余 README
及训练证据只用于追溯，当前 scratch-only 训练不得加载其中权重。

## 代码入口

| 任务 | 文件 |
|---|---|
| 训练 | `train.py` |
| 推理/概率导出 | `test.py` |
| 当前模型 | `networks/models/DeepPro-Plus_BCTPro.py` |
| 当前结构适配器 | `networks/layers/bc_tpro_adapter.py` |
| 无门控实验调度 | `tools/run_bc_tpro_nongate.py` |
| 无门控结果核验与汇总 | `tools/analyze_bc_tpro_nongate.py` |
| 当前实验清单 | `experiments/bc_tpro_nongate_noise8_seed47_2026-09-10/manifest.tsv` |
| 历史 29 模型清单 | `experiments/nudt_mirsdt_all_models_2026-09-01/manifest.tsv` |
| 损失函数 | `networks/losses/segmentation_losses.py` |
| 运行环境和 GPU 白名单 | `tools/project_runtime_env.sh` |
| 历史工具导航 | `tools/README.md` |
