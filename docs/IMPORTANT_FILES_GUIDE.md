# DeepPro 重要文件与目录说明

本文档面向论文作者、代码审查者和复现实验人员，说明当前仓库中会影响模型定义、训练、
测试、指标计算、实验复现和结果追溯的重要文件。

历史比赛推理产物已清理；2026-09-08 又移除了 Python 字节码缓存及重复终端日志。
模型权重、SwanLab 记录、训练日志和源码快照继续保留，用于实验追溯。
日志产物不逐个列名，而在“日志与产物目录”中说明结构；重要源码和配置逐项说明。
本次核验见 [训练结果分析](TRAINING_REVIEW_2026-09-08.md)，删除范围见
[清理清单](CLEANUP_2026-09-08.md)。

## 1. 建议阅读顺序

首次接手项目时建议按以下顺序阅读：

1. `README.md`：当前研发状态、结果入口和原始 DeepPro 项目说明。
2. `docs/VALIDATION_SCHEDULE_2026-09-11.md`：新 NUDT BC-TPro 实验的 best checkpoint 规则与 final80 暂停边界。
3. `experiments/bc_tpro_stage1_noise8_bestval_seed47_2026-09-11/README.md`：当前七结构重跑协议和状态。
4. `paper/DEEPPRO_PLUS_METRIC_ALIGNMENT.md`：当前论文工作的指标口径和可比性边界。
5. `experiments/<实验名>/README.md` 与 `manifest.tsv`：固定协议和实际运行清单。
6. `train.py`、`test.py`：当前权威训练与测试入口。
7. `networks/models/`、`networks/layers/`、`networks/losses/`：模型、模块和损失实现。
8. `tools/run_bc_tpro_bestval.py`：当前逐 epoch 验证和 best checkpoint 重跑入口。

## 2. 根目录文件

| 文件 | 作用 | 当前使用建议 |
|---|---|---|
| `.gitignore` | 定义不进入 Git 的大体积日志、checkpoint、缓存和临时文件。 | 整理仓库或提交前必须保留。 |
| `README.md` | 上游 DeepPro 项目总览，包含论文简介、数据下载、基础训练测试命令和原论文结果。 | 用于了解 baseline；部分比赛说明和旧命令不是当前论文实验协议。 |
| `train.py` | 当前统一训练入口。负责参数解析、随机种子、单卡/DDP、数据加载、模型动态导入、损失构造、AMP、梯度累积、逐 epoch 验证、best checkpoint、SwanLab 和安全续训。 | 新 BC-TPro 以 internal-val16 官方逐窗口 micro pixel IoU@0.5 选择同一 run 的 `best_model.pth`；当前代码拒绝非空预训练 checkpoint。 |
| `test.py` | 当前统一推理和评测入口。按序列拼接重叠窗口，计算目标级 Pd、像素级 Fa 与官方 27 阈值 Pd-Fa AUC；旧 Pixel 指标只作兼容诊断。支持 AMP、分块推理、可视化、质心 TXT 和机器可读 JSON。 | best-validation 重跑不传 `--epoch`，默认加载 `best_model.pth`；不同架构只按 Pd/Fa/AUC 综合比较。 |
| `runtime_utils.py` | 训练和测试共享的运行工具，包括 GPU 参数解析、DDP 上下文、原子 checkpoint 写入、checkpoint 加载和进程环境处理。 | 不直接执行，由入口脚本导入。 |
| `sequence_utils.py` | 时序窗口辅助模块。`SequenceAccumulator` 将有重叠的窗口预测合并成完整序列，并处理有效帧范围。 | 需要改推理拼接规则时检查。 |
| `ShootingRules.py` | 论文目标级评测规则。对标签连通域统计目标，在目标邻域判定命中，并统计保护区域外的误警像素；支持一次扫描多个阈值。 | Pd/Fa 的核心定义，改动会直接改变论文结果。 |
| `write_results.py` | 汇总各序列的目标计数与误警计数，划分 NUDT 低/高 SNR 序列，计算 Pd、Fa、AUC，并按论文单位输出。 | 与 `ShootingRules.py` 共同构成论文指标实现。 |
| `train_BRTD.py` | 早期 BRTD 专用训练入口。功能和当前 `train.py` 有重叠，但配置与维护状态较旧。 | 历史复现使用；新实验不应优先选它。 |
| `test_BRTD.py` | 早期 BRTD 专用测试入口。包含旧版推理和 Pd/Fa 评测流程。 | 历史对照使用；论文最终数据统一由 `test.py` 生成。 |
| `validate_BRTD.py` | 对 BRTD checkpoint 和序列输出进行验证，检查模型装载和验证指标。 | 用于结构或旧 checkpoint 排错，不是最终主评测入口。 |
| `pretrain_tdcsta_branches.py` | 分别预训练 TDCSTA 的分支后再组合的历史工具。 | 与当前“所有新实验从零训练”的规则冲突，仅保留作历史审计，不执行。 |

## 3. 数据加载：`data_utils/`

| 文件 | 作用 | 关键行为 |
|---|---|---|
| `data_utils/TrainDataLoader.py` | 多帧红外序列训练数据集。 | 读取 `train.txt`，构造40帧窗口，随机裁剪与几何增强，处理序列尾部零填充；PointCenter 模型可额外生成中心热图。 |
| `data_utils/TestDataLoader.py` | 验证/测试序列数据集。 | 读取 `test.txt`，以10%重叠构造窗口，返回原图、mask、目标质心图和窗口帧范围；支持 NUDT、Noise8、SatVideo 等布局。 |
| `data_utils/loader_utils.py` | 数据布局的共享校验函数和常量。 | 校验图像与标注是否逐帧对应，发现数据集划分，保存 SatVideoIRSDT_v1 的训练均值和标准差。 |
| `data_utils/__pycache__/` | Python 自动生成的字节码缓存。 | 可删除，不属于源码或复现证据。 |

当前论文实验使用的数据不存放在 Git 仓库内：

- 干净集：`../datasets/NUDT-MIRSDT`
- 强噪声集：`../datasets/NUDT-MIRSDT-Noise8.0_FJY`

两个数据集使用相同的 `train.txt/test.txt` 划分，便于做逐模型抗噪对比。

## 4. 模型定义：`networks/models/`

每个模型文件都公开名为 `detector` 的 PyTorch 模型类，`train.py/test.py` 按文件名动态
导入。训练目录会复制一份当时的模型文件，确保后续修改主分支代码不会改变旧实验。

| 文件 | 模型说明 |
|---|---|
| `DeepPro.py` | 论文基础 DeepPro。主要使用时间差分卷积、TD-ResBlock 和 TPro，从长时序 temporal profile 中提取目标信号。 |
| `DeepPro-Plus.py` | 当前论文 baseline。将少量空间-时间差分卷积加入 DeepPro，并精简为单层级结构；支持按图像行分块运行 TPro 以降低推理显存。 |
| `DeepPro-Plus_BCTPro.py` | 当前 Noise8 研究模型。在官方 DeepPro-Plus 的 TPro 后加入可选加法残差证据分支，覆盖 B1/C0/C1/C2 与 NG1/NG2/NG3。 |
| `DeepPro_TDCR.py` | 在 DeepPro 前端加入 TDCR 多时间步差分分支的变体，用于比较显式时序差分的作用。 |
| `DeepPro-Plus_forMovingScenes.py` | 面向运动场景的 DeepPro-Plus 历史变体，加入空间降采样/运动适配路径。 |
| `DeepPro-Plus_TDCSTA.py` | TDC、时空自注意力和 DeepPro-Plus 的组合模型。模型较大，使用 batch 1 + 梯度累积，并在序列间释放验证缓存。 |
| `DeepPro-Plus_BRTD.py` | 第一代 BRTD，在 DeepPro-Plus 特征中加入环形背景参考和自适应时域差分。 |
| `DeepPro-Plus_BRTD2.py` | 第二代稳定 BRTD，保留显式外观通路，使用多时间范围的深度卷积和保守可靠性门控。 |
| `DeepPro-Plus_BRTD3.py` | 当前结构消融母模型。通过 `structure_variant` 选择20种结构适配器，并保持统一的 DeepPro-Plus 主干。 |
| `DeepPro-Plus_BRTD3_PointCenter.py` | Hybrid-RMS 主干加渐进中心过滤和 gated restoration 的模型；输出辅助中心信息，必须配合 `center_consistency_f1` 损失。 |
| `DeepPro-FeedbackSTS.py` | 从零训练的双向时空语义反馈网络，包含金字塔可变形对齐、稀疏语义传播和反馈迭代。FP16 数值不稳定时使用 FP32。 |
| `__init__.py` | Python 包标记。 |

### BRTD3 的20个结构变体

这些变体在 `networks/layers/structure_adapters.py` 的 `STRUCTURE_VARIANTS` 和
`build_structure_adapter()` 中注册：

| 变体 | 主要变化 |
|---|---|
| `raw_apmd` | 从原始帧提取 appearance，并融合时间步1/2/4的一阶、二阶 motion。 |
| `raw_apmd_rms` | Raw-APMD 使用 frame-wise RMS normalization。 |
| `raw_apmd_channel_rms` | 使用按通道 RMS normalization。 |
| `raw_apmd_hybrid_rms` | 融合 frame RMS 与 channel RMS，是当前主要结构之一。 |
| `raw_apmd_motion_detrend` | 在 RMS Raw-APMD 中去除慢变化运动趋势。 |
| `raw_apmd_multiscale_contrast` | 加入3/5/7尺度局部对比特征。 |
| `raw_apmd_hybrid_rms_scratch_init` | Hybrid-RMS 的输出投影采用小幅非零初始化，使适配器从第一步就能获得梯度。 |
| `raw_apmd_hybrid_rms_scratch_bandpass` | 在 scratch-init 上加入短/长时域 band-pass 响应。 |
| `raw_apmd_hybrid_rms_scratch_detail` | 在 scratch-init 上加入主干细节编码。 |
| `raw_apmd_hybrid_rms_motion_detrend` | Hybrid-RMS 加运动去趋势。 |
| `raw_apmd_hybrid_rms_multiscale_contrast` | Hybrid-RMS 加多尺度局部对比，是干净 NUDT 的 F1 初筛最佳结构。 |
| `raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | 同时加入运动去趋势和多尺度对比。 |
| `second_order` | 仅使用二阶运动差分的历史 BRTD3 默认结构。 |
| `lfp_shallow` | 在浅层进行低频净化。 |
| `lfp_deep` | 在深层进行低频净化。 |
| `global_align` | 进行全局运动对齐后再提取时序特征。 |
| `local_align` | 使用局部对齐估计空间位移。 |
| `multiscale_head` | 在 TPro 后加入多尺度上下文预测头。 |
| `bidirectional` | 使用正向/反向传播单元融合双向时序信息。 |
| `tdc_dual_stream` | 外观与 TDC 差分双流融合。 |

## 5. 网络模块：`networks/layers/`

| 文件 | 作用 |
|---|---|
| `basic.py` | 原始 DeepPro 基础层：时间差分卷积 `TDifferenceConv`、空间差分卷积 `SDifferenceConv`、TD/STD residual blocks。 |
| `TPro.py` | Temporal Probing 核心模块。对每个空间位置的长时间序列执行多头 temporal correlation，是 DeepPro 系列的核心。 |
| `bc_tpro_adapter.py` | 当前 BC-TPro 证据模块：时间控制、中心多尺度、中心/环形参照及三个无门控固定差分或平滑变体。 |
| `tdc.py` | 多时间步 Temporal Difference Convolution 与 `TDCR` 组合。 |
| `tdcsta.py` | 3D window attention、窗口拆分/还原、自注意力和交叉注意力，实现 TDCSTA 模型的注意力部分。 |
| `brtd_adapter.py` | 第一代 BRTD 模块：环形背景参考、AdaptiveTDCR 和残差适配器。 |
| `brtd_v2_adapter.py` | 第二代稳定适配器：多膨胀时域分支、外观保留和可靠性门控。 |
| `structure_adapters.py` | BRTD3 的全部20种结构适配器、有效帧 mask、各类 RMS normalization、对齐、双向传播和构造工厂。修改消融结构时最重要的文件。 |
| `feedback_sts.py` | FeedbackSTS 的金字塔可变形对齐、稀疏语义传播、反馈 block 和完整 backbone。 |
| `__init__.py` | Python 包标记。 |

## 6. 损失函数：`networks/losses/`

| 文件 | 作用 |
|---|---|
| `segmentation_losses.py` | 当前统一损失注册表、参数检查和16种损失实现。所有损失接收 `[B,T,H,W]` logits 与标签。 |
| `HAM_loss.py` | 来自 DTUM 思路的单帧 hard-aware mining 历史损失。 |
| `HAM_loss_MultiFrame.py` | HAM 的多帧版本，保留给旧模型文件兼容。 |
| `HPM_loss_MultiFrame.py` | 多帧 hard positive mining 历史实现。 |
| `__init__.py` | 向训练入口导出 `LOSS_NAMES`、说明、命名函数和 loss factory。 |

`segmentation_losses.py` 中的重要损失包括：

- `soft_iou`：原论文 DeepPro Soft-IoU，也是严格论文训练协议应使用的损失。
- `frame_soft_iou`、`dice`、`bce_dice`：逐帧区域重叠类目标。
- `bce`、`focal`、`hard_focal`：像素分类及困难背景挖掘。
- `tversky`、`focal_tversky`、`tversky_hard_focal`：显式调节漏检与误警权重。
- `lovasz`：IoU 的排序代理损失。
- `sls_iou`、`tda_sls`：尺度/位置敏感和局部目标感知损失。
- `stc_f1`：中心响应与时序一致性组合。
- `f1_calibrated_ohem`：当前29模型初筛使用的 Tversky + Dice + 自适应困难负样本损失。
- `center_consistency_f1`：PointCenter 专用损失，同时约束分割、中心热图和过滤前后一致性。

## 7. 实验定义：`experiments/`

`experiments/` 只保存可版本控制的实验协议和结果摘要；大体积权重与日志位于 `log/`。

### `experiments/README.md`

实验目录总索引，说明实验定义与运行产物分离原则。其状态文字可能落后于实时训练，具体
总入口为 [实验索引](../experiments/README.md)。完成情况以子目录的 `RESULTS.md`
和 `log/sem_seg/_queues/<批次>/status/` 为准。

### 当前 Noise8 BC-TPro 实验

- `bc_tpro_stage1_noise8_bestval_seed47_2026-09-11/`：当前七结构 seed47 重跑。每个 epoch
  完整验证 internal-val16，按官方逐窗口 micro pixel IoU@0.5 最大化选择
  `best_model.pth`（overlap 帧重复计权），32 轮不早停，再由不带 `--epoch` 的 `test.py`
  评测 best checkpoint。
- `bc_tpro_stage1_noise8_upstream_2026-09-09/`：上游 B1/C0/C1/C2 固定 epoch32 历史证据；
  已被 best-validation 协议取代，不再代表当前结论。
- `bc_tpro_nongate_noise8_seed47_2026-09-10/`：NG1/NG2/NG3 及旧七结构固定 epoch32
  Pareto 汇总；结果和 Excel 仅作历史审计，不能作为当前结构结论。
- 原三 seed、12 项 Stage1 矩阵在 6/12 后因用户改为单 seed 而停止，不作为当前待完成队列。

### `experiments/nudt_mirsdt_all_models_2026-09-01/`

| 文件 | 作用 |
|---|---|
| `README.md` | 干净 NUDT-MIRSDT 的数据、32 epoch、batch、seed、损失、GPU 和复现命令。 |
| `manifest.tsv` | 29项任务的权威清单：run_id、模型文件、BRTD3 variant、损失。 |
| `results.csv` | 从训练日志提取的 pixel IoU/Precision/Recall/F1 和相对本地 DeepPro-Plus 的差值。 |
| `RESULTS.md` | `results.csv` 的可读表格。它是训练筛选指标，不是论文主表。 |
| `ANALYSIS.md` | 基于 pixel F1 的模型族、结构消融和收敛诊断。只用于初筛。 |
| `paper_metrics.csv` | 29个模型的论文指标机器表格。 |
| `PAPER_METRICS.md` | 与 DeepPro-Plus 论文对齐的主要结果表，论文写作优先使用。 |

### `experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03/`

文件结构与干净集目录相同，但数据来自 `NUDT-MIRSDT-Noise8.0_FJY`。其用途是检验模型
在高噪声、低 SNR 条件下的鲁棒性，并与论文的 NUDT-MIRSDT-HiNo 结果对齐。

## 8. 论文材料：`paper/`

| 文件 | 作用 |
|---|---|
| `Li 等 - 2026 - Probing Deep into Temporal Profile Makes the Infrared Small Target Detector Much Better.pdf` | DeepPro/DeepPro-Plus baseline 原论文。模型描述、训练协议、Pd/Fa/AUC 定义和官方结果均以此为准。 |
| `DEEPPRO_PLUS_METRIC_ALIGNMENT.md` | 从论文提取的可执行评价协议，列出阈值、SNR 分组、指标单位、官方 baseline 数值，以及本地训练协议与论文协议的差异。 |

论文结果中必须区分：

- `DeepPro-Plus (paper)`：论文已经报告的官方数字；
- `deeppro_plus (local controlled baseline)`：与本地候选模型采用相同训练协议的从零训练结果。

两者训练损失和学习率不同，不能合并或把本地结果写成官方配置复现。

## 9. 当前复现工具：`tools/`

### 当前 Noise8 BC-TPro

| 文件 | 作用 |
|---|---|
| `run_bc_tpro_bestval.py` | 当前入口：在物理 GPU 0/1/2 排队重跑七个 seed47 结构，每 epoch 验证并在训练后评测 `best_model.pth`。 |
| `analyze_bc_tpro_bestval.py` | 核验 best checkpoint 与逐轮验证记录，并输出 Pd/Fa/AUC Pareto 报告及 Excel。 |
| `check_bc_tpro_nongate_memory.py` | 使用真实 train64 batch 验证三个无门控分支的 FP32 显存、loss 和梯度。 |
| `run_bc_tpro_nongate.py` | 历史固定 epoch32 复现入口；不再作为当前实验启动器。 |
| `analyze_bc_tpro_nongate.py` | 核验已被取代的七个 epoch32 产物；其 Markdown、CSV 和 Excel 仅作历史审计。 |
| `run_bc_tpro_stage1_noise8_upstream.sh` | 原三 seed upstream 矩阵入口；当前单 seed 修订不再要求补齐它。 |
| `analyze_bc_tpro_noise8_paper.py` | 三 seed候选锁工具；只有 Pareto 可决时才可锁定，指标权衡时 fail-closed。 |

当前单 seed 结果不支持显著性或跨 seed 稳定性结论。不同架构按 Pd 高、Fa 低、AUC 高的
三指标 Pareto 关系联合解释，不使用 AUC 优先或未登记的加权综合分。

从 2026-09-11 起，NUDT-MIRSDT 系列上所有显式提供内部 train/val 划分的新 BC-TPro
launcher 必须设置
`eval_interval=1`、`skip_inprocess_validation=0`、`validation_safe_cudnn=1`、
`validation_overlap_policy=official_window`、`eval_chunk_rows=32`、
`early_stopping_patience=0` 和 `run_test_after_train=0`。每 epoch 完整验证 internal-val16，
并以官方逐窗口累计的 micro pixel IoU@0.5 选择同一 run 的 `best_model.pth`（overlap 帧
重复计权，平局取较晚 epoch）；验证与评测使用数值等价且在本机稳定的 32 行分块路径。
训练运行满 32 epochs，随后由不带 `--epoch` 的 `test.py`
对 best 独立计算 Pd/Fa/AUC。pixel IoU 不
用于结构排名。旧 upstream/nongate launcher 与结果只保留作历史复现。official test20
不参与；final80 因无独立 val 而暂停，不能每 epoch 读取 test20。详见
[验证节奏](VALIDATION_SCHEDULE_2026-09-11.md)。

### 环境与历史批量实验

| 文件 | 作用 |
|---|---|
| `project_runtime_env.sh` | 统一解析仓库、数据和 Python 路径；设置显存分配方式；强制 GPU 0/1/2 白名单。所有 launcher 的共同安全入口。 |
| `run_nudt_mirsdt_all_models.sh` | 根据 manifest 在三张 GPU 上运行29项任务；支持文件锁、断点续训、失败归档、TDCSTA/FeedbackSTS 特殊配置和 SwanLab。 |
| `run_clean_then_noise8_all_models.sh` | 先完成干净 NUDT，再完成 Noise8 的自动流水线；检查 `.done`，失败时有限重试，最后生成筛选汇总。 |
| `summarize_nudt_mirsdt_results.py` | 从训练日志提取 pixel IoU、Precision、Recall、F1，生成 `results.csv/RESULTS.md`。 |
| `analyze_nudt_experiments.py` | 对训练筛选 CSV 做模型族、BRTD3 消融、稳定性和跨数据集诊断。 |

### 论文指标

| 文件 | 作用 |
|---|---|
| `run_nudt_paper_metrics.sh` | 三卡批量运行29个 checkpoint 的 `test.py`，固定阈值0.5并输出每个模型的论文指标 JSON。 |
| `summarize_nudt_paper_metrics.py` | 汇总 Pd/Fa/AUC，加入论文官方 DeepPro-Plus 参照行，生成 CSV 与 Markdown；不虚构单一综合分。 |
| `run_paper_metrics_after_noise8.sh` | 等待 Noise8 训练锁释放，然后依次评测干净集和 Noise8 的所有模型。 |

### 模型结构验证

| 文件 | 作用 |
|---|---|
| `check_brtd.py` | 检查第一代 BRTD 的零初始化行为、参数和相对 DeepPro-Plus 的输出一致性。 |
| `check_brtd3.py` | 遍历 BRTD3 variants，验证残差初始化、梯度流和形状契约。结构修改后应运行。 |

### SatVideoIRSDT 比赛提交复现

这些文件属于已经结束的比赛提交阶段，不是当前 NUDT 论文主实验：

| 文件 | 作用 |
|---|---|
| `run_final_test_hrms_scratch.sh` | 使用冻结的 epoch-86 Hybrid-RMS scratch 权重执行最终推理、阈值后处理和打包。 |
| `run_final_test_pointcenter_scratch.sh` | PointCenter scratch 候选的最终推理与提交包生成。 |
| `centroid_f1_sweep.py` | 对概率图扫描阈值和面积过滤，以质心级 F1 选择后处理参数。 |
| `probability_ensemble_sweep.py` | 扫描两个模型概率图的融合权重与阈值。 |
| `build_single_submission.py` | 从 checkpoint/阈值扫描中保留最佳单模型质心提交。 |
| `build_ensemble_submission.py` | 根据最佳融合参数生成比赛质心 TXT。 |
| `select_eval_checkpoints.py` | 按训练日志的验证 F1 选择候选 checkpoint。 |
| `resume_structure_candidate_postprocess.sh` | 使用文件锁恢复概率图后处理、阈值扫描和 ZIP 生成。 |
| `validate_submission_zip.py` | 校验提交 ZIP 是否扁平、文件名是否合法、序列/帧数是否匹配、内容能否解析。 |

### SwanLab 与历史研究脚本

| 文件或模式 | 作用与注意事项 |
|---|---|
| `stream_training_log_to_swanlab.py` | 将已有本地训练日志按 epoch 回放到 SwanLab，适合云端中断后的补录。 |
| `launch_swanlab_sidecars_8run.sh` | 历史8任务 SwanLab 侧车入口，只用于旧实验追溯。 |
| `run_structure_candidate_experiment.sh` | SatVideo BRTD3 单候选训练、推理和后处理的历史通用 runner。 |
| `run_structure_candidate_queue.sh` | 在单卡上串行运行多个结构候选。 |
| `launch_raw_apmd_experiment.sh` | 早期 Raw-APMD 单实验入口。 |
| `launch_raw_apmd_rms_2gpu.sh` | 两个 seed 的 Raw-APMD-RMS 历史配对实验。 |
| `launch_raw_apmd_optimizations_6gpu.sh` | 文件名保留6卡历史语义；当前必须经过 GPU 白名单，不作为论文入口。 |
| `launch_structure_round2_8gpu.sh` | 第二轮结构候选旧入口，包含已过时的8卡假设。 |
| `launch_scratch_model_candidates_3gpu.sh` | scratch-init/bandpass/detail 三候选调度。 |
| `launch_priority_scratch_ddp3.sh` | 三卡优先 scratch 候选训练。 |
| `launch_hybrid_rms_scratch_3gpu.sh` | Hybrid-RMS scratch 三卡入口的兼容别名。 |
| `launch_hybrid_rms_pretrain_ablation_3gpu.sh` | 名称含 pretrain 的历史入口；当前训练代码拒绝加载预训练权重。 |
| `launch_hybrid_rms_pretrain_ablation_8gpu.sh` | 已退休，会重定向到三卡 scratch launcher。 |
| `launch_feedbacksts_f1_ddp3.sh` / `run_feedbacksts_f1_experiment.sh` | FeedbackSTS 三卡 F1 历史训练流程。 |
| `resume_feedbacksts_f1_ddp3.sh` | 恢复上述 FeedbackSTS 训练和 SwanLab run。 |
| `run_feedbacksts_hrms_ensemble_probe.sh` | FeedbackSTS 与 HRMS 概率融合的历史探针。 |
| `run_pointcenter_f1_experiment.sh` | PointCenter 三卡训练及后处理历史入口。 |
| `resume_priority_scratch_ddp3_fast.sh` | 恢复并加速 SatVideo scratch-init 训练。 |
| `finish_init_then_launch_bandpass_ddp3.sh` | 等待 scratch-init 后处理完成，再启动 bandpass 的串行控制脚本。 |
| `run_valid_frame_architecture_experiment.sh` | 有效帧 mask 与零填充行为的历史结构实验。 |
| `tools/README.md` | 当前入口、比赛复现入口和历史脚本的导航索引。 |

## 10. 研究文档：`docs/`

| 文件 | 内容 | 使用边界 |
|---|---|---|
| `IMPORTANT_FILES_GUIDE.md` | 本文件，当前仓库重要文件字典。 | 接手和整理仓库的第一入口。 |
| `README.md` | 文档总索引、当前规则和历史材料导航。 | 某些“当前任务”状态需结合 experiments 实时文件。 |
| `VALIDATION_SCHEDULE_2026-09-11.md` | 新 BC-TPro 每 epoch internal-val16、按 IoU 选 best、best 外部检测评测、历史实验与 final80 暂停。 | 2026-09-11 起的新实验必须遵循。 |
| `EXPERIMENT_OPTIMIZATION_HANDOFF_2026-09-10.md` | 当前 BC-TPro、完整网络/损失演进、官方指标、现场状态、恢复和 final80 协议。 | 新对话和当前论文实验的首选入口。 |
| `MODEL_EVOLUTION_ARCHITECTURE_AND_LOSS_2026-08-26.md` | DeepPro 到 BRTD/Raw-APMD/Hybrid-RMS/FeedbackSTS 前期的历史结构与损失快照。 | 仅作历史素材；完整更新和当前状态见 `EXPERIMENT_OPTIMIZATION_HANDOFF_2026-09-10.md`。 |
| `F1_MAXIMIZATION_RESEARCH_2026-08-27.md` | 遥感、视频恢复、检测等跨领域思路及 PointCenter 决策。 | 历史研究依据；当前论文主指标已改为 Pd/Fa/AUC。 |
| `SCRATCH_MODEL_IMPROVEMENT_2026-08-25.md` | 非零投影、bandpass、detail 三个 scratch候选的设计和梯度验收。 | 解释 scratch-only 结构设计。 |
| `WEBSITE_RESULTS_ANALYSIS_2026-08-25.md` | 比赛网站得分及 pretrained/scratch 决策。 | 仅用于比赛历史，不作为 NUDT 论文实验。 |
| `raw_apmd.md` | Raw Appearance + Multi-scale first/second-order Motion Difference 的设计。 | BRTD3 方法部分基础。 |
| `brtd2_research.md` | BRTD2 外观保留、多尺度时域和可靠性门控研究。 | 第二代结构的历史依据。 |
| `structure_optimization_2026-08-20.md` | RMS、Channel-RMS、motion detrend、multiscale contrast 的比较设计。 | Hybrid-RMS 和消融解释。 |
| `structure_round2.md` | 对齐、传播、低频净化、双流等第二轮候选。 | 未必有效，需以当前实验结果为准。 |
| `experiment_analysis_2026-08-20.md` | 早期结构实验复盘。 | 历史结果，不替代新统一协议。 |
| `MIGRATION_HANDOFF_2026-08-24.md` | 从旧服务器迁移时的数据、环境、模型和任务交接。 | 旧 GPU 数量和路径可能过时。 |
| `MIGRATION_ACCEPTANCE_2026-08-25.md` | 新服务器环境、数据、GPU和推理验收结果。 | 排查运行环境时使用。 |
| `CONVERSATION_HANDOFF.md` | 早期整体对话与任务交接。 | 决策追溯。 |
| `BRTD_CONVERSATION_HANDOFF_2026-08-12.md` | 第一阶段 BRTD 对话记录。 | 决策追溯。 |
| `BRTD_CONVERSATION_HANDOFF_2026-08-20.md` | 第二阶段 BRTD 与结构优化对话记录。 | 决策追溯。 |
| `environment_sjyPID_2026-08-24.yml` | Conda 环境 YAML 导出。 | 便于重建主要依赖。 |
| `conda_explicit_sjyPID_2026-08-24.txt` | 带构建号的 Conda 精确包清单。 | 环境审计和严格复现。 |
| `pip_freeze_sjyPID_2026-08-24.txt` | pip 包及版本清单。 | 补足 Conda YAML 未固定的 Python 包。 |

## 11. 冻结发布集：`release/`

### `release/README.md`

说明发布集与普通日志的区别：只有具备来源、配置、校验值和验证证据的产物才进入该目录。

### `release/2026-08-29_final_submission_score91.30_scratch/`

这是比赛最终 91.30 分 scratch-only 版本的历史审计目录。2026-09-04 已删除其中的
提交 ZIP、轨迹 TXT 相关哈希、提交校验记录和轨迹生成日志；checkpoint、源码快照、
训练证据与验证扫描仍保留。

| 子目录/文件 | 作用 |
|---|---|
| `README.md` | 发布包总说明和复现入口。 |
| `MANIFEST.md` | 发布包文件清单与角色说明。 |
| `RESULT.md` | 最终网站成绩与实验结论。 |
| `checkpoint/epoch_86_model.pth` | 最终提交使用的 scratch Hybrid-RMS 权重。 |
| `scripts/reproduce_submission.sh` | 已停用的历史提交生成实现，仅供阅读。 |
| `scripts/verify_release.sh` | 已停用的历史发布校验实现，仅供阅读。 |
| `source_snapshot/` | 生成该结果时冻结的训练、测试、模型、适配器、损失、数据加载和提交校验源码。不要用当前源码替换后仍声称完全复现。 |
| `environment/environment.yml` | 最终发布环境。 |
| `validation/*.csv/*.json` | epoch-86 和高分辨率阈值扫描数值。 |
| `evidence/training.log` | 原训练日志。 |
| `evidence/inference_part1.log` / `inference_part2.log` | 分段推理日志。 |
| `evidence/scratch_training_evidence.txt` | 未加载预训练权重的证据。 |
| `evidence/provenance.txt` | 权重、源码与提交产物的来源链。 |
| `evidence/independent_audit.txt` | 独立复核记录。 |
| `evidence/reproduction_verification.md` | 复现结果和验收结论。 |

### `release/2026-08-22_pretrained_vs_scratch_seed47/`

这是8组 pretrained/scratch 历史对照快照。`README.md` 解释协议，`results.csv` 保存结果，
`source_snapshot/` 保存当时的 BRTD3、适配器和损失源码；提交 ZIP、校验 TXT 和哈希
清单已清理。
它只用于历史审计；当前论文新训练不得用其权重初始化。

## 12. 比赛后处理工具：`tools_forSatVideoIRSTD/`

| 文件 | 作用 |
|---|---|
| `README.md` | 从分割结果生成质心与轨迹的说明。 |
| `seg2centroid_txt.py` | 将二值分割 mask 转成逐帧目标质心 TXT。 |
| `seg2tracked_centroid_txt.py` | 在相邻帧间关联质心，生成比赛要求的轨迹记录。 |
| `codalab比赛操作流程.pdf` | 评测平台提交操作说明。 |
| `参赛须知与提交结果说明(3).pdf` | 官方提交格式和竞赛规则。 |
| `参赛须知与提交结果说明.docx` | 上述说明的可编辑文档版本。 |
| `__pycache__/` | Python 字节码缓存，可删除。 |

## 13. Attribution 目录

原 `attribution/core.py` 和 `attribution/utils.py` 只有 import、没有任何函数实现，已在
2026-09-11 仓库清理中删除。`test.py --attribution` 继续明确报 `NotImplementedError`，
避免把缺失功能误写成可复现工具；论文中若需要归因图，必须重新实现并验证。

## 14. 日志与大体积产物：`log/`

`log/` 占仓库绝大部分空间，但它不是手写源码。需要理解以下结构，而不是逐帧阅读文件。

### 当前 NUDT 实验根目录

2026-09-08 整理说明与核验结果见 [日志目录整理](LOG_LAYOUT_2026-09-08.md)。

- `log/sem_seg/<日期>/<数据集>__<开始时间>__F1OHEM-<run_id>_seed49_E32/`：两批共58项训练产物，按训练日志第一条记录的实际日期归档。
- PointCenter 使用 `CenterConsistencyF1` 损失标签，其余命名规则相同。
- `experiments/<批次>/manifest.tsv` 的 `log_dir` 列给出相对 `log/sem_seg/` 的精确路径。
- `log/sem_seg/_queues/<批次>/`：批次状态、启动日志、失败记录，以及旧 run_id 的兼容链接。
- `log/sem_seg/_pipeline/`：跨批次调度日志与锁；`_archive/`：原有历史归档。
- 旧 `log/nudt_*/` 及 `log/archive` 等目录兼容入口已移除；请使用规范路径。
- 历史归档位于 `_archive/2026-08-31/untracked_tools/` 和 `_archive/2026-09-01/nudt_setup/`。

每个 `_queues/<批次>/` 目录包含：

| 路径模式 | 作用 |
|---|---|
| `.launch.lock` | 防止两个 launcher 同时写同一实验目录。进程结束后文件可以仍存在，锁是否被持有才表示运行状态。 |
| `status/<run_id>.running` | 当前运行状态，记录 GPU 和开始时间。 |
| `status/<run_id>.done` | 成功完成标记，记录结束时间和耗时。 |
| `status/<run_id>.failed` | 失败标记和退出码；重试成功后应被移除。 |
| `launcher_logs/<run_id>.log` | launcher 捕获的完整 stdout/stderr。 |
| `failed_attempts/<run_id>_<时间>/` | 被归档的失败实验目录与错误证据，防止重试覆盖。 |
| `sem_seg/<run_id>/` | 指向按日期命名实验目录的兼容符号链接。 |
| `paper_metrics/raw/` | 论文评测原始 JSON。 |
| `paper_metrics/status/` | 论文评测运行/成功/失败状态。 |
| `paper_metrics/logs/` | 论文评测终端日志。 |
| `PIPELINE_COMPLETE` | 整批训练完成标记（如存在）。 |

### 单个实验目录 `log/sem_seg/<日期>/<实验名>/`

| 文件/目录 | 作用 |
|---|---|
| `<ModelName>.py` | 训练启动时复制的模型源码快照。测试旧权重时优先加载它。 |
| `structure_adapters.py` / `brtd_adapter.py` 等 | 该模型依赖的模块快照，仅在相应模型中存在。 |
| `segmentation_losses.py` | 当次训练使用的损失源码快照。 |
| `logs/<ModelName>.txt` | epoch 级学习率、训练损失、IoU/F1、验证结果、checkpoint 路径和错误信息。 |
| `checkpoints/best_model.pth` | 按验证 pixel IoU 保存的最佳 checkpoint，也是默认 `test.py` 读取文件。 |
| `checkpoints/latest_model.pth` | 最近一次可恢复训练状态，主要用于断点续训。 |
| `checkpoints/epoch_<N>_model.pth` | 固定轮次快照，仅用于回溯；当前 BC-TPro 最终评测读取验证选出的 `best_model.pth`。 |
| `swanlog/` | SwanLab 本地/离线运行记录，包括 `.swanlab` 备份文件。 |
| `eval.txt` / `eval_epoch-<N>.txt` | `test.py` 生成的目标级和像素级完整评测日志。 |

### 历史 SatVideo 日志

- `log/sem_seg/<日期>/`：按日期保存的大量比赛训练和推理实验。
- `visual*/*.png`：模型输出概率图或可视化 mask，每张对应一个视频帧。
- `out_centroid*/*.txt`：历史上用于保存逐序列质心检测结果，已于 2026-09-04 清理。
- `out_centroid_tracked*/*.txt`：历史轨迹结果，已于 2026-09-04 清理。
- `*.zip`：历史提交包或中间归档，已于 2026-09-04 全部清理。
- `*.pth`：不同 epoch、best/latest 或早停 checkpoint。

保留的 checkpoint、训练日志和 provenance 可用于结果追溯；不能根据文件名判断模型
是否使用预训练，必须同时检查 checkpoint 配置和训练日志。

### 其他日志文件

| 文件/目录 | 作用 |
|---|---|
| `clean_then_noise8_all_models_2026-09-03.log` | 干净集到 Noise8 自动训练流水线总日志。 |
| `noise8_resume_offline_2026-09-04.screen.log` | SwanLab 云端失败后以 offline 模式恢复 Noise8 的 Screen 日志。 |
| `paper_metrics_after_noise8_2026-09-04.log` | 等待训练完成并运行论文指标评测的流水线日志。 |
| `*.screen.log` | 对应 GNU Screen 会话的终端副本。 |
| `.clean_then_noise8_all_models.lock` / `.paper_metrics_after_noise8.lock` | 防止重复启动整条流水线。 |
| `archive/` | 历史日志归档。 |
| `archived_untracked_tools_2026-08-31/` | 仓库整理时归档的未跟踪旧工具。 |

## 15. 自动生成和可删除内容

以下内容不是源代码，可按需要清理，但清理训练日志或权重前应先确认是否仍需论文审计：

| 模式 | 是否可删 | 说明 |
|---|---|---|
| `**/__pycache__/`、`*.pyc` | 可以 | Python 自动缓存，会重新生成。 |
| 空闲的 `*.lock` 文件 | 通常可以 | 文件存在不代表锁被占用；先用 `flock` 或检查进程。 |
| SwanLab 本地缓存 | 谨慎 | 云端同步完成且不需离线恢复时可删。 |
| 重复的逐帧 PNG/TXT | 谨慎 | 确认最终 ZIP、阈值结果和必要示例图已冻结后再删。 |
| `latest_model.pth` | 谨慎 | 训练完成且 best/固定 epoch 已验证后才可考虑删除。 |
| `failed_attempts/` | 暂不建议 | 包含故障原因和重试证据，对论文可复现性排查有价值。 |
| `release/` | 不应删除 | 冻结发布和真实性审查证据。 |

## 16. 当前论文工作真正需要保留的最小集合

如果后续需要另建一个论文复现分支，至少保留：

1. 根目录的 `train.py`、`test.py`、`runtime_utils.py`、`sequence_utils.py`、
   `ShootingRules.py`、`write_results.py`。
2. 完整 `data_utils/` 和 `networks/`。
3. `tools/project_runtime_env.sh`、NUDT训练/评测/汇总脚本。
4. 两个 NUDT experiment 目录中的 README、manifest、CSV、Markdown 和 paper metric JSON。
5. 每个正式报告模型的模型源码快照、loss/adapter 快照、训练日志和选定 checkpoint。
6. `paper/` 中 baseline 论文与指标协议。
7. 环境 YAML、Conda/pip 版本清单和数据集下载/划分说明。

不要把所有历史 SatVideo 可视化、旧8卡 launcher 或 pretrained 对照权重混入论文最小复现包；
它们可以留在完整仓库中用于审计，但会模糊当前方法、数据和实验协议。
