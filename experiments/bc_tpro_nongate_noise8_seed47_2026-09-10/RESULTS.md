# BC-TPro 无门控结构实验结果

> [!WARNING]
> 本页是**已被取代的固定 epoch32 历史结果**。这些 run 训练时没有逐 epoch 验证，训练后
> 仅评测 `epoch_32_model.pth`，不能据此知道真正的验证集最佳 checkpoint。下表及原 C1
> 判读只用于 provenance 审计，不再代表当前模型结论。当前七结构重跑见
> [`bc_tpro_stage1_noise8_bestval_seed47_2026-09-11`](../bc_tpro_stage1_noise8_bestval_seed47_2026-09-11/README.md)，
> 启动入口为 `tools/run_bc_tpro_bestval.py`。

## Material Passport

- Origin workflow: academic-research-suite / experiment-agent / analyze
- Verification: HISTORICAL; ARTIFACTS_VERIFIED; SINGLE_SEED; SUPERSEDED_BY_BESTVAL

| 模型 | Seed | GPU | 参数量 | Pd@0.5 (%) ↑ | Fa@0.5 (×10⁻⁵) ↓ | AUC27 ↑ | Pareto非支配 |
|---|---:|---:|---:|---:|---:|---:|---|
| B1 | 47 | 0 | 70913 | 78.921569 | 3.227150 | 0.932218228 | 是 |
| C0 | 47 | 0 | 71233 | 76.540616 | 2.961721 | 0.912537383 | 否 |
| C1 | 47 | 1 | 71233 | 78.851541 | 2.799961 | 0.971487124 | 是 |
| C2 | 47 | 2 | 71281 | 78.011204 | 2.959040 | 0.973573062 | 是 |
| NG1 | 47 | 0 | 71233 | 77.731092 | 3.605185 | 0.942356952 | 否 |
| NG2 | 47 | 1 | 71233 | 77.871148 | 3.416614 | 0.960096460 | 否 |
| NG3 | 47 | 2 | 71233 | 75.490196 | 2.680206 | 0.941664698 | 是 |

历史 epoch32 表内 Pareto 非支配集：B1、C1、C2、NG3；不得外推为当前非支配集。

## 相对 B1

| 模型 | ΔPd (百分点) | ΔFa (绝对比例) | Fa相对降幅 (%) | ΔAUC27 | 关系 |
|---|---:|---:|---:|---:|---|
| B1 | +0.000000 | +0 | -0.000000 | +0.000000000 | equal |
| C0 | -2.380952 | -2.654288329e-06 | +8.224868 | -0.019680846 | tradeoff |
| C1 | -0.070028 | -4.27188492007e-06 | +13.237330 | +0.039268896 | tradeoff |
| C2 | -0.910364 | -2.68109932222e-06 | +8.307948 | +0.041354833 | tradeoff |
| NG1 | -1.190476 | +3.78035004433e-06 | -11.714207 | +0.010138723 | tradeoff |
| NG2 | -1.050420 | +1.89464352103e-06 | -5.870950 | +0.027878232 | tradeoff |
| NG3 | -3.431373 | -5.46944261732e-06 | +16.948214 | +0.009446469 | tradeoff |

## 相对 C1

| 模型 | ΔPd (百分点) | ΔFa (绝对比例) | Fa相对降幅 (%) | ΔAUC27 | 关系 |
|---|---:|---:|---:|---:|---|
| B1 | +0.070028 | +4.27188492007e-06 | -15.256942 | -0.039268896 | tradeoff |
| C0 | -2.310924 | +1.61759659107e-06 | -5.777210 | -0.058949742 | dominated |
| C1 | +0.000000 | +0 | -0.000000 | +0.000000000 | equal |
| C2 | -0.840336 | +1.59078559785e-06 | -5.681455 | +0.002085937 | tradeoff |
| NG1 | -1.120448 | +8.0522349644e-06 | -28.758379 | -0.029130173 | dominated |
| NG2 | -0.980392 | +6.1665284411e-06 | -22.023620 | -0.011390664 | dominated |
| NG3 | -3.361345 | -1.19755769726e-06 | +4.277051 | -0.029822427 | tradeoff |

## 历史结果判读（不再是当前结论）

- 相对 C1 被三指标共同支配：NG1、NG2。
- 相对 C1 存在指标取舍：NG3。
- 在这批固定 epoch32 历史结果中，没有新增结构在 Pd、Fa、AUC 三项上综合支配 C1；新
  best-validation 重跑完成前，不对当前最优结构下结论。
- 单seed结果不用于宣称统计稳定性或唯一综合最优模型。

## 协议与限制

- 状态：HISTORICAL; ARTIFACTS_VERIFIED; SINGLE_SEED; SUPERSEDED。所有旧指标经原始整数
  计数重算，但 checkpoint 选择协议不符合当前要求。
- 数据集：NUDT-MIRSDT-Noise8.0_FJY；固定 train64/internal-val16；仅 seed47。
- 历史协议为 FP32、scratch-only、固定 epoch32、T=40、batch4、crop128、Soft-IoU；它跳过
  逐 epoch 验证且未按验证集选择 best，故已被取代。
- 联合目标为 Pd@0.5 越高、Fa@0.5 越低、官方 27 阈值 Pd-Fa AUC 越高。
- dominates：三项均非劣且至少一项严格改善；dominated：被参照支配；
- tradeoff：有改善也有退步；equal：三个未四舍五入数值完全相等。
- Pareto 非支配集表示没有被表内其他模型支配，不代表三项都优于基线或唯一最佳。
- 差值均为当前行减参照；Fa 相对降幅为 (参照Fa-当前Fa)/参照Fa，正值表示减少。
- 单seed没有训练随机性标准差、显著性结论或跨seed稳定性证据；没有自动候选锁。
- 这些历史结果来自 internal-val16，不能直接与论文 official test20 数值比较。
- 本轮不评测 official test20；该划分曾在项目历史实验使用，不是项目级从未见过的外部测试。
- NG1/NG2/NG3均为71,233参数；保留既有SiLU激活，没有新增可学习乘法门控。
- NG1固定中心减环形背景，NG2固定时间带通，NG3固定十字空间平滑；均用加法残差。
- 等参数不意味着同噪声增益；NG3平滑同时改变噪声幅度与单像素目标响应。
- B1/C0/C1/C2复用已有seed47结果；C1/C2实际GPU为1/2，覆盖旧manifest的GPU0记录。
- 行顺序为预登记 B1/C0/C1/C2/NG1/NG2/NG3；无加权分数，无单指标排序。当前协议中
  micro pixel IoU@0.5 只选择同一 run 的 `best_model.pth`；结构之间仍按 Pd 高、Fa 低、
  AUC 高的三指标 Pareto 关系比较。

## 原始证据

- B1：[指标](../bc_tpro_stage1_noise8_upstream_2026-09-09/metrics/b1_none_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-B1_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-B1_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-B1_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09/status/b1_none_seed47.done`。
- C0：[指标](../bc_tpro_stage1_noise8_upstream_2026-09-09/metrics/c0_temporal_control_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C0_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C0_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C0_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09/status/c0_temporal_control_seed47.done`。
- C1：[指标](../bc_tpro_stage1_noise8_upstream_2026-09-09/metrics/c1_center_multiscale_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C1_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C1_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C1_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09/status/c1_center_multiscale_seed47.done`。
- C2：[指标](../bc_tpro_stage1_noise8_upstream_2026-09-09/metrics/c2_center_ring_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C2_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C2_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-09/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-BCTPro-C2_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09/status/c2_center_ring_seed47.done`。
- NG1：[指标](metrics/ng1_ring_difference_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG1_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG1_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG1_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_nongate_noise8_seed47_2026-09-10/status/ng1_ring_difference_seed47.done`。
- NG2：[指标](metrics/ng2_bandpass_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG2_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG2_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG2_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_nongate_noise8_seed47_2026-09-10/status/ng2_bandpass_seed47.done`。
- NG3：[指标](metrics/ng3_spatial_smooth_seed47__noise8_internal_val.json)；本地运行证据：`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG3_seed47_E32/checkpoints/epoch_32_model.pth`、`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG3_seed47_E32/logs/DeepPro-Plus_BCTPro.txt`、`log/sem_seg/2026-09-10/NUDT-MIRSDT-Noise8.0_FJY__Upstream8fa1a68-FP32-SoftIoU-NG3_seed47_E32/eval_epoch-32.txt`、`log/sem_seg/_queues/bc_tpro_nongate_noise8_seed47_2026-09-10/status/ng3_spatial_smooth_seed47.done`。
