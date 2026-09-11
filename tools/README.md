# Tools

工具保留在单层目录，避免改变已有启动器之间的相对引用。本索引区分当前入口和历史
脚本；历史脚本可能记录旧 GPU 数量或预训练流程，不能直接复制执行。

## 当前 NUDT-MIRSDT / Noise8 论文实验

| 文件 | 用途 |
|---|---|
| `project_runtime_env.sh` | 解释器、数据路径和 GPU 0/1/2 白名单 |
| `check_bc_tpro_nongate_memory.py` | 用真实 train64 batch 检查三个无门控分支的 FP32 显存、loss 和梯度 |
| `run_bc_tpro_nongate.py` | 在 GPU 0/1/2 并行运行 NG1/NG2/NG3 单 seed 训练并串行评测 |
| `analyze_bc_tpro_nongate.py` | 核验七个 seed47 结果，报告 Pd/Fa/AUC27、Pareto 关系并生成 CSV/Markdown/Excel |
| `run_bc_tpro_stage1_noise8_upstream.sh` | 原三 seed、12 项 upstream Stage1 调度；当前单 seed 修订已停止该矩阵 |
| `analyze_bc_tpro_noise8_stage1.py` | 原三 seed矩阵的 `Pd@0.5`、`Fa@0.5` 和官方 27 阈值 AUC 汇总 |
| `analyze_bc_tpro_noise8_paper.py` | 三 seed候选锁工具；只接受 Pareto 可决结果，存在指标权衡时 fail-closed |
| `run_bc_tpro_noise8_exact_logit_upstream.sh` | 可选 raw-logit 敏感性分析，不参与选模 |

当前论文主线联合考虑 TinaLRJ/DeepPro 的 `Pd@0.5` 越高、`Fa@0.5` 越低与官方
27 阈值 Pd-Fa AUC 越高，使用 Pareto 关系，不采用 AUC 优先或未登记的加权分数。
训练 loss/IoU 仅用于优化诊断；旧脚本中的 F1 汇总不得用于 Noise8 论文检测结论或
模型选择。单 seed 无门控实验不生成候选锁。新训练/评测采用路径、清单、参数、
checkpoint metadata 和整数计数做语义校验，不生成或要求文件哈希。
旧发布/迁移脚本中的 SHA256 逻辑只服务其历史归档，不应复制到新实验。

## 新 BC-TPro 验证节奏

从 2026-09-11 起，凡是 NUDT-MIRSDT 系列上显式提供内部 train/val 划分的新
BC-TPro 实验，launcher 必须
显式传入 `--eval_interval 1 --skip_inprocess_validation 0
--validation_safe_cudnn 1 --early_stopping_patience 0 --run_test_after_train 0`。
每个 epoch 的进程内验证只提供 validation loss 和 pixel IoU/P/R/F1 诊断；完整
32 epoch 后，launcher 仍须对
`epoch_32_model.pth` 独立运行一次 `test.py --epoch 32`，生成 Pd/Fa/AUC。

已完成的 upstream 和 nongate 实验保持原 external-only 参数与产物，不追改。final80
使用全部 train80、没有独立 val，必须继续跳过进程内验证，否则当前 loader 会读取
official test20。完整规则与启动前 smoke 要求见
[验证节奏](../docs/VALIDATION_SCHEDULE_2026-09-11.md)。

无门控实验完成后的结果入口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/anaconda3/envs/sjyPID/bin/python \
  tools/analyze_bc_tpro_nongate.py --xlsx
```

## SatVideoIRSDT 最终提交复现

| 文件 | 用途 |
|---|---|
| `run_final_test_hrms_scratch.sh` | 最终 scratch Hybrid-RMS 推理入口 |
| `validate_submission_zip.py` | 提交 ZIP 结构、帧数和内容校验 |
| `centroid_f1_sweep.py` | 质心阈值和面积扫描 |
| `resume_structure_candidate_postprocess.sh` | 带文件锁的后处理恢复 |

最终 91.30 分版本应优先使用
`release/2026-08-29_final_submission_score91.30_scratch/scripts/` 中冻结的脚本。

## 研究与历史工具

`run_nudt_mirsdt_all_models.sh`、`summarize_nudt_mirsdt_results.py` 以及其余
`launch_*`、`run_*`、BRTD 检查、概率融合和 SwanLab 侧车脚本用于历史研究或
消融复现。文件名中包含 `8gpu`、`6gpu`、`pretrain` 的脚本只作为审计材料；当前
`train.py` 会拒绝预训练参数，服务器策略也会阻止 GPU 3 及更高编号。
