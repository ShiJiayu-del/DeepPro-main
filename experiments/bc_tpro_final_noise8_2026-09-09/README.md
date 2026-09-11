# Noise8 BC-TPro 最终 80/20 论文实验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + reproducibility validation
- Origin Date: 2026-09-09 (Asia/Shanghai)
- Verification Status: TOOLING_VERIFIED; CANDIDATE_LOCK_REQUIRED; NOT_RUN
- Version Label: noise8_bc_tpro_final80_upstream_schema2_v2

本目录用于 upstream profile `upstream8fa1a68_fp32` 及
`experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/OFFICIAL_METRIC_AMENDMENT_2026-09-10.md`
冻结的最终实验。当前仅完成
工具设计和 CPU 检查，**未启动最终训练，未读取官方 test 图像**。

## 强制执行顺序

1. upstream Stage1 的 12 个概率评测全部通过语义验证；raw-logit 与 dense-grid
   产物不参与继续、资格、排序或锁定。
2. `tools/analyze_bc_tpro_noise8_paper.py` 完成冻结规则重算，并最后原子生成
   `experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/LOCKED_CANDIDATE.json`。
   若官方 Pd@0.5、Fa@0.5 与 27 阈值 AUC 门槛要求 C3，分析器以
   `C3_REQUIRED` 停止，必须先
   完成预登记的 C3，不能提前锁定 C0-C2。
3. 正式启动器再次重放选择器，并排他、原子地创建不可覆盖的
   `FINAL_PROTOCOL_LOCK.json` 和 `final_manifest.tsv`；恢复运行只接受逐字段完全一致
   的候选、split 和 jobs。
4. B1 与锁定候选（若 B1 胜出则只训练 B1）分别在官方 train 80 序列上完成
   seeds 47/49/51 的固定 epoch-32 随机初始化训练。
5. 六个或三个训练 checkpoint 全部验证完成后，才逐个启动官方 test 20 序列的
   独立评测。每次评测前再次重放并核对冻结协议，使用显式 `--split test`；每个
   checkpoint 只有一次自动评测机会，失败不自动重试。
6. 只汇总 Pd@0.5、Fa@0.5、论文 27 阈值 Pd-Fa AUC 的 mean ± sample SD。

## 固定配置

- 数据：`/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY`
- 训练：official train 80、T=40、crop=128、global batch=4、32 epochs。
- 优化：Adam，lr=0.001，weight decay=0.0001，每 10 epochs 乘 0.7。
- 损失：Soft-IoU；训练与评测均为 FP32，`upstream_compat=1`；无额外几何增强。
- seeds 47/49/51 分别使用物理 GPU 0/1/2；GPU 3 不参与。
- 全部从随机权重开始；不续训、不导入任何外部模型权重。
- SwanLab cloud：project `DeepPro-BC-TPro`，group
  `bc-tpro-final80-noise8-upstream8fa1a68-fp32-scratch-locked`。
- 训练期间保持 `skip_inprocess_validation=1`、`early_stopping_patience=0` 和
  `run_test_after_train=0`。final80 使用全部 official train80，没有独立内部 val；若在未提供
  验证清单时开启进程内验证，当前 loader 会默认读取 official `test.txt`，使 test20 在每个
  epoch 进入训练流程。因此 final80 不适用 2026-09-11 的逐 epoch 内部验证规则。
- `train.py` 会在构造 `TestIRSeqDataLoader` 前直接返回。final validator 只检查 test 清单和
  文件名元数据，不解码图像；当前 BC-TPro 的实际 test 图像读取只发生在全部 final80 训练
  通过屏障后的独立 `test.py --epoch 32` 阶段。历史 29 模型实验曾评测过同一 test20，因此
  该屏障是当前协议的防泄漏约束，不代表 test20 在整个项目中从未暴露。
- 若需要逐 epoch validation，必须另建使用 train-only 内部划分的非 final80 协议。规则见
  [`docs/VALIDATION_SCHEDULE_2026-09-11.md`](../../docs/VALIDATION_SCHEDULE_2026-09-11.md)。
- 本流程只核对规范化路径、逐帧唯一清单与文件名配对、参数、checkpoint 元数据和
  逐序列整数计数；不生成或校验 SHA256、MD5 等内容哈希。

## 命令

候选锁尚未生成时，下列命令必须失败；这是预期的封闭行为：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_noise8_final.sh
```

锁定文件生成后，先检查打印出的 3 或 6 条训练命令及对应 test 计划：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_noise8_final.sh
```

确认后才可正式执行：

```bash
bash tools/run_bc_tpro_noise8_final.sh
```

正式 schema-2 协议锁和结果将写入本目录的 `FINAL_PROTOCOL_LOCK.json`、
`official_test_metrics/`、`final80_results.csv`、
`final80_summary.csv`、`final80_paired_differences.csv` 和
`FINAL80_PAPER_RESULTS.md`。训练日志仍统一位于 `log/sem_seg/2026-09-10/`。

## 解释边界

最终 80/20 结果才可与 DeepPro-Plus 论文 HiNo 表格并列。三个 seed 只能描述
初步训练波动；不能将小差异或区间重叠解释为等效性。Noise8 数据缺少
`masks_centroid` 时，Pd/Fa 评测使用相邻 Clean 数据集同名派生质心，该事实必须
在论文方法和实验设置中披露。
