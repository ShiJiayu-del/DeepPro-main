# Noise8 BC-TPro 最终 80/20 论文实验

> [!CAUTION]
> **本方案已暂停，不得执行。** 它使用全部 train80，因而没有独立验证集，无法按
> 2026-09-11 确认的“每 epoch 验证并测试 `best_model.pth`”规则选择 checkpoint。原固定
> epoch32 后测试的活动方案已废止；也绝不能把 official test20 当作逐 epoch 验证集，否则
> 会让测试集参与选模。当前仅保留本目录作历史设计审计。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + reproducibility validation
- Origin Date: 2026-09-09 (Asia/Shanghai)
- Verification Status: HISTORICAL_TOOLING_ONLY; SUSPENDED; DO_NOT_RUN
- Version Label: noise8_bc_tpro_final80_upstream_schema2_v2

本目录记录 upstream profile `upstream8fa1a68_fp32` 及
`experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/OFFICIAL_METRIC_AMENDMENT_2026-09-10.md`
冻结的旧 final80 设计。当前仅完成工具设计和 CPU 检查，**未启动最终训练，未读取官方
test 图像**；该固定 epoch32 设计已被 best-validation 协议取代。

## 历史执行顺序（已废止，不执行）

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
   seeds 47/49/51 的固定 epoch-32 随机初始化训练。此项与当前 best-checkpoint 规则冲突，
   是本方案被暂停的直接原因。
5. 六个或三个训练 checkpoint 全部验证完成后，才逐个启动官方 test 20 序列的
   独立评测。每次评测前再次重放并核对冻结协议，使用显式 `--split test`；每个
   checkpoint 只有一次自动评测机会，失败不自动重试。
6. 只汇总 Pd@0.5、Fa@0.5、论文 27 阈值 Pd-Fa AUC 的 mean ± sample SD。

## 历史固定配置（只读）

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
- 原设计中 `train.py` 会在构造 `TestIRSeqDataLoader` 前直接返回，独立评测原拟使用
  `test.py --epoch 32`。该流程现已废止，不得启动。历史 29 模型实验曾评测过同一 test20，
  因而 test20 也不能描述为项目级从未暴露。
- 若需要逐 epoch validation，必须另建使用 train-only 内部划分的非 final80 协议。规则见
  [`docs/VALIDATION_SCHEDULE_2026-09-11.md`](../../docs/VALIDATION_SCHEDULE_2026-09-11.md)。
- 本流程只核对规范化路径、逐帧唯一清单与文件名配对、参数、checkpoint 元数据和
  逐序列整数计数；不生成或校验 SHA256、MD5 等内容哈希。

## 历史命令（禁止执行）

以下命令只记录旧入口；即使候选锁将来存在，也不得按本方案执行：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_noise8_final.sh
```

不要运行非 `DRY_RUN` 命令。若将来恢复 final 阶段，必须先另行登记带独立 train-only
验证划分的协议、明确 best checkpoint 选择方法，并重新审查 test20 屏障；不能直接复活
本目录的 schema-2 锁或 launcher。

## 解释边界

当前没有可与 DeepPro-Plus 论文 HiNo 表格并列的活动 final80 结果。未来若建立合规协议，
三个 seed 也只能描述初步训练波动，不能将小差异或区间重叠解释为等效性。Noise8 数据缺少
`masks_centroid` 时，Pd/Fa 评测使用相邻 Clean 数据集同名派生质心，该事实必须
在论文方法和实验设置中披露。
