# Noise8 BC-TPro 第一阶段实验

本目录是 `NUDT-MIRSDT-Noise8.0_FJY` 专门训练协议；它与此前“在 Clean
训练、在 Noise8 零样本评测”的实验不是同一问题。当前状态为 **正式训练进行中**；
2026-09-09 13:31（Asia/Shanghai）已通过预检查后启动 B1/C0/C1/C2 × 3 seeds 队列。

## 固定范围

- 数据根：`/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY`
- 数据集名：`NUDT-MIRSDT-Noise8.0_FJY`
- 仅从官方 `train.txt` 的 80 个序列生成固定 64/16 train/val 划分。
- 官方 `test.txt` 的 20 个序列仅在启动前读取序列名做隔离检查；不加载其图像，
  不用于训练、验证、选模或阈值选择。
- B1/C0/C1/C2，各 seeds 47/49/51；物理 GPU 固定为 0/1/2，一卡一种子。
- 全部随机初始化，32 epochs、global batch 4、T=40、crop=128、Soft-IoU、AMP。
- SwanLab：project `DeepPro-BC-TPro`，group
  `bc-tpro-stage1-noise8-scratch`，cloud 模式。
- 按用户最新要求，本协议不生成或依赖 SHA256；身份由规范化绝对路径、集合、
  计数、manifest 精确字段和 scratch-only 参数做语义校验。

## 文件

- `EXPERIMENT_PLAN.md`：揭盲前冻结的研究问题、消融和分析顺序。
- `manifest.tsv`：12 个运行的结构、seed、GPU 和日志目录。
- `splits/`：固定 64/16 划分及无哈希语义清单。
- `PRECHECK.md`：静态检查和 dry-run 证据。
- `RAW_LOGIT_PROTOCOL.md`：揭盲前固定的 raw-logit 精确工作点、重复计数与 C2
  继续门槛。
- `FINAL_EVALUATION_PLAN_2026-09-09.md`：在候选结果产生前冻结的选模规则和
  最终 80/20 论文测试协议。
- `metrics/`：正式运行后生成，每个运行一个
  `__noise8_internal_val.json`；配置阶段为空。

## 命令

只做完整校验并打印 12 条训练、12 条验证命令：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_stage1_noise8.sh
```

正式启动命令：

```bash
bash tools/run_bc_tpro_stage1_noise8.sh
```

12 个训练和概率网格评测全部完成后，再串行运行 raw-logit 敏感性评测：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_noise8_exact_logit.sh
bash tools/run_bc_tpro_noise8_exact_logit.sh
```

日志位于 `log/sem_seg/2026-09-09/`；队列状态和启动日志位于
`log/sem_seg/_queues/bc_tpro_stage1_noise8_2026-09-09/`。失败运行不会自动
重试，也不会覆盖非空运行目录。

本次后台会话名为 `bctpro_noise8_stage1_0909`。最终状态以 queue 下的 `.done`
文件、12 份 internal-val metrics 和严格分析器结果为准，而不是仅以进程退出为准。
