# Noise8 BC-TPro upstream-aligned Stage1

新对话继续本实验时，先阅读
[`docs/EXPERIMENT_OPTIMIZATION_HANDOFF_2026-09-10.md`](../../docs/EXPERIMENT_OPTIMIZATION_HANDOFF_2026-09-10.md)。

## 2026-09-10 单 seed 修订与完成状态

用户将实验矩阵修订为只运行 seed47。B1/C0/C1/C2 的四个 seed47 训练与 internal-val16
评测均已完成并通过产物核验；B1 已有的 seed49/51 结果继续保留。原三 seed、12 项矩阵
实际完成 6/12 后停止，不再是待补齐任务，也不生成旧 schema2 candidate lock。

三项指标联合考虑 Pd 越高、Fa 越低、AUC27 越高，使用 Pareto 关系，不采用 AUC 优先或
未登记的加权分数。C1 是当前最均衡的方案；C2 仅在 AUC27 上略高，不能据此单项胜出。

## 上游锚点

- 仓库：`https://github.com/TinaLRJ/DeepPro.git`
- commit：`8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`
- 配置身份：`upstream8fa1a68_fp32`
- 所有 log_dir 都包含 `Upstream8fa1a68-FP32`，避免与现代化 AMP 运行混淆。

## 严格对齐的训练语义

`train.py --upstream_compat 1` 只对齐审计确认会改变训练分布的三项 loader 语义：

1. NUDT mask 使用 Pillow 默认 resize，然后将所有正像素二值化；
2. 使用上游时间窗口端点，因而 100 帧序列生成 96 个窗口且不纳入最后一帧；
3. 使用上游 crop 上界 `size - patch - 1`。

训练和评测均为 FP32；评测命令不带 `--amp`。`eval_chunk_rows=32` 只改变等价的
分块执行方式，不改变模型或评价公式。训练参数保持 32 epoch、global batch 4、
T=40、crop=128、Adam、lr=0.001、weight decay=0.0001、step 10 × 0.7、Soft-IoU。

## 有意保留的科研改进

本实验不复制上游代码中不可复现或会泄漏测试集的行为：

- seeds 47/49/51 显式固定并启用 deterministic；
- official train80 先固定划分为 64/16 internal train/val；
- 全部随机初始化，不加载 pretrained/base/spatial/st checkpoint；
- 固定选择 epoch-32，不按验证集或 official test 挑 checkpoint；
- Stage1 不访问 official test 图像，也不把 official test 用于训练、评测或选模；
- 使用当前可运行的索引采样实现，语义等价于上游的加权类别抽样，但兼容现代 NumPy。

因此它是“上游训练数据语义对齐的可复现对照”，不是逐 bug、逐随机轨迹复制。
机器可读定义见 `UPSTREAM_PROTOCOL.json`。本协议不生成或校验 SHA256、MD5 等哈希。

这里的 test 隔离只描述当前 BC-TPro Stage1。历史 29 模型实验已经评测过同一 test20，
所以它不是整个项目从未见过的数据；后续论文必须披露这一历史暴露，不能把 final80/test20
写成完全无偏的首次外部验证。

## 原三 seed 预注册矩阵（历史，未完成）

- 原计划为 B1/C0/C1/C2 × seeds 47/49/51，共 12 个运行；最终完成 6/12；
- seed 47/49/51 分别固定物理 GPU 0/1/2；
- SwanLab cloud：project `DeepPro-BC-TPro`，group
  `bc-tpro-stage1-noise8-upstream8fa1a68-fp32-scratch`；
- 日志写入 `log/sem_seg/2026-09-09/` 的独立 upstream 命名目录；
- 队列状态写入
  `log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09/`；
- 结果写入本目录 `metrics/`，不会使用现代化 Stage1 的结果目录。

## 当前 seed47 结果

以下结果来自固定 internal-val16，不是 official test20，因此不能直接与官方 README
数值作同测试集比较。

| 模型 | Pd@0.5 (%) ↑ | Fa@0.5 (×1e-5) ↓ | 27-threshold AUC ↑ |
|---|---:|---:|---:|
| B1 | 78.921569 | 3.227150 | 0.932218228 |
| C0 | 76.540616 | 2.961721 | 0.912537383 |
| C1 | 78.851541 | 2.799961 | 0.971487124 |
| C2 | 78.011204 | 2.959040 | 0.973573062 |

可读汇总见 [Excel](BC_TPRO_STAGE1_SEED47_RESULTS_2026-09-10.xlsx)。当前结果来自固定
internal-val16，不是 official test20，单 seed 也不支持训练随机性或显著性结论。

## 历史 B1 三 seed 结果

| Seed | Pd@0.5 (%) ↑ | Fa@0.5 (×1e-5) ↓ | 27-threshold AUC ↑ |
|---:|---:|---:|---:|
| 47 | 78.9216 | 3.2271 | 0.932218 |
| 49 | 82.0028 | 5.1263 | 0.945569 |
| 51 | 81.5826 | 3.1610 | 0.942659 |
| Mean ± sample SD | 80.8357 ± 1.6709 | 3.8381 ± 1.1160 | 0.940149 ± 0.007020 |

## 原三 seed 命令（历史协议）

只校验并打印 12 条训练和 12 条评测命令：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_stage1_noise8_upstream.sh
```

正式运行入口：

```bash
bash tools/run_bc_tpro_stage1_noise8_upstream.sh
```

12 份结果全部生成后使用同一个分析器的严格 upstream profile：

```bash
python tools/analyze_bc_tpro_noise8_stage1.py \
  --profile upstream8fa1a68_fp32 \
  --experiment-root experiments/bc_tpro_stage1_noise8_upstream_2026-09-09
```

分析器会拒绝 AMP 日志、缺少 `upstream_compat=1`、错误 SwanLab group 或现代化
log_dir，从而不能把两套实验混在一起。

原计划要求 12 份指标后执行候选锁定；由于用户已改为单 seed，该入口不用于当前结果。
后续若重新登记三 seed 协议，selector 必须联合使用三指标 Pareto 判定，并在权衡未决时
fail-closed：

```bash
python tools/analyze_bc_tpro_noise8_paper.py \
  --profile upstream8fa1a68_fp32 \
  --experiment-root experiments/bc_tpro_stage1_noise8_upstream_2026-09-09
```

paper selector 只要求 12 份概率指标、对应训练日志和 epoch-32 checkpoint；不要求
raw-logit 产物。若论文附录需要阈值敏感性分析，可选执行 18 条 upstream FP32
raw-logit 评测（12 primary + 6 B1/C2 repeats）：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_noise8_exact_logit_upstream.sh
bash tools/run_bc_tpro_noise8_exact_logit_upstream.sh
```

训练 provenance、概率指标、checkpoint log_dir 和 paper selector inputs 都绑定同一
profile。upstream 要求 FP32，旧 modernized 默认仍要求 AMP；跨 profile 产物会被拒绝。

## 2026-09-10 官方指标修订

当前检测指标与 TinaLRJ/DeepPro 对齐，只有阈值 0.5 的 Pd、阈值 0.5 的 Fa 和
官方 27 阈值 Pd-Fa AUC 能参与继续门槛、候选资格、排序和锁定。既有 B1 JSON 中的
其他像素诊断字段不会被 active analyzer 读取或汇总。dense-grid 与 raw-logit 输出若
存在，只作为补充敏感性分析，不再是 paper selector 的必需输入。

完整、带时间边界的修订见 `OFFICIAL_METRIC_AMENDMENT_2026-09-10.md`。该指令发生在
三份 B1 已完成、但任何 C0/C1/C2 候选完成训练或产生结果之前。
