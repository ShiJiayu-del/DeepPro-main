# DeepPro 官方仓库对齐说明（2026-09-09）

## 对齐基准

- 官方仓库：<https://github.com/TinaLRJ/DeepPro>
- 固定版本：`8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`
- 当前数据：`/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY`
- 当前实验：`experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/`

启动前再次查询官方 `main`，其版本仍为上述提交。本地 Noise8 的 `train.txt` 和
`test.txt` 与官方仓库中的 NUDT-MIRSDT 清单逐字节一致，分别包含 8000 和 2000 个
帧条目。该事实不证明当前去后缀目录中的噪声图像与官方
`NUDT-MIRSDT-Noise8.0_FJY(g0.15-o1.3)` 发布副本逐文件相同；因此正式结论优先使用
配对本地 B1，并把官方结果标为外部参考。

## 已严格对齐的部分

| 项目 | 官方语义 | 当前实现 |
|---|---|---|
| Baseline | DeepPro-Plus | `DeepPro-Plus_BCTPro` 的 `none` 变体 |
| Baseline 参数 | 70,913 | 70,913 |
| 输入长度 | 40 帧 | 40 帧 |
| 训练裁块 | 128 × 128 | 128 × 128 |
| 优化器 | Adam | Adam |
| 初始学习率 | 0.001 | 0.001 |
| 权重衰减 | 0.0001 | 0.0001 |
| 学习率衰减 | 每 10 epoch 乘 0.7 | 相同 |
| 训练轮数 | 32 | 固定 32 |
| 损失 | Soft-IoU | Soft-IoU |
| 训练精度 | FP32 | FP32 |
| mask resize | Pillow 默认插值后正像素二值化 | `--upstream_compat 1` 时相同 |
| 训练时间窗 | 每个 100 帧序列 96 个候选窗，末帧不进入候选窗 | 相同 |
| crop 上界 | 起点最大 127 | 相同 |

对 B1 baseline 的独立 CPU 审计覆盖三个随机种子：模型 state 项数量、参数量、初始
参数、训练前向、Soft-IoU 以及一次 Adam 更新均与官方 DeepPro-Plus 对齐。新训练的
`base_ckpt`、`spatial_ckpt`、`st_ckpt` 均为空且 `resume=never`，不会加载官方发布权重
或任何其他预训练权重。

## 有意保留的科研约束

当前协议是“官方架构与训练数据语义对齐”，不是复制官方代码中的随机性和测试集选模
行为：

- 官方 train80 固定拆为 train64/internal-val16，用于消融和候选锁定；
- seeds 47、49、51 固定，并启用确定性训练；
- 所有候选使用固定 epoch-32，不按 internal validation 或 official test 挑 epoch；
- Stage1 不读取 official test 图像或标签，只读取 `test.txt` 元数据来验证隔离；
- 正式候选锁定后才允许用 train80 重训，并对 official test20 做一次论文指标评测；
- 推理使用 `eval_chunk_rows=32` 控制显存。它与整图执行数值等价，但浮点舍入可能产生
  极小差异，因此不宣称逐位相同。

这些差异用于避免测试集泄漏并提高复现性，必须在论文实验设置中明确披露。
这里的隔离仅针对当前 BC-TPro 流水线；历史 29 模型实验已评测过同一 test20，所以
final80/test20 不能描述为项目级完全未见的首次外部验证。

## 当前实验矩阵与指标

Stage1 包含 B1/C0/C1/C2 × seeds 47/49/51，共 12 个从零训练运行；seed 47/49/51
分别使用物理 GPU 0/1/2。所有运行通过 SwanLab cloud 记录到 project
`DeepPro-BC-TPro`、group
`bc-tpro-stage1-noise8-upstream8fa1a68-fp32-scratch`。

检测指标与官方论文对齐为 `Pd@0.5`、`Fa@0.5` 和 27 阈值 Pd-Fa AUC。只有这三项
可以进入继续门槛、候选资格、排序、锁定和论文主结果；训练 loss/IoU 仅用于优化诊断。
密集阈值和 raw-logit 精确工作点若执行，只作为补充敏感性分析。2026-09-10 的正式
修订见 `OFFICIAL_METRIC_AMENDMENT_2026-09-10.md`；修订发生在 B1 完成后、任何
C0/C1/C2 候选完成训练或产生指标之前。

官方 README 报告的 Noise 数据参考值为 DeepPro-Plus：Pd 76.23%、Fa 1.69 × 10^-5、
AUC 0.9171。该数值来自官方权重和官方评测流程，不可与 Stage1 的 internal-val16
结果直接作优劣结论；只有最终 train80/test20 阶段才进行同测试划分比较。

## 运行入口与产物

- Stage1：`tools/run_bc_tpro_stage1_noise8_upstream.sh`
- 可选 raw-logit 敏感性分析：`tools/run_bc_tpro_noise8_exact_logit_upstream.sh`
- Stage1 配置与结果：`experiments/bc_tpro_stage1_noise8_upstream_2026-09-09/`
- 模型、训练日志和队列状态：`log/sem_seg/`
- 运行状态：B1 3/3 完成；C0 因当时 SwanLab 代理连接失败而在训练前退出；流水线已停止

Stage1 的 12 份官方指标齐全后即可执行论文候选锁定；raw-logit 不再是锁定前置条件。
若修订后的官方指标门槛要求新增 C3，流程会以 `C3_REQUIRED` 停止，不会提前锁定候选
或访问 official test。

本实验链不生成或校验文件内容哈希。
