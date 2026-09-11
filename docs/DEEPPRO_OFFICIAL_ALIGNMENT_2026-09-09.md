# DeepPro 官方仓库处理对齐复核（更新于 2026-09-11）

## 结论

官方仓库为 <https://github.com/TinaLRJ/DeepPro>。2026-09-11 实时查询确认其 `main` 和
`HEAD` 仍指向 `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`，与本项目锁定锚点一致，
没有上游版本漂移。

当前 Noise8 best-validation 流程已经按该版本重新核对。审计发现旧本地验证先合并重叠
窗口再计算 IoU，而官方 `train.py` 对每个验证滑窗直接累计交并集、重复计入 overlap 帧；
该差异会改变 IoU 数值并可能改变 best epoch。2026-09-11 已停止受影响的刚启动重跑，
并将当前实现改为官方逐窗口聚合规则。最终 `test.py` 仍按官方方式对 sigmoid 概率做
overlap max 拼接后计算 Pd、Fa 和 AUC。

## 逐项结果

| 项目 | 官方处理 | 当前有效处理 | 结论 |
|---|---|---|---|
| 基准模型 | `DeepPro-Plus` | `DeepPro-Plus_BCTPro` 的 `none` 变体 | 参数、state keys、初始化和基础前向一致 |
| 参数量 | 70,913 | B1 为 70,913 | 一致 |
| 基础层 | 官方 `basic.py` | 本地 `basic.py` | 文件内容一致 |
| 输入长度 | 40 | 40 | 一致 |
| 训练裁块 | 128×128 | 128×128 | 一致 |
| 候选训练窗 | 100 帧序列生成 96 个窗，末帧排除 | `upstream_compat=1` | 一致 |
| mask resize | Pillow 默认插值后正像素二值化 | `upstream_compat=1` | 一致 |
| crop 上界 | 起点最大 127 | `upstream_compat=1` | 一致 |
| 优化器 | Adam，lr 0.001，weight decay 1e-4 | 相同 | 一致 |
| LR 调度 | 每 10 epoch 乘 0.7 | 相同 | 一致 |
| 训练轮数 | 32 | 32 | 一致 |
| 损失 | Soft-IoU | `soft_iou` | 前向和梯度一致 |
| 验证频率 | 每个 epoch | 每个 epoch | 一致 |
| 验证 IoU | 每滑窗累计，overlap 帧重复计权 | `validation_overlap_policy=official_window` | 一致 |
| best 判定 | IoU@0.5 最大化，`>=` 时较晚 epoch 覆盖 | 相同并保存 `best_model.pth` | 一致 |
| 最终 checkpoint | `test.py` 默认加载 `best_model.pth` | 启动器不传 `--epoch` | 一致 |
| 推理窗口 | T=40、overlap=4 | 相同 | 一致 |
| 推理 overlap | sigmoid 概率逐像素 max | `SequenceAccumulator` max | 随机输入逐位一致 |
| 论文阈值 | 固定 27 点 | 相同数组与顺序 | 一致 |
| Pd/Fa/AUC | Pd=True/Tgt；Fa=False/总像素；`auc(Fa,Pd)` | 相同 | 一致 |
| SNR 分组 | 按 test 顺序的固定索引 | 按序列名 | 当前清单映射一致 |

B1 在相同 seed 下的全部参数和 BN buffer、FP32 训练前向、Soft-IoU、一次 Adam 更新以及
整图验证前向均与官方对照一致。官方发布的 Noise8 `best_model.pth` 也可以严格加载到本地
B1；其元数据为 0-based epoch 19，即官方实际选择第 20 轮，而不是固定第 32 轮。

## 本次修正

1. `train.py` 默认学习率由错误的 0.005 恢复为官方 0.001；当前启动器此前已显式使用
   0.001，因此旧 active 命令不受该默认值影响。
2. 新增 `validation_overlap_policy=official_window`：验证 checkpoint 的 micro pixel IoU
   按官方逐滑窗累计，重叠帧重复计权；`sequence_max` 只作为非官方诊断选项保留。
3. 当前 best-validation 协议保留 `eval_chunk_rows=32`。该分块路径与官方整图的特征和
   logit 最大差约为 `4.47e-8` 和 `1.68e-8`。整图 smoke 期间物理 GPU1 被另一训练进程
   并发占用且新增 Xid 31，无法把故障单独归因于整图实现，但该现场不足以批准整图长跑；
   当前确认性实验继续使用已通过三卡验证的分块路径。
4. `ShootingRules.evaluate_thresholds()` 在内部把阈值转换到预测数组 dtype，再进行峰值
   比较和排序计数，复现官方逐阈值 float32 比较的边界语义；输出和 AUC 仍使用原始
   float64 27 点阈值。
5. checkpoint 写入 `checkpoint_selection` 和 `validation_metrics`，记录选择指标、overlap
   规则、最佳值和最佳 epoch，供训练后评测与结果汇总交叉核验。

旧 stitched 验证的实测差异并非仅有格式变化：已停止的 B1 审计现场中，epoch4 的官方
逐窗 IoU 为 `0.375341`、旧 stitched IoU 为 `0.380750`；epoch5 分别为 `0.362364` 和
`0.367631`。因此旧启动不能继续，必须使用修正后的代码从随机初始化重新训练。

## 有意保留、且不应“对齐”回官方的差异

| 差异 | 保留原因 |
|---|---|
| train64/internal-val16，而不是官方每轮直接使用 test20 | 防止 official test20 参与 checkpoint 选择 |
| seed47、确定性训练和独立 DataLoader generator | 官方定义了 seed 函数但未调用，worker NumPy seed 依赖时间；当前方案提高可复现性 |
| 验证期间临时使用安全 cuDNN 路径 | 官方/确定性全分辨率组合在本机曾触发 Xid 31；验证后立即恢复训练标志 |
| manifest 安全解析、帧配对检查、原子 checkpoint、源码快照 | 运行安全与审计增强，不改变模型数学定义 |
| BC-TPro 非 `none` adapter | 当前研究变量；零初始化残差保证初始输出等于官方主干，训练后的差异属于显式结构改动 |

这些差异意味着当前协议是“官方模型、数据处理、训练超参数、验证聚合、best 选择和评测
指标语义对齐”，而不是复制官方未固定的随机轨迹，也不是复用官方用 test20 选模的泄漏
行为。

本次 full-frame 与最终 chunk32/official-window smoke 的本地日志、checkpoint 和 dry-run
输出保存在 `log/sem_seg/_alignment_smokes/2026-09-11_official_alignment/`，该目录属于
忽略的运行证据，不进入普通 Git 提交。

## 数据边界

- 本地 Noise8 `train.txt`、`test.txt` 与官方仓库对应清单内容一致；
- 当前 train64/val16 无重叠，并集严格等于 official train80，且与 test20 隔离；
- 本地 12,000 张 Noise8 mask 与 clean NUDT-MIRSDT 同名 mask 一致；
- 官方 Git 仓库不包含 Noise8 图像本体，因此仅凭该仓库不能证明本地噪声图像与官方发布
  压缩包逐文件一致。正式结果仍应配对报告本地 B1，并把论文数字视为外部参考。

## 当前入口

- 协议：
  [`experiments/bc_tpro_stage1_noise8_bestval_seed47_2026-09-11`](../experiments/bc_tpro_stage1_noise8_bestval_seed47_2026-09-11/README.md)
- 启动器：`tools/run_bc_tpro_bestval.py`
- 汇总器：`tools/analyze_bc_tpro_bestval.py`
- official test20：不参与当前 Stage1 的训练、逐轮验证或 checkpoint 选择。
