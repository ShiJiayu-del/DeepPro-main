# BC-TPro 无门控结构消融：Noise8 / seed47

本轮源于用户 2026-09-10 指令：修改网络结构并实际测试；尽量减少门控；只跑一个随机种子；显存允许时多个单卡任务并行；共同追求 Pd 高、Fa 低、AUC 高。

## 设计与假设（训练前登记）

三组均以 C1 的原始帧证据分支为结构参照，保留官方 DeepPro-Plus 主干、post-TPro 加法残差和 3→8→32 投影。新增结构全无可学习乘法门控，投影末层全零初始化，其他网络参数从随机权重训练。保留 C1 的既有 SiLU 激活；没有新增 Sigmoid/Softmax、SE、CBAM 或路由器。每组预计 71,233 个参数，与 C1/C0 相同。

| 代码 | variant | 单一结构变化 | 待检验假设 |
|---|---|---|---|
| NG1 | center_ring_difference | 对 5/9/17 帧中心响应，直接减去对应 11×11 外窗去 5×5 内窗的环形背景时间响应；只保留 3 通道差值 | 固定背景相消是否能减少 Fa，同时保留弱目标 Pd |
| NG2 | temporal_bandpass | 有效帧 mean3 − mean5/9/17，使用实际嵌套窗口系数的 L2 范数归一化 | 窄脉冲噪声平滑与持续目标证据是否取得更好的三指标权衡 |
| NG3 | center_spatial_smooth | 原始帧先做固定十字 3×3 平滑（中心 0.5，四邻居各 0.125），再提取原 C1 时间响应 | 轻度空间平均能否抑制独立噪声，又避免过度摊薄小目标 |

NG1 采用硬编码差值而非 C2 的 9 通道拼接；NG2 边界处以 `sqrt(1/n3 - 1/nK)` 归一化，两个有效窗口相同时输出 0；NG3 使用 replicate 空间边界。三者使用相同的输入 padding 识别，保留已有输出端口。边界归一化和 padding mask 是确定性有效样本处理，不是可学习门控。

等参数不等于同噪声增益：NG1 的空间差分核在内区对白噪声的方差增益为 `1+1/96`，NG3 十字核能量为 `0.3125`。空间平滑也会削弱孤立小目标；这些均是本轮要实际检验的取舍，而非已证实收益。

## 固定实验设置

详情见 `PROTOCOL.json` 与 `manifest.tsv`。Noise8 固定 train64/internal-val16，seed47、FP32、epoch32、T=40、batch=4、crop128、Soft-IoU、Adam、lr=0.001。复用旧 upstream 的 train/val 清单和 B1/C1 seed47 结果。三组在物理 GPU0/1/2 同时训练，评测按锁串行，GPU3禁用。

当前研究是单seed探索；不生成旧三seed schema2 candidate lock。所有模型均报告 Pd@0.5、Fa@0.5、官方27阈值 AUC，以及相对 B1/C1 的三个变化量。三项均非劣且至少一项变好，才称为 Pareto 支配；若不同指标相互冲突，报告权衡，不自定义权重或宣称唯一综合冠军。

## 数据和证据边界

- 图像/mask 来自 `/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY`，评测所需质心依赖相邻 Clean NUDT 的 `masks_centroid`。
- 不读取 official test20 图像。该 test20 已在项目历史实验中使用，不能称为项目级从未见过的外部测试。
- 单seed不给出训练随机性标准差或多seed显著性结论。当前内部验证结果不能直接与论文 test20 表格比较。
- 保存本轮运行清单、命令、权重、训练日志、评测日志和源码快照；不用 SHA256/MD5。
- 不修改旧 B1/C0/C1/C2 结果；新实验使用独立路径。失败保留现场，不自动重训。

## 验收

1. CPU 数学性质、padding、旧变体一致性、初始logits、梯度、非零残差分块一致性。
2. 真实 train64 数据 batch=4、T=40、128×128 的 FP32 前后向显存检查。
3. 三个新run的源码执行闭包、随机初始化、SwanLab、首个epoch启动验收。
4. 每组完整32epoch、固定checkpoint、internal-val16 JSON整数计数与AUC重算；更新结果表。

## 执行入口

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/anaconda3/envs/sjyPID/bin/python tools/run_bc_tpro_nongate.py --dry-run
PYTHONDONTWRITEBYTECODE=1 /home/user/anaconda3/envs/sjyPID/bin/python tools/run_bc_tpro_nongate.py --run
PYTHONDONTWRITEBYTECODE=1 /home/user/anaconda3/envs/sjyPID/bin/python tools/analyze_bc_tpro_nongate.py --xlsx
```

显存预检已记录于 `MEMORY_SMOKE.json`：真实 train64 batch=4、T=40、128×128，两步 FP32 Adam，全部新分支参数数目、loss有限性和投影/上游梯度通过。峰值已分配约 6.404 GiB，最高预留约 9.596 GiB。此测试未保存任何训练权重，不能用于评价检测性能。

## 本轮执行记录

- 2026-09-10 15:18 CST：全仓 CPU 回归 123/123 通过（包含无门控9项、启动器7项），`git diff --check` 通过；原四变体与历史源码快照的非零投影输出和初始化逐位一致。
- 15:18:50 CST：`bc_tpro_nongate_seed47_20260910` Screen 启动 NG1/NG2/NG3。
- 15:19 CST 启动验收：NG1→GPU0/PID2132260，NG2→GPU1/PID2132262，NG3→GPU2/PID2132261；全部 SwanLab 登录成功，614个训练样本/epoch，FP32、scratch、epoch1；每张训练卡约8111 MiB、100%利用率，GPU3没有实验计算进程。
- 日志/状态：`log/sem_seg/_queues/bc_tpro_nongate_noise8_seed47_2026-09-10/`；新启动器使用独立run锁及排他创建日志，完成前验证固定epoch32权重、训练/评测身份和整数计数。
- 评测在各run补齐的`source_snapshot`目录内执行，已验证train/test依赖可全部从快照导入。
- 15:54 CST：NG1/NG2/NG3 全部完成训练、固定 epoch32 评测和产物验证；15:58 生成汇总文件。

## 最终结果

| 模型 | Pd@0.5 (%) ↑ | Fa@0.5 (×10⁻⁵) ↓ | AUC27 ↑ |
|---|---:|---:|---:|
| B1 | 78.921569 | 3.227150 | 0.932218228 |
| C1 | 78.851541 | 2.799961 | 0.971487124 |
| NG1 | 77.731092 | 3.605185 | 0.942356952 |
| NG2 | 77.871148 | 3.416614 | 0.960096460 |
| NG3 | 75.490196 | 2.680206 | 0.941664698 |
| C2 | 78.011204 | 2.959040 | 0.973573062 |

NG1 和 NG2 在 Pd、Fa、AUC27 三项上均被 C1 支配。NG3 的 Fa 最低，但 Pd 比 C1 低
3.361345 个百分点，AUC27 也更低。三个新分支均未综合超过 C1；C1 与 B1、C2、NG3
之间仍存在指标权衡，因此本轮不宣称唯一综合冠军。

完整报告：[RESULTS.md](RESULTS.md)、[results.csv](results.csv)、
[Excel](NG_EXPERIMENT_RESULTS_2026-09-10.xlsx)。
