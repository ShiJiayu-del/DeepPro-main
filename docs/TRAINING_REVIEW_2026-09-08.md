## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-08
- Verification Status: ANALYZED
- Version Label: validation_20260908_v1

## 完成状态

更新：本文记录的是较早的训练筛选核查。随后已完成58项论文指标评测；下文“论文指标未完成”是历史状态。
请以 [论文指标对比](../experiments/PAPER_COMPARISON_2026-09-08.md) 为最新结果。

2026-09-08 现场检查：没有 train.py/test.py 或训练队列进程，GPU 0–3 利用率均为 0%。
以下为现有日志、CSV 和文件核验结果；未重新训练或重跑推理。

| 数据集 | 完成数 | 最后一次验证为 epoch 32 | best_model.pth 存在数 |
|---|---:|---:|---:|
| NUDT-MIRSDT | 29/29 | 29/29 | 29/29 |
| NUDT-MIRSDT-Noise8.0_FJY | 29/29 | 29/29 | 29/29 |

两套训练采用 scratch 初始化、seed 49、40 帧、32 epoch；主损失为 f1_calibrated_ohem，
PointCenter 使用 center_consistency_f1。完整参数以各实验 README、manifest 和运行日志为准。
队列在 2026-09-04 13:14:07 +08:00 记录 PIPELINE_COMPLETE。曾经失败后重试的历史记录保留；
当前 58 项均完成，但完成并不意味着每个模型都学到了有效检测器。

## 同协议 baseline 对比

以下全部是最佳验证轮次的 pixel F1，百分数形式；增量单位为百分点。
它们是训练诊断与结构初筛指标，不等于论文的目标级 Pd/Fa/AUC。

| 模型（BRTD3 前缀简写） | 原始 F1 (%) | 相对 baseline | Noise8 F1 (%) | 相对 baseline |
|---|---:|---:|---:|---:|
| DeepPro-Plus baseline | 92.2399 | — | 66.6254 | — |
| raw_apmd_hybrid_rms_multiscale_contrast | 93.1044 | +0.8645 | 69.1722 | +2.5468 |
| raw_apmd | 92.6475 | +0.4076 | 69.7043 | +3.0789 |
| raw_apmd_rms | 93.0580 | +0.8181 | 68.3317 | +1.7063 |
| raw_apmd_multiscale_contrast | 92.9430 | +0.7031 | 69.0927 | +2.4673 |
| BRTD2 | 92.1004 | -0.1395 | 68.7529 | +2.1275 |

原始数据有 16 个模型超过 baseline，Noise8 有 22 个。
Hybrid-RMS 多尺度对比模型在原始数据第 1、Noise8 第 2，是这次筛选中跨数据集排名最均衡的候选。
简单 Raw-APMD 在 Noise8 第 1，说明本轮实验中增加模块并不总能取得更高噪声数据 F1。
这是各数据集分别训练后的表现，不能据此声称模型未经适配即可跨域泛化。

原始数据冠军相对 baseline 的 Precision 增加 2.1306 个百分点，Recall 下降 0.5971 个百分点；
Noise8 冠军则是 Recall 增加 5.5902 个百分点、Precision 下降 1.2109 个百分点。
两种改善对应不同的误检/漏检权衡，不能仅用 F1 推断目标级检测性能。

## 稳定性与异常

- 原始冠军 Best/Final F1 为 0.931044/0.930698，末期回落很小。
- 原始 raw_apmd_motion_detrend 为 0.926909/0.844848，回落 0.082061。
  单凭此差值无法区分过拟合、优化不稳定和阈值校准漂移，需要结合训练曲线分析。
- Noise8 raw_apmd 为 0.697043/0.685803；Hybrid-RMS 多尺度为 0.691722/0.679892。
  最佳 checkpoint 的表现不能当作最后一轮表现。
- Noise8 BRTD2 Best/Final 均为 0.687529，但单次末轮达到最好值不能证明跨随机种子的稳定性。
- FeedbackSTS、lfp_shallow 接近失效，TDCSTA 在 Noise8 F1 为 0。
  lfp_shallow 的高 Recall 同时伴随极低 Precision，不能解释为有效检测优势。
  这些结果应保留作异常证据，不能只保留高分模型。

## 论文指标未完成

原始数据 PAPER_METRICS.md 仍为 0/29；Noise8 尚无完整论文指标表。
paper_metrics_after_noise8_2026-09-04.log 记录 3 个评测队列失败。
test.py 的 metrics_json 导出代码引用 checkpoint.get('epoch')，
但当前作用域没有可靠的 checkpoint 绑定；此前评测报 UnboundLocalError。
因此训练完成与论文指标评测完成是两个独立状态。

推理日志中的零散 Pd/Fa/AUC 尚未形成完整、成功导出的对照表，本报告不将它们当作最终论文结果。
下一步应修复该导出问题，再完成两套数据的 Pd/Fa/AUC 评测和汇总。

当前训练的 lr=0.005、f1_calibrated_ohem 与论文原协议存在差异。
本地 DeepPro-Plus 是同训练协议的结构 baseline，不能冒充原论文完整复现。
原始 F1 前两名仅相差 0.0464 个百分点；所有结果仅 seed 49，没有多种子方差、置信区间或显著性检验。

## 统计核验

Overall Confidence: CAUTION。11/11 类统计谬误已检查；此覆盖率不是正确性证明。

| 检查项 | 结论 |
|---|---|
| Simpson 悖论 | 两数据集分别呈现；缺少逐序列分层结果，无法充分检验 |
| 生态谬误 | 不从汇总 F1 推断每条序列均改善 |
| Berkson 选择偏差 | 本轮报告覆盖全部 29 个候选；更广泛的数据选择影响未检验 |
| Collider 偏差 | 无相关条件回归分析，不适用 |
| 基率忽视 | 前景稀少，联合解释 Precision/Recall；高 Recall 不能单独代表优秀 |
| 均值回归 | 单种子最佳轮次排名有乐观选择风险，尚无重复实验验证 |
| 幸存者偏差 | 完整保留低分与失败重试记录，不剔除坍塌模型 |
| 多处寻找效应 | 29 结构、多轮筛选；差值是探索性结果，不声称统计显著 |
| 分叉分析路径 | 模型配置已列出，但无预注册或多种子验证；不作确认性结论 |
| 相关等同因果 | 模块与损失、兼容配置存在差异，不能将每个差值归因于单个模块 |
| 反向因果 | 此处没有观察性因果方向推断，不适用 |

另外，如果 test.txt 被同时用于最佳轮次/结构选择和最终测试报告，会产生测试集选择偏差。
正式评估应明确验证集与最终测试集的职责。复现状态为 ANALYZED，尚未进行独立重跑。

## 来源

- [原始完整结果](../experiments/nudt_mirsdt_all_models_2026-09-01/RESULTS.md)
- [原始分析](../experiments/nudt_mirsdt_all_models_2026-09-01/ANALYSIS.md)
- [Noise8 完整结果](../experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03/RESULTS.md)
- [Noise8 分析](../experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03/ANALYSIS.md)
- [论文指标状态](../experiments/nudt_mirsdt_all_models_2026-09-01/PAPER_METRICS.md)
- [清理清单](CLEANUP_2026-09-08.md)
