# NUDT-MIRSDT 训练筛选分析

> 本文档分析 pixel F1/IoU，仅作为训练诊断和结构初筛；论文主表使用 `PAPER_METRICS.md` 的 Pd/Fa/AUC。

## 完整性与口径

- 清单共 29 项：完成 29，运行中 0，失败 0。
- 主排名采用验证期间最佳 pixel F1；IoU、Precision、Recall 均来自该最佳 F1 的同一轮。
- `Final F1` 是最后一次验证结果，`Best-Final gap` 用来观察训练末期回落。
- 同协议 baseline 是 `deeppro_plus`；所有模型均为随机初始化，不加载预训练权重。

## 总排名

| Rank | Run | Model / Variant | Best epoch | IoU | Precision | Recall | Best F1 | Final F1 | Best-Final gap | ΔF1 vs baseline |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_multiscale_contrast | 20 | 0.870985 | 0.905670 | 0.957881 | 0.931044 | 0.930698 | 0.000346 | +0.008645 |
| 2 | `brtd3_raw_apmd_rms` | DeepPro-Plus_BRTD3 / raw_apmd_rms | 32 | 0.870172 | 0.895963 | 0.967979 | 0.930580 | 0.930580 | 0.000000 | +0.008181 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_multiscale_contrast | 20 | 0.868163 | 0.899303 | 0.961644 | 0.929430 | 0.927001 | 0.002429 | +0.007031 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_init | 20 | 0.866147 | 0.903533 | 0.954407 | 0.928273 | 0.926695 | 0.001578 | +0.005874 |
| 5 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_detail | 30 | 0.865922 | 0.890872 | 0.968670 | 0.928144 | 0.920008 | 0.008136 | +0.005745 |
| 6 | `brtd3_raw_apmd_hybrid_rms` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms | 28 | 0.865690 | 0.885829 | 0.974410 | 0.928011 | 0.926526 | 0.001485 | +0.005612 |
| 7 | `brtd3_raw_apmd_channel_rms` | DeepPro-Plus_BRTD3 / raw_apmd_channel_rms | 26 | 0.865434 | 0.896259 | 0.961778 | 0.927864 | 0.927635 | 0.000229 | +0.005465 |
| 8 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend | 20 | 0.863940 | 0.903280 | 0.952007 | 0.927004 | 0.924371 | 0.002633 | +0.004605 |
| 9 | `brtd3_raw_apmd_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_motion_detrend | 20 | 0.863776 | 0.893239 | 0.963218 | 0.926909 | 0.844848 | 0.082061 | +0.004510 |
| 10 | `brtd3_raw_apmd` | DeepPro-Plus_BRTD3 / raw_apmd | 20 | 0.863021 | 0.903816 | 0.950299 | 0.926475 | 0.911768 | 0.014707 | +0.004076 |
| 11 | `brtd1` | DeepPro-Plus_BRTD | 28 | 0.862926 | 0.884010 | 0.973105 | 0.926420 | 0.925885 | 0.000535 | +0.004021 |
| 12 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast | 24 | 0.862767 | 0.888356 | 0.967691 | 0.926328 | 0.916184 | 0.010144 | +0.003929 |
| 13 | `brtd3_global_align` | DeepPro-Plus_BRTD3 / global_align | 20 | 0.860849 | 0.881609 | 0.973374 | 0.925222 | 0.914915 | 0.010307 | +0.002823 |
| 14 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_bandpass | 20 | 0.859957 | 0.903582 | 0.946843 | 0.924707 | 0.920084 | 0.004623 | +0.002308 |
| 15 | `pointcenter` | DeepPro-Plus_BRTD3_PointCenter / raw_apmd_hybrid_rms | 28 | 0.859536 | 0.888657 | 0.963276 | 0.924463 | 0.924110 | 0.000353 | +0.002064 |
| 16 | `brtd3_tdc_dual_stream` | DeepPro-Plus_BRTD3 / tdc_dual_stream | 24 | 0.857215 | 0.874330 | 0.977674 | 0.923119 | 0.914974 | 0.008145 | +0.000720 |
| 17 | `deeppro_plus` | DeepPro-Plus | 28 | 0.855974 | 0.884364 | 0.963852 | 0.922399 | 0.918659 | 0.003740 | +0.000000 |
| 18 | `brtd3_second_order` | DeepPro-Plus_BRTD3 / second_order | 20 | 0.855854 | 0.879887 | 0.969073 | 0.922329 | 0.917193 | 0.005136 | -0.000070 |
| 19 | `brtd3_lfp_deep` | DeepPro-Plus_BRTD3 / lfp_deep | 20 | 0.855630 | 0.878200 | 0.970839 | 0.922199 | 0.914257 | 0.007942 | -0.000200 |
| 20 | `brtd2` | DeepPro-Plus_BRTD2 | 20 | 0.853575 | 0.880203 | 0.965771 | 0.921004 | 0.919497 | 0.001507 | -0.001395 |
| 21 | `brtd3_local_align` | DeepPro-Plus_BRTD3 / local_align | 32 | 0.851179 | 0.869556 | 0.975773 | 0.919607 | 0.919607 | 0.000000 | -0.002792 |
| 22 | `brtd3_multiscale_head` | DeepPro-Plus_BRTD3 / multiscale_head | 30 | 0.850065 | 0.867983 | 0.976291 | 0.918957 | 0.911493 | 0.007464 | -0.003442 |
| 23 | `brtd3_bidirectional` | DeepPro-Plus_BRTD3 / bidirectional | 10 | 0.841359 | 0.863857 | 0.969976 | 0.913846 | 0.913202 | 0.000644 | -0.008553 |
| 24 | `deeppro_tdcr` | DeepPro_TDCR | 30 | 0.833417 | 0.894088 | 0.924709 | 0.909141 | 0.906851 | 0.002290 | -0.013258 |
| 25 | `deeppro` | DeepPro | 24 | 0.822276 | 0.881122 | 0.924881 | 0.902472 | 0.899597 | 0.002875 | -0.019927 |
| 26 | `deeppro_plus_moving` | DeepPro-Plus_forMovingScenes | 20 | 0.731266 | 0.822814 | 0.867943 | 0.844776 | 0.844495 | 0.000281 | -0.077623 |
| 27 | `deeppro_plus_tdcsta` | DeepPro-Plus_TDCSTA | 14 | 0.471639 | 0.847789 | 0.515271 | 0.640971 | 0.612926 | 0.028045 | -0.281428 |
| 28 | `feedbacksts` | DeepPro-FeedbackSTS | 32 | 0.006394 | 0.011131 | 0.014801 | 0.012706 | 0.012706 | 0.000000 | -0.909693 |
| 29 | `brtd3_lfp_shallow` | DeepPro-Plus_BRTD3 / lfp_shallow | 32 | 0.001215 | 0.001215 | 0.988040 | 0.002427 | 0.002427 | 0.000000 | -0.919972 |

## 关键结论

- 最佳模型为 `brtd3_raw_apmd_hybrid_rms_multiscale_contrast`，Best F1=0.931044，相对 baseline 提升 +0.008645。
- 共 16/29 个有有效指标的模型超过 baseline。
- 最高 Precision 来自 `brtd3_raw_apmd_hybrid_rms_multiscale_contrast`（0.905670）；最高 Recall 来自 `brtd3_lfp_shallow`（0.988040）。
- 最佳与最终 F1 差距最小的是 `brtd3_raw_apmd_rms`（0.000000）。

## 模型族统计

| Family | Count | Mean Best F1 | Median Best F1 | Best run | Best F1 |
|---|---:|---:|---:|---|---:|
| standalone networks | 8 | 0.759986 | 0.905806 | `brtd1` | 0.926420 |
| BRTD3 variants | 21 | 0.881283 | 0.926328 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | 0.931044 |

## BRTD3 结构消融

`brtd3_second_order` 作为 BRTD3 默认二阶结构锚点；下表按相对它的 F1 变化排序。

| Run | Variant | Best F1 | ΔF1 vs second_order | Precision | Recall |
|---|---|---:|---:|---:|---:|
| `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | `raw_apmd_hybrid_rms_multiscale_contrast` | 0.931044 | +0.008715 | 0.905670 | 0.957881 |
| `brtd3_raw_apmd_rms` | `raw_apmd_rms` | 0.930580 | +0.008251 | 0.895963 | 0.967979 |
| `brtd3_raw_apmd_multiscale_contrast` | `raw_apmd_multiscale_contrast` | 0.929430 | +0.007101 | 0.899303 | 0.961644 |
| `brtd3_raw_apmd_hybrid_rms_scratch_init` | `raw_apmd_hybrid_rms_scratch_init` | 0.928273 | +0.005944 | 0.903533 | 0.954407 |
| `brtd3_raw_apmd_hybrid_rms_scratch_detail` | `raw_apmd_hybrid_rms_scratch_detail` | 0.928144 | +0.005815 | 0.890872 | 0.968670 |
| `brtd3_raw_apmd_hybrid_rms` | `raw_apmd_hybrid_rms` | 0.928011 | +0.005682 | 0.885829 | 0.974410 |
| `brtd3_raw_apmd_channel_rms` | `raw_apmd_channel_rms` | 0.927864 | +0.005535 | 0.896259 | 0.961778 |
| `brtd3_raw_apmd_hybrid_rms_motion_detrend` | `raw_apmd_hybrid_rms_motion_detrend` | 0.927004 | +0.004675 | 0.903280 | 0.952007 |
| `brtd3_raw_apmd_motion_detrend` | `raw_apmd_motion_detrend` | 0.926909 | +0.004580 | 0.893239 | 0.963218 |
| `brtd3_raw_apmd` | `raw_apmd` | 0.926475 | +0.004146 | 0.903816 | 0.950299 |
| `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | `raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | 0.926328 | +0.003999 | 0.888356 | 0.967691 |
| `brtd3_global_align` | `global_align` | 0.925222 | +0.002893 | 0.881609 | 0.973374 |
| `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | `raw_apmd_hybrid_rms_scratch_bandpass` | 0.924707 | +0.002378 | 0.903582 | 0.946843 |
| `pointcenter` | `raw_apmd_hybrid_rms` | 0.924463 | +0.002134 | 0.888657 | 0.963276 |
| `brtd3_tdc_dual_stream` | `tdc_dual_stream` | 0.923119 | +0.000790 | 0.874330 | 0.977674 |
| `brtd3_second_order` | `second_order` | 0.922329 | +0.000000 | 0.879887 | 0.969073 |
| `brtd3_lfp_deep` | `lfp_deep` | 0.922199 | -0.000130 | 0.878200 | 0.970839 |
| `brtd3_local_align` | `local_align` | 0.919607 | -0.002722 | 0.869556 | 0.975773 |
| `brtd3_multiscale_head` | `multiscale_head` | 0.918957 | -0.003372 | 0.867983 | 0.976291 |
| `brtd3_bidirectional` | `bidirectional` | 0.913846 | -0.008483 | 0.863857 | 0.969976 |
| `brtd3_lfp_shallow` | `lfp_shallow` | 0.002427 | -0.919902 | 0.001215 | 0.988040 |

## 收敛与选择建议

- 最大训练末期回落为 `brtd3_raw_apmd_motion_detrend`：0.082061。最终部署应使用保存的 best checkpoint，而不是机械采用 latest checkpoint。
- 若以 F1 为唯一主指标，首选 `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` 的 best checkpoint（epoch 20）。
- Precision 与 Recall 的极值来自不同模型时，应避免只看单项指标；F1 排名仍是本轮模型选择依据。
- 单 seed 结果适合筛选结构，但很小的差距仍需多 seed 复验后才能视为稳定改进。
