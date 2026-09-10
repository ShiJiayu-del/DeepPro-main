# NUDT-MIRSDT-Noise8.0_FJY 训练筛选分析

> 本文档分析 pixel F1/IoU，仅作为训练诊断和结构初筛；论文主表使用 `PAPER_METRICS.md` 的 Pd/Fa/AUC。

## 完整性与口径

- 清单共 29 项：完成 29，运行中 0，失败 0。
- 筛选顺序采用验证期间最佳 pixel F1；IoU、Precision、Recall 均来自该最佳 F1 的同一轮。
- `Final F1` 是最后一次验证结果，`Best-Final gap` 用来观察训练末期回落。
- 同协议 baseline 是 `deeppro_plus`；所有模型均为随机初始化，不加载预训练权重。

## 总排名

| Rank | Run | Model / Variant | Best epoch | IoU | Precision | Recall | Best F1 | Final F1 | Best-Final gap | ΔF1 vs baseline |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd` | DeepPro-Plus_BRTD3 / raw_apmd | 30 | 0.534970 | 0.773943 | 0.634044 | 0.697043 | 0.685803 | 0.011240 | +0.030789 |
| 2 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_multiscale_contrast | 16 | 0.528727 | 0.748603 | 0.642875 | 0.691722 | 0.679892 | 0.011830 | +0.025468 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_multiscale_contrast | 18 | 0.527798 | 0.740093 | 0.647885 | 0.690927 | 0.663109 | 0.027818 | +0.024673 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_init | 26 | 0.524095 | 0.751925 | 0.633660 | 0.687745 | 0.669775 | 0.017970 | +0.021491 |
| 5 | `brtd2` | DeepPro-Plus_BRTD2 | 32 | 0.523843 | 0.759501 | 0.628016 | 0.687529 | 0.687529 | 0.000000 | +0.021275 |
| 6 | `brtd3_global_align` | DeepPro-Plus_BRTD3 / global_align | 30 | 0.523391 | 0.779654 | 0.614252 | 0.687140 | 0.663028 | 0.024112 | +0.020886 |
| 7 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend | 16 | 0.520530 | 0.780373 | 0.609875 | 0.684669 | 0.661606 | 0.023063 | +0.018415 |
| 8 | `brtd3_lfp_deep` | DeepPro-Plus_BRTD3 / lfp_deep | 30 | 0.519948 | 0.759439 | 0.622468 | 0.684166 | 0.668832 | 0.015334 | +0.017912 |
| 9 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_bandpass | 18 | 0.519227 | 0.759587 | 0.621336 | 0.683541 | 0.670793 | 0.012748 | +0.017287 |
| 10 | `brtd3_raw_apmd_rms` | DeepPro-Plus_BRTD3 / raw_apmd_rms | 30 | 0.518968 | 0.755117 | 0.623985 | 0.683317 | 0.667460 | 0.015857 | +0.017063 |
| 11 | `brtd3_raw_apmd_channel_rms` | DeepPro-Plus_BRTD3 / raw_apmd_channel_rms | 18 | 0.517257 | 0.748131 | 0.626327 | 0.681832 | 0.671406 | 0.010426 | +0.015578 |
| 12 | `brtd3_bidirectional` | DeepPro-Plus_BRTD3 / bidirectional | 32 | 0.516660 | 0.737418 | 0.633142 | 0.681313 | 0.681313 | 0.000000 | +0.015059 |
| 13 | `brtd3_local_align` | DeepPro-Plus_BRTD3 / local_align | 18 | 0.515777 | 0.703589 | 0.658962 | 0.680545 | 0.674194 | 0.006351 | +0.014291 |
| 14 | `brtd3_raw_apmd_hybrid_rms` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms | 18 | 0.511704 | 0.756327 | 0.612716 | 0.676989 | 0.661510 | 0.015479 | +0.010735 |
| 15 | `brtd3_raw_apmd_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_motion_detrend | 30 | 0.510982 | 0.742867 | 0.620779 | 0.676358 | 0.664497 | 0.011861 | +0.010104 |
| 16 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_detail | 30 | 0.510386 | 0.716987 | 0.639151 | 0.675835 | 0.666384 | 0.009451 | +0.009581 |
| 17 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast | 14 | 0.510285 | 0.753697 | 0.612409 | 0.675747 | 0.655911 | 0.019836 | +0.009493 |
| 18 | `pointcenter` | DeepPro-Plus_BRTD3_PointCenter / raw_apmd_hybrid_rms | 12 | 0.510066 | 0.713365 | 0.641550 | 0.675554 | 0.671279 | 0.004275 | +0.009300 |
| 19 | `brtd3_tdc_dual_stream` | DeepPro-Plus_BRTD3 / tdc_dual_stream | 22 | 0.508881 | 0.685540 | 0.663838 | 0.674515 | 0.659047 | 0.015468 | +0.008261 |
| 20 | `brtd3_second_order` | DeepPro-Plus_BRTD3 / second_order | 24 | 0.507830 | 0.730371 | 0.625002 | 0.673591 | 0.662834 | 0.010757 | +0.007337 |
| 21 | `brtd1` | DeepPro-Plus_BRTD | 18 | 0.503652 | 0.706677 | 0.636770 | 0.669905 | 0.650446 | 0.019459 | +0.003651 |
| 22 | `brtd3_multiscale_head` | DeepPro-Plus_BRTD3 / multiscale_head | 30 | 0.500110 | 0.735332 | 0.609894 | 0.666765 | 0.657727 | 0.009038 | +0.000511 |
| 23 | `deeppro_plus` | DeepPro-Plus | 32 | 0.499536 | 0.786052 | 0.578142 | 0.666254 | 0.666254 | 0.000000 | +0.000000 |
| 24 | `deeppro_plus_moving` | DeepPro-Plus_forMovingScenes | 18 | 0.475152 | 0.726410 | 0.578718 | 0.644207 | 0.617866 | 0.026341 | -0.022047 |
| 25 | `deeppro` | DeepPro | 32 | 0.286919 | 0.573976 | 0.364554 | 0.445900 | 0.445900 | 0.000000 | -0.220354 |
| 26 | `deeppro_tdcr` | DeepPro_TDCR | 18 | 0.286312 | 0.477943 | 0.416598 | 0.445167 | 0.442459 | 0.002708 | -0.221087 |
| 27 | `brtd3_lfp_shallow` | DeepPro-Plus_BRTD3 / lfp_shallow | 20 | 0.000826 | 0.000826 | 0.798084 | 0.001651 | 0.001545 | 0.000106 | -0.664603 |
| 28 | `feedbacksts` | DeepPro-FeedbackSTS | 2 | 0.000389 | 0.000392 | 0.041715 | 0.000778 | 0.000000 | 0.000778 | -0.665476 |
| 29 | `deeppro_plus_tdcsta` | DeepPro-Plus_TDCSTA | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.666254 |

## 关键结论

- 最佳模型为 `brtd3_raw_apmd`，Best F1=0.697043，相对 baseline 提升 +0.030789。
- 共 22/29 个有有效指标的模型超过 baseline。
- 最高 Precision 来自 `deeppro_plus`（0.786052）；最高 Recall 来自 `brtd3_lfp_shallow`（0.798084）。
- 最佳与最终 F1 差距最小的是 `brtd2`（0.000000）。

## 模型族统计

| Family | Count | Mean Best F1 | Median Best F1 | Best run | Best F1 |
|---|---:|---:|---:|---|---:|
| standalone networks | 8 | 0.444967 | 0.545053 | `brtd2` | 0.687529 |
| BRTD3 variants | 21 | 0.649094 | 0.681313 | `brtd3_raw_apmd` | 0.697043 |

## BRTD3 结构消融

`brtd3_second_order` 作为 BRTD3 默认二阶结构锚点；下表按相对它的 F1 变化排序。

| Run | Variant | Best F1 | ΔF1 vs second_order | Precision | Recall |
|---|---|---:|---:|---:|---:|
| `brtd3_raw_apmd` | `raw_apmd` | 0.697043 | +0.023452 | 0.773943 | 0.634044 |
| `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | `raw_apmd_hybrid_rms_multiscale_contrast` | 0.691722 | +0.018131 | 0.748603 | 0.642875 |
| `brtd3_raw_apmd_multiscale_contrast` | `raw_apmd_multiscale_contrast` | 0.690927 | +0.017336 | 0.740093 | 0.647885 |
| `brtd3_raw_apmd_hybrid_rms_scratch_init` | `raw_apmd_hybrid_rms_scratch_init` | 0.687745 | +0.014154 | 0.751925 | 0.633660 |
| `brtd3_global_align` | `global_align` | 0.687140 | +0.013549 | 0.779654 | 0.614252 |
| `brtd3_raw_apmd_hybrid_rms_motion_detrend` | `raw_apmd_hybrid_rms_motion_detrend` | 0.684669 | +0.011078 | 0.780373 | 0.609875 |
| `brtd3_lfp_deep` | `lfp_deep` | 0.684166 | +0.010575 | 0.759439 | 0.622468 |
| `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | `raw_apmd_hybrid_rms_scratch_bandpass` | 0.683541 | +0.009950 | 0.759587 | 0.621336 |
| `brtd3_raw_apmd_rms` | `raw_apmd_rms` | 0.683317 | +0.009726 | 0.755117 | 0.623985 |
| `brtd3_raw_apmd_channel_rms` | `raw_apmd_channel_rms` | 0.681832 | +0.008241 | 0.748131 | 0.626327 |
| `brtd3_bidirectional` | `bidirectional` | 0.681313 | +0.007722 | 0.737418 | 0.633142 |
| `brtd3_local_align` | `local_align` | 0.680545 | +0.006954 | 0.703589 | 0.658962 |
| `brtd3_raw_apmd_hybrid_rms` | `raw_apmd_hybrid_rms` | 0.676989 | +0.003398 | 0.756327 | 0.612716 |
| `brtd3_raw_apmd_motion_detrend` | `raw_apmd_motion_detrend` | 0.676358 | +0.002767 | 0.742867 | 0.620779 |
| `brtd3_raw_apmd_hybrid_rms_scratch_detail` | `raw_apmd_hybrid_rms_scratch_detail` | 0.675835 | +0.002244 | 0.716987 | 0.639151 |
| `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | `raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | 0.675747 | +0.002156 | 0.753697 | 0.612409 |
| `pointcenter` | `raw_apmd_hybrid_rms` | 0.675554 | +0.001963 | 0.713365 | 0.641550 |
| `brtd3_tdc_dual_stream` | `tdc_dual_stream` | 0.674515 | +0.000924 | 0.685540 | 0.663838 |
| `brtd3_second_order` | `second_order` | 0.673591 | +0.000000 | 0.730371 | 0.625002 |
| `brtd3_multiscale_head` | `multiscale_head` | 0.666765 | -0.006826 | 0.735332 | 0.609894 |
| `brtd3_lfp_shallow` | `lfp_shallow` | 0.001651 | -0.671940 | 0.000826 | 0.798084 |

## 收敛与选择建议

- 最大训练末期回落为 `brtd3_raw_apmd_multiscale_contrast`：0.027818。最终部署应使用保存的 best checkpoint，而不是机械采用 latest checkpoint。
- 若以 F1 为唯一主指标，首选 `brtd3_raw_apmd` 的 best checkpoint（epoch 30）。
- Precision 与 Recall 的极值来自不同模型时，应避免只看单项指标；F1 排名仍是本轮模型选择依据。
- 单 seed 结果适合筛选结构，但很小的差距仍需多 seed 复验后才能视为稳定改进。

## 跨数据集抗噪对比

以下比较 `NUDT-MIRSDT clean` 与 `NUDT-MIRSDT-Noise8.0_FJY` 的同名模型；共匹配 29 项。

| Noise rank | Run | Clean Best F1 | Noise Best F1 | Noise-Clean ΔF1 | F1 retention | Rank shift |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd` | 0.926475 | 0.697043 | -0.229432 | 75.24% | +9 |
| 2 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | 0.931044 | 0.691722 | -0.239322 | 74.30% | -1 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | 0.929430 | 0.690927 | -0.238503 | 74.34% | +0 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | 0.928273 | 0.687745 | -0.240528 | 74.09% | +0 |
| 5 | `brtd2` | 0.921004 | 0.687529 | -0.233475 | 74.65% | +15 |
| 6 | `brtd3_global_align` | 0.925222 | 0.687140 | -0.238082 | 74.27% | +7 |
| 7 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | 0.927004 | 0.684669 | -0.242335 | 73.86% | +1 |
| 8 | `brtd3_lfp_deep` | 0.922199 | 0.684166 | -0.238033 | 74.19% | +11 |
| 9 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | 0.924707 | 0.683541 | -0.241166 | 73.92% | +5 |
| 10 | `brtd3_raw_apmd_rms` | 0.930580 | 0.683317 | -0.247263 | 73.43% | -8 |
| 11 | `brtd3_raw_apmd_channel_rms` | 0.927864 | 0.681832 | -0.246032 | 73.48% | -4 |
| 12 | `brtd3_bidirectional` | 0.913846 | 0.681313 | -0.232533 | 74.55% | +11 |
| 13 | `brtd3_local_align` | 0.919607 | 0.680545 | -0.239062 | 74.00% | +8 |
| 14 | `brtd3_raw_apmd_hybrid_rms` | 0.928011 | 0.676989 | -0.251022 | 72.95% | -8 |
| 15 | `brtd3_raw_apmd_motion_detrend` | 0.926909 | 0.676358 | -0.250551 | 72.97% | -6 |
| 16 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | 0.928144 | 0.675835 | -0.252309 | 72.82% | -11 |
| 17 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | 0.926328 | 0.675747 | -0.250581 | 72.95% | -5 |
| 18 | `pointcenter` | 0.924463 | 0.675554 | -0.248909 | 73.08% | -3 |
| 19 | `brtd3_tdc_dual_stream` | 0.923119 | 0.674515 | -0.248604 | 73.07% | -3 |
| 20 | `brtd3_second_order` | 0.922329 | 0.673591 | -0.248738 | 73.03% | -2 |
| 21 | `brtd1` | 0.926420 | 0.669905 | -0.256515 | 72.31% | -10 |
| 22 | `brtd3_multiscale_head` | 0.918957 | 0.666765 | -0.252192 | 72.56% | +0 |
| 23 | `deeppro_plus` | 0.922399 | 0.666254 | -0.256145 | 72.23% | -6 |
| 24 | `deeppro_plus_moving` | 0.844776 | 0.644207 | -0.200569 | 76.26% | +2 |
| 25 | `deeppro` | 0.902472 | 0.445900 | -0.456572 | 49.41% | +0 |
| 26 | `deeppro_tdcr` | 0.909141 | 0.445167 | -0.463974 | 48.97% | -2 |
| 27 | `brtd3_lfp_shallow` | 0.002427 | 0.001651 | -0.000776 | 68.03% | +2 |
| 28 | `feedbacksts` | 0.012706 | 0.000778 | -0.011928 | 6.12% | +0 |
| 29 | `deeppro_plus_tdcsta` | 0.640971 | 0.000000 | -0.640971 | 0.00% | -2 |

### 抗噪结论

- 全模型平均 Noise-Clean ΔF1 为 -0.255039。
- F1 保持率最高的是 `deeppro_plus_moving`（76.26%）。
- F1 绝对下降最小的是 `brtd3_lfp_shallow`（-0.000776）。
- `Rank shift` 为正表示在噪声数据中的相对名次上升；它反映相对鲁棒性，不等同于绝对 F1 提高。
