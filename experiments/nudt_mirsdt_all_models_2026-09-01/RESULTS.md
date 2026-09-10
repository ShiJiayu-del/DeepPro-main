# NUDT-MIRSDT 训练筛选指标

> 本表的 pixel IoU/F1 只用于训练诊断和结构初筛。论文主结果请使用同目录的 `PAPER_METRICS.md`（Pd/Fa/AUC）。

进度：完成 29/29，运行中 0，失败 0。

筛选顺序按验证集最佳 pixel F1；IoU、Precision、Recall 均取自同一个最佳 F1 epoch。

| Rank | Run | Model / Variant | Loss | Status | Epoch | IoU | Precision | Recall | F1 | Final F1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_multiscale_contrast | `f1_calibrated_ohem` | done | 20 | 0.870985 | 0.905670 | 0.957881 | 0.931044 | 0.930698 |
| 2 | `brtd3_raw_apmd_rms` | DeepPro-Plus_BRTD3 / raw_apmd_rms | `f1_calibrated_ohem` | done | 32 | 0.870172 | 0.895963 | 0.967979 | 0.930580 | 0.930580 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_multiscale_contrast | `f1_calibrated_ohem` | done | 20 | 0.868163 | 0.899303 | 0.961644 | 0.929430 | 0.927001 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_init | `f1_calibrated_ohem` | done | 20 | 0.866147 | 0.903533 | 0.954407 | 0.928273 | 0.926695 |
| 5 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_detail | `f1_calibrated_ohem` | done | 30 | 0.865922 | 0.890872 | 0.968670 | 0.928144 | 0.920008 |
| 6 | `brtd3_raw_apmd_hybrid_rms` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms | `f1_calibrated_ohem` | done | 28 | 0.865690 | 0.885829 | 0.974410 | 0.928011 | 0.926526 |
| 7 | `brtd3_raw_apmd_channel_rms` | DeepPro-Plus_BRTD3 / raw_apmd_channel_rms | `f1_calibrated_ohem` | done | 26 | 0.865434 | 0.896259 | 0.961778 | 0.927864 | 0.927635 |
| 8 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend | `f1_calibrated_ohem` | done | 20 | 0.863940 | 0.903280 | 0.952007 | 0.927004 | 0.924371 |
| 9 | `brtd3_raw_apmd_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_motion_detrend | `f1_calibrated_ohem` | done | 20 | 0.863776 | 0.893239 | 0.963218 | 0.926909 | 0.844848 |
| 10 | `brtd3_raw_apmd` | DeepPro-Plus_BRTD3 / raw_apmd | `f1_calibrated_ohem` | done | 20 | 0.863021 | 0.903816 | 0.950299 | 0.926475 | 0.911768 |
| 11 | `brtd1` | DeepPro-Plus_BRTD | `f1_calibrated_ohem` | done | 28 | 0.862926 | 0.884010 | 0.973105 | 0.926420 | 0.925885 |
| 12 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast | `f1_calibrated_ohem` | done | 24 | 0.862767 | 0.888356 | 0.967691 | 0.926328 | 0.916184 |
| 13 | `brtd3_global_align` | DeepPro-Plus_BRTD3 / global_align | `f1_calibrated_ohem` | done | 20 | 0.860849 | 0.881609 | 0.973374 | 0.925222 | 0.914915 |
| 14 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_bandpass | `f1_calibrated_ohem` | done | 20 | 0.859957 | 0.903582 | 0.946843 | 0.924707 | 0.920084 |
| 15 | `pointcenter` | DeepPro-Plus_BRTD3_PointCenter / raw_apmd_hybrid_rms | `center_consistency_f1` | done | 28 | 0.859536 | 0.888657 | 0.963276 | 0.924463 | 0.924110 |
| 16 | `brtd3_tdc_dual_stream` | DeepPro-Plus_BRTD3 / tdc_dual_stream | `f1_calibrated_ohem` | done | 24 | 0.857215 | 0.874330 | 0.977674 | 0.923119 | 0.914974 |
| 17 | `deeppro_plus` | DeepPro-Plus | `f1_calibrated_ohem` | done | 28 | 0.855974 | 0.884364 | 0.963852 | 0.922399 | 0.918659 |
| 18 | `brtd3_second_order` | DeepPro-Plus_BRTD3 / second_order | `f1_calibrated_ohem` | done | 20 | 0.855854 | 0.879887 | 0.969073 | 0.922329 | 0.917193 |
| 19 | `brtd3_lfp_deep` | DeepPro-Plus_BRTD3 / lfp_deep | `f1_calibrated_ohem` | done | 20 | 0.855630 | 0.878200 | 0.970839 | 0.922199 | 0.914257 |
| 20 | `brtd2` | DeepPro-Plus_BRTD2 | `f1_calibrated_ohem` | done | 20 | 0.853575 | 0.880203 | 0.965771 | 0.921004 | 0.919497 |
| 21 | `brtd3_local_align` | DeepPro-Plus_BRTD3 / local_align | `f1_calibrated_ohem` | done | 32 | 0.851179 | 0.869556 | 0.975773 | 0.919607 | 0.919607 |
| 22 | `brtd3_multiscale_head` | DeepPro-Plus_BRTD3 / multiscale_head | `f1_calibrated_ohem` | done | 30 | 0.850065 | 0.867983 | 0.976291 | 0.918957 | 0.911493 |
| 23 | `brtd3_bidirectional` | DeepPro-Plus_BRTD3 / bidirectional | `f1_calibrated_ohem` | done | 10 | 0.841359 | 0.863857 | 0.969976 | 0.913846 | 0.913202 |
| 24 | `deeppro_tdcr` | DeepPro_TDCR | `f1_calibrated_ohem` | done | 30 | 0.833417 | 0.894088 | 0.924709 | 0.909141 | 0.906851 |
| 25 | `deeppro` | DeepPro | `f1_calibrated_ohem` | done | 24 | 0.822276 | 0.881122 | 0.924881 | 0.902472 | 0.899597 |
| 26 | `deeppro_plus_moving` | DeepPro-Plus_forMovingScenes | `f1_calibrated_ohem` | done | 20 | 0.731266 | 0.822814 | 0.867943 | 0.844776 | 0.844495 |
| 27 | `deeppro_plus_tdcsta` | DeepPro-Plus_TDCSTA | `f1_calibrated_ohem` | done | 14 | 0.471639 | 0.847789 | 0.515271 | 0.640971 | 0.612926 |
| 28 | `feedbacksts` | DeepPro-FeedbackSTS | `f1_calibrated_ohem` | done | 32 | 0.006394 | 0.011131 | 0.014801 | 0.012706 | 0.012706 |
| 29 | `brtd3_lfp_shallow` | DeepPro-Plus_BRTD3 / lfp_shallow | `f1_calibrated_ohem` | done | 32 | 0.001215 | 0.001215 | 0.988040 | 0.002427 | 0.002427 |

## 与 baseline 对比

baseline 固定为 `deeppro_plus`（DeepPro-Plus）：它是 BRTD 系列的直接父网络。
对比继续使用上述同一口径；`Δ` 为当前模型减 baseline，正值表示提高。

| Rank | Run | ΔIoU | ΔPrecision | ΔRecall | ΔF1 | ΔFinal F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | +0.015011 | +0.021306 | -0.005971 | +0.008645 | +0.012039 |
| 2 | `brtd3_raw_apmd_rms` | +0.014198 | +0.011599 | +0.004127 | +0.008181 | +0.011921 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | +0.012189 | +0.014939 | -0.002208 | +0.007031 | +0.008342 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | +0.010173 | +0.019169 | -0.009445 | +0.005874 | +0.008036 |
| 5 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | +0.009948 | +0.006508 | +0.004818 | +0.005745 | +0.001349 |
| 6 | `brtd3_raw_apmd_hybrid_rms` | +0.009716 | +0.001465 | +0.010558 | +0.005612 | +0.007867 |
| 7 | `brtd3_raw_apmd_channel_rms` | +0.009460 | +0.011895 | -0.002074 | +0.005465 | +0.008976 |
| 8 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | +0.007966 | +0.018916 | -0.011845 | +0.004605 | +0.005712 |
| 9 | `brtd3_raw_apmd_motion_detrend` | +0.007802 | +0.008875 | -0.000634 | +0.004510 | -0.073811 |
| 10 | `brtd3_raw_apmd` | +0.007047 | +0.019452 | -0.013553 | +0.004076 | -0.006891 |
| 11 | `brtd1` | +0.006952 | -0.000354 | +0.009253 | +0.004021 | +0.007226 |
| 12 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | +0.006793 | +0.003992 | +0.003839 | +0.003929 | -0.002475 |
| 13 | `brtd3_global_align` | +0.004875 | -0.002755 | +0.009522 | +0.002823 | -0.003744 |
| 14 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | +0.003983 | +0.019218 | -0.017009 | +0.002308 | +0.001425 |
| 15 | `pointcenter` | +0.003562 | +0.004293 | -0.000576 | +0.002064 | +0.005451 |
| 16 | `brtd3_tdc_dual_stream` | +0.001241 | -0.010034 | +0.013822 | +0.000720 | -0.003685 |
| 17 | `deeppro_plus` | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 18 | `brtd3_second_order` | -0.000120 | -0.004477 | +0.005221 | -0.000070 | -0.001466 |
| 19 | `brtd3_lfp_deep` | -0.000344 | -0.006164 | +0.006987 | -0.000200 | -0.004402 |
| 20 | `brtd2` | -0.002399 | -0.004161 | +0.001919 | -0.001395 | +0.000838 |
| 21 | `brtd3_local_align` | -0.004795 | -0.014808 | +0.011921 | -0.002792 | +0.000948 |
| 22 | `brtd3_multiscale_head` | -0.005909 | -0.016381 | +0.012439 | -0.003442 | -0.007166 |
| 23 | `brtd3_bidirectional` | -0.014615 | -0.020507 | +0.006124 | -0.008553 | -0.005457 |
| 24 | `deeppro_tdcr` | -0.022557 | +0.009724 | -0.039143 | -0.013258 | -0.011808 |
| 25 | `deeppro` | -0.033698 | -0.003242 | -0.038971 | -0.019927 | -0.019062 |
| 26 | `deeppro_plus_moving` | -0.124708 | -0.061550 | -0.095909 | -0.077623 | -0.074164 |
| 27 | `deeppro_plus_tdcsta` | -0.384335 | -0.036575 | -0.448581 | -0.281428 | -0.305733 |
| 28 | `feedbacksts` | -0.849580 | -0.873233 | -0.949051 | -0.909693 | -0.905953 |
| 29 | `brtd3_lfp_shallow` | -0.854759 | -0.883149 | +0.024188 | -0.919972 | -0.916232 |

### 当前最佳模型相对 baseline

| Run | IoU | Precision | Recall | F1 | Final F1 |
|---|---:|---:|---:|---:|---:|
| `deeppro_plus` | 0.855974 | 0.884364 | 0.963852 | 0.922399 | 0.918659 |
| `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | 0.870985 | 0.905670 | 0.957881 | 0.931044 | 0.930698 |
| 绝对变化 | +0.015011 | +0.021306 | -0.005971 | +0.008645 | +0.012039 |

## 解释边界

- 这些是 NUDT-MIRSDT 的 `test.txt` 划分上的本地 pixel 指标，不是原比赛网站分数。
- baseline 为相同数据、seed、训练轮数、损失和阈值协议下的 `DeepPro-Plus`，不是 SatVideoIRSDT_v1 的历史网站 baseline。
- 所有任务均从零初始化；PointCenter 使用专用中心损失，跨损失比较需谨慎。
- 未完成任务不参与排名；完整配置见 `manifest.tsv` 与训练目录中的日志。
