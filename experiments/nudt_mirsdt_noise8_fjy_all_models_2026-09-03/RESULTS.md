# NUDT-MIRSDT-Noise8.0_FJY 训练筛选指标

> 本表的 pixel IoU/F1 只用于训练诊断和结构初筛。论文主结果请使用同目录的 `PAPER_METRICS.md`（Pd/Fa/AUC）。

进度：完成 29/29，运行中 0，失败 0。

筛选顺序按验证集最佳 pixel F1；IoU、Precision、Recall 均取自同一个最佳 F1 epoch。

| Rank | Run | Model / Variant | Loss | Status | Epoch | IoU | Precision | Recall | F1 | Final F1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd` | DeepPro-Plus_BRTD3 / raw_apmd | `f1_calibrated_ohem` | done | 30 | 0.534970 | 0.773943 | 0.634044 | 0.697043 | 0.685803 |
| 2 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_multiscale_contrast | `f1_calibrated_ohem` | done | 16 | 0.528727 | 0.748603 | 0.642875 | 0.691722 | 0.679892 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_multiscale_contrast | `f1_calibrated_ohem` | done | 18 | 0.527798 | 0.740093 | 0.647885 | 0.690927 | 0.663109 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_init | `f1_calibrated_ohem` | done | 26 | 0.524095 | 0.751925 | 0.633660 | 0.687745 | 0.669775 |
| 5 | `brtd2` | DeepPro-Plus_BRTD2 | `f1_calibrated_ohem` | done | 32 | 0.523843 | 0.759501 | 0.628016 | 0.687529 | 0.687529 |
| 6 | `brtd3_global_align` | DeepPro-Plus_BRTD3 / global_align | `f1_calibrated_ohem` | done | 30 | 0.523391 | 0.779654 | 0.614252 | 0.687140 | 0.663028 |
| 7 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend | `f1_calibrated_ohem` | done | 16 | 0.520530 | 0.780373 | 0.609875 | 0.684669 | 0.661606 |
| 8 | `brtd3_lfp_deep` | DeepPro-Plus_BRTD3 / lfp_deep | `f1_calibrated_ohem` | done | 30 | 0.519948 | 0.759439 | 0.622468 | 0.684166 | 0.668832 |
| 9 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_bandpass | `f1_calibrated_ohem` | done | 18 | 0.519227 | 0.759587 | 0.621336 | 0.683541 | 0.670793 |
| 10 | `brtd3_raw_apmd_rms` | DeepPro-Plus_BRTD3 / raw_apmd_rms | `f1_calibrated_ohem` | done | 30 | 0.518968 | 0.755117 | 0.623985 | 0.683317 | 0.667460 |
| 11 | `brtd3_raw_apmd_channel_rms` | DeepPro-Plus_BRTD3 / raw_apmd_channel_rms | `f1_calibrated_ohem` | done | 18 | 0.517257 | 0.748131 | 0.626327 | 0.681832 | 0.671406 |
| 12 | `brtd3_bidirectional` | DeepPro-Plus_BRTD3 / bidirectional | `f1_calibrated_ohem` | done | 32 | 0.516660 | 0.737418 | 0.633142 | 0.681313 | 0.681313 |
| 13 | `brtd3_local_align` | DeepPro-Plus_BRTD3 / local_align | `f1_calibrated_ohem` | done | 18 | 0.515777 | 0.703589 | 0.658962 | 0.680545 | 0.674194 |
| 14 | `brtd3_raw_apmd_hybrid_rms` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms | `f1_calibrated_ohem` | done | 18 | 0.511704 | 0.756327 | 0.612716 | 0.676989 | 0.661510 |
| 15 | `brtd3_raw_apmd_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_motion_detrend | `f1_calibrated_ohem` | done | 30 | 0.510982 | 0.742867 | 0.620779 | 0.676358 | 0.664497 |
| 16 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_detail | `f1_calibrated_ohem` | done | 30 | 0.510386 | 0.716987 | 0.639151 | 0.675835 | 0.666384 |
| 17 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast | `f1_calibrated_ohem` | done | 14 | 0.510285 | 0.753697 | 0.612409 | 0.675747 | 0.655911 |
| 18 | `pointcenter` | DeepPro-Plus_BRTD3_PointCenter / raw_apmd_hybrid_rms | `center_consistency_f1` | done | 12 | 0.510066 | 0.713365 | 0.641550 | 0.675554 | 0.671279 |
| 19 | `brtd3_tdc_dual_stream` | DeepPro-Plus_BRTD3 / tdc_dual_stream | `f1_calibrated_ohem` | done | 22 | 0.508881 | 0.685540 | 0.663838 | 0.674515 | 0.659047 |
| 20 | `brtd3_second_order` | DeepPro-Plus_BRTD3 / second_order | `f1_calibrated_ohem` | done | 24 | 0.507830 | 0.730371 | 0.625002 | 0.673591 | 0.662834 |
| 21 | `brtd1` | DeepPro-Plus_BRTD | `f1_calibrated_ohem` | done | 18 | 0.503652 | 0.706677 | 0.636770 | 0.669905 | 0.650446 |
| 22 | `brtd3_multiscale_head` | DeepPro-Plus_BRTD3 / multiscale_head | `f1_calibrated_ohem` | done | 30 | 0.500110 | 0.735332 | 0.609894 | 0.666765 | 0.657727 |
| 23 | `deeppro_plus` | DeepPro-Plus | `f1_calibrated_ohem` | done | 32 | 0.499536 | 0.786052 | 0.578142 | 0.666254 | 0.666254 |
| 24 | `deeppro_plus_moving` | DeepPro-Plus_forMovingScenes | `f1_calibrated_ohem` | done | 18 | 0.475152 | 0.726410 | 0.578718 | 0.644207 | 0.617866 |
| 25 | `deeppro` | DeepPro | `f1_calibrated_ohem` | done | 32 | 0.286919 | 0.573976 | 0.364554 | 0.445900 | 0.445900 |
| 26 | `deeppro_tdcr` | DeepPro_TDCR | `f1_calibrated_ohem` | done | 18 | 0.286312 | 0.477943 | 0.416598 | 0.445167 | 0.442459 |
| 27 | `brtd3_lfp_shallow` | DeepPro-Plus_BRTD3 / lfp_shallow | `f1_calibrated_ohem` | done | 20 | 0.000826 | 0.000826 | 0.798084 | 0.001651 | 0.001545 |
| 28 | `feedbacksts` | DeepPro-FeedbackSTS | `f1_calibrated_ohem` | done | 2 | 0.000389 | 0.000392 | 0.041715 | 0.000778 | 0.000000 |
| 29 | `deeppro_plus_tdcsta` | DeepPro-Plus_TDCSTA | `f1_calibrated_ohem` | done | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 与 baseline 对比

baseline 固定为 `deeppro_plus`（DeepPro-Plus）：它是 BRTD 系列的直接父网络。
对比继续使用上述同一口径；`Δ` 为当前模型减 baseline，正值表示提高。

| Rank | Run | ΔIoU | ΔPrecision | ΔRecall | ΔF1 | ΔFinal F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `brtd3_raw_apmd` | +0.035434 | -0.012109 | +0.055902 | +0.030789 | +0.019549 |
| 2 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | +0.029191 | -0.037449 | +0.064733 | +0.025468 | +0.013638 |
| 3 | `brtd3_raw_apmd_multiscale_contrast` | +0.028262 | -0.045959 | +0.069743 | +0.024673 | -0.003145 |
| 4 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | +0.024559 | -0.034127 | +0.055518 | +0.021491 | +0.003521 |
| 5 | `brtd2` | +0.024307 | -0.026551 | +0.049874 | +0.021275 | +0.021275 |
| 6 | `brtd3_global_align` | +0.023855 | -0.006398 | +0.036110 | +0.020886 | -0.003226 |
| 7 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | +0.020994 | -0.005679 | +0.031733 | +0.018415 | -0.004648 |
| 8 | `brtd3_lfp_deep` | +0.020412 | -0.026613 | +0.044326 | +0.017912 | +0.002578 |
| 9 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | +0.019691 | -0.026465 | +0.043194 | +0.017287 | +0.004539 |
| 10 | `brtd3_raw_apmd_rms` | +0.019432 | -0.030935 | +0.045843 | +0.017063 | +0.001206 |
| 11 | `brtd3_raw_apmd_channel_rms` | +0.017721 | -0.037921 | +0.048185 | +0.015578 | +0.005152 |
| 12 | `brtd3_bidirectional` | +0.017124 | -0.048634 | +0.055000 | +0.015059 | +0.015059 |
| 13 | `brtd3_local_align` | +0.016241 | -0.082463 | +0.080820 | +0.014291 | +0.007940 |
| 14 | `brtd3_raw_apmd_hybrid_rms` | +0.012168 | -0.029725 | +0.034574 | +0.010735 | -0.004744 |
| 15 | `brtd3_raw_apmd_motion_detrend` | +0.011446 | -0.043185 | +0.042637 | +0.010104 | -0.001757 |
| 16 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | +0.010850 | -0.069065 | +0.061009 | +0.009581 | +0.000130 |
| 17 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | +0.010749 | -0.032355 | +0.034267 | +0.009493 | -0.010343 |
| 18 | `pointcenter` | +0.010530 | -0.072687 | +0.063408 | +0.009300 | +0.005025 |
| 19 | `brtd3_tdc_dual_stream` | +0.009345 | -0.100512 | +0.085696 | +0.008261 | -0.007207 |
| 20 | `brtd3_second_order` | +0.008294 | -0.055681 | +0.046860 | +0.007337 | -0.003420 |
| 21 | `brtd1` | +0.004116 | -0.079375 | +0.058628 | +0.003651 | -0.015808 |
| 22 | `brtd3_multiscale_head` | +0.000574 | -0.050720 | +0.031752 | +0.000511 | -0.008527 |
| 23 | `deeppro_plus` | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 24 | `deeppro_plus_moving` | -0.024384 | -0.059642 | +0.000576 | -0.022047 | -0.048388 |
| 25 | `deeppro` | -0.212617 | -0.212076 | -0.213588 | -0.220354 | -0.220354 |
| 26 | `deeppro_tdcr` | -0.213224 | -0.308109 | -0.161544 | -0.221087 | -0.223795 |
| 27 | `brtd3_lfp_shallow` | -0.498710 | -0.785226 | +0.219942 | -0.664603 | -0.664709 |
| 28 | `feedbacksts` | -0.499147 | -0.785660 | -0.536427 | -0.665476 | -0.666254 |
| 29 | `deeppro_plus_tdcsta` | -0.499536 | -0.786052 | -0.578142 | -0.666254 | -0.666254 |

### 当前最佳模型相对 baseline

| Run | IoU | Precision | Recall | F1 | Final F1 |
|---|---:|---:|---:|---:|---:|
| `deeppro_plus` | 0.499536 | 0.786052 | 0.578142 | 0.666254 | 0.666254 |
| `brtd3_raw_apmd` | 0.534970 | 0.773943 | 0.634044 | 0.697043 | 0.685803 |
| 绝对变化 | +0.035434 | -0.012109 | +0.055902 | +0.030789 | +0.019549 |

## 解释边界

- 这些是 NUDT-MIRSDT-Noise8.0_FJY 的 `test.txt` 划分上的本地 pixel 指标，不是原比赛网站分数。
- baseline 为相同数据、seed、训练轮数、损失和阈值协议下的 `DeepPro-Plus`，不是 SatVideoIRSDT_v1 的历史网站 baseline。
- 所有任务均从零初始化；PointCenter 使用专用中心损失，跨损失比较需谨慎。
- 未完成任务不参与排名；完整配置见 `manifest.tsv` 与训练目录中的日志。
