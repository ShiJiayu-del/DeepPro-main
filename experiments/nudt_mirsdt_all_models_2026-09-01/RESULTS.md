# NUDT-MIRSDT 全历史模型结果

进度：完成 0/29，运行中 3，失败 0。

排名按验证集最佳 pixel F1；IoU、Precision、Recall 均取自同一个最佳 F1 epoch。

| Rank | Run | Model / Variant | Loss | Status | Epoch | IoU | Precision | Recall | F1 | Final F1 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `deeppro_plus` | DeepPro-Plus | `f1_calibrated_ohem` | running | 14 | 0.850616 | 0.882722 | 0.958995 | 0.919279 | 0.919279 |
| 2 | `deeppro_tdcr` | DeepPro_TDCR | `f1_calibrated_ohem` | running | 12 | 0.792324 | 0.869579 | 0.899176 | 0.884130 | 0.884130 |
| 3 | `deeppro` | DeepPro | `f1_calibrated_ohem` | running | 14 | 0.765676 | 0.847992 | 0.887485 | 0.867290 | 0.867290 |
| - | `deeppro_plus_tdcsta` | DeepPro-Plus_TDCSTA | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `deeppro_plus_moving` | DeepPro-Plus_forMovingScenes | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd1` | DeepPro-Plus_BRTD | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd2` | DeepPro-Plus_BRTD2 | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `feedbacksts` | DeepPro-FeedbackSTS | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `pointcenter` | DeepPro-Plus_BRTD3_PointCenter | `center_consistency_f1` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd` | DeepPro-Plus_BRTD3 / raw_apmd | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_rms` | DeepPro-Plus_BRTD3 / raw_apmd_rms | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_channel_rms` | DeepPro-Plus_BRTD3 / raw_apmd_channel_rms | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_motion_detrend | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_multiscale_contrast | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms_scratch_init` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_init | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_bandpass | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_scratch_detail | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_multiscale_contrast | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | DeepPro-Plus_BRTD3 / raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_second_order` | DeepPro-Plus_BRTD3 / second_order | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_lfp_shallow` | DeepPro-Plus_BRTD3 / lfp_shallow | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_lfp_deep` | DeepPro-Plus_BRTD3 / lfp_deep | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_global_align` | DeepPro-Plus_BRTD3 / global_align | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_local_align` | DeepPro-Plus_BRTD3 / local_align | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_multiscale_head` | DeepPro-Plus_BRTD3 / multiscale_head | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_bidirectional` | DeepPro-Plus_BRTD3 / bidirectional | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |
| - | `brtd3_tdc_dual_stream` | DeepPro-Plus_BRTD3 / tdc_dual_stream | `f1_calibrated_ohem` | pending | - | - | - | - | - | - |

## 解释边界

- 这些是 NUDT-MIRSDT 官方 `test.txt` 划分上的本地 pixel 指标，不是原比赛网站分数。
- 所有任务均从零初始化；PointCenter 使用专用中心损失，跨损失比较需谨慎。
- 未完成任务不参与排名；完整配置见 `manifest.tsv` 与训练目录中的日志。
