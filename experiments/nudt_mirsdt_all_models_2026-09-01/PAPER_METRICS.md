# NUDT-MIRSDT 论文对齐指标

完成 29/29。以下 `Pd` 使用 sigmoid 后阈值 0.5，单位为百分数；`Fa` 单位为 10^-5；`AUC` 来自论文代码采用的预定义阈值组扫描。

排序列 `AUC order` 只是便于阅读；Pd、Fa、AUC 是多目标评价，不定义虚构的综合分数。

## 本地同协议模型

| AUC order | Run | Epoch | Pd | Fa | AUC | Params (M) | Pixel IoU (supp.) | Pixel F1 (supp.) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `deeppro` | 24 | 98.84 | 2.32 | 0.9989 | 0.049 | 0.822276 | 0.902472 |
| 2 | `brtd3_tdc_dual_stream` | 24 | 99.65 | 2.83 | 0.9989 | 0.072 | 0.857215 | 0.923119 |
| 3 | `brtd1` | 28 | 99.83 | 2.60 | 0.9989 | 0.072 | 0.862926 | 0.926420 |
| 4 | `brtd3_multiscale_head` | 30 | 99.77 | 2.95 | 0.9989 | 0.072 | 0.850065 | 0.918957 |
| 5 | `brtd3_raw_apmd_hybrid_rms_scratch_detail` | 30 | 98.61 | 2.74 | 0.9989 | 0.074 | 0.865907 | 0.928135 |
| 6 | `brtd3_raw_apmd_channel_rms` | 26 | 97.40 | 2.71 | 0.9989 | 0.073 | 0.865449 | 0.927872 |
| 7 | `deeppro_tdcr` | 30 | 98.15 | 2.20 | 0.9989 | 0.049 | 0.833417 | 0.909141 |
| 8 | `brtd3_second_order` | 20 | 99.25 | 2.90 | 0.9989 | 0.072 | 0.855854 | 0.922329 |
| 9 | `brtd3_local_align` | 32 | 99.54 | 3.09 | 0.9989 | 0.073 | 0.851179 | 0.919607 |
| 10 | `brtd3_lfp_deep` | 20 | 99.60 | 2.98 | 0.9989 | 0.072 | 0.855630 | 0.922199 |
| 11 | `brtd3_raw_apmd_hybrid_rms` | 28 | 99.42 | 2.72 | 0.9989 | 0.073 | 0.865700 | 0.928016 |
| 12 | `deeppro_plus_moving` | 20 | 99.71 | 2.41 | 0.9989 | 0.129 | 0.731254 | 0.844768 |
| 13 | `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | 20 | 98.50 | 2.61 | 0.9989 | 0.073 | 0.871018 | 0.931063 |
| 14 | `deeppro_plus` | 28 | 99.60 | 2.56 | 0.9989 | 0.071 | 0.855974 | 0.922399 |
| 15 | `brtd3_global_align` | 20 | 99.36 | 2.89 | 0.9989 | 0.073 | 0.860849 | 0.925222 |
| 16 | `brtd3_bidirectional` | 10 | 99.36 | 2.86 | 0.9989 | 0.072 | 0.841359 | 0.913846 |
| 17 | `brtd2` | 20 | 99.71 | 2.89 | 0.9989 | 0.072 | 0.853575 | 0.921004 |
| 18 | `brtd3_raw_apmd_hybrid_rms_motion_detrend` | 20 | 97.57 | 2.67 | 0.9989 | 0.073 | 0.863940 | 0.927004 |
| 19 | `pointcenter` | 28 | 96.65 | 2.72 | 0.9988 | 0.085 | 0.859490 | 0.924436 |
| 20 | `brtd3_raw_apmd_hybrid_rms_scratch_init` | 20 | 97.74 | 2.56 | 0.9987 | 0.073 | 0.866132 | 0.928265 |
| 21 | `brtd3_raw_apmd_multiscale_contrast` | 20 | 98.73 | 2.63 | 0.9987 | 0.073 | 0.868180 | 0.929439 |
| 22 | `brtd3_raw_apmd` | 20 | 97.69 | 2.65 | 0.9986 | 0.073 | 0.863038 | 0.926485 |
| 23 | `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | 24 | 98.38 | 2.80 | 0.9986 | 0.073 | 0.862764 | 0.926327 |
| 24 | `brtd3_raw_apmd_rms` | 32 | 99.25 | 2.64 | 0.9983 | 0.073 | 0.870187 | 0.930588 |
| 25 | `brtd3_raw_apmd_motion_detrend` | 20 | 98.44 | 2.69 | 0.9981 | 0.073 | 0.863756 | 0.926898 |
| 26 | `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | 20 | 95.89 | 2.68 | 0.9971 | 0.074 | 0.859972 | 0.924715 |
| 27 | `deeppro_plus_tdcsta` | 14 | 66.28 | 2.47 | 0.9124 | 0.151 | 0.471614 | 0.640948 |
| 28 | `brtd3_lfp_shallow` | 32 | 99.36 | 31957.25 | 0.8381 | 0.071 | 0.001215 | 0.002427 |
| 29 | `feedbacksts` | 32 | 7.52 | 52.48 | 0.7942 | 5.678 | 0.006356 | 0.012631 |

## 论文已报告的 DeepPro-Plus 参照值

| Source | Pd | Fa | AUC | FP32 model storage (MB)* | GFLOPs/frame (256x256) | FPS (V100) |
|---|---:|---:|---:|---:|---:|---:|
| Li et al., TPAMI 2026 | 99.71 | 2.69 | 0.9978 | 0.284 | 3.89 | 224.05 |

\* 官方表列名容易被理解为 million parameters，但官方 DeepPro-Plus 实际为 70,913
个标量参数；`70,913×4 bytes=0.283652 MB`。此列按 FP32 存储量解释，不能与上表本地
`Params (M)` 直接比较。

## SNR <= 3 子集

| Run | Pd | Fa | AUC (supp.) |
|---|---:|---:|---:|
| `deeppro` | 97.73 | 1.49 | 0.9992 |
| `brtd3_tdc_dual_stream` | 98.87 | 1.75 | 0.9992 |
| `brtd1` | 99.43 | 1.44 | 0.9992 |
| `brtd3_multiscale_head` | 99.24 | 2.00 | 0.9992 |
| `brtd3_raw_apmd_hybrid_rms_scratch_detail` | 95.46 | 1.48 | 0.9992 |
| `brtd3_raw_apmd_channel_rms` | 91.49 | 1.35 | 0.9992 |
| `deeppro_tdcr` | 94.33 | 1.15 | 0.9992 |
| `brtd3_second_order` | 97.54 | 1.86 | 0.9992 |
| `brtd3_local_align` | 98.49 | 2.35 | 0.9991 |
| `brtd3_lfp_deep` | 98.68 | 2.09 | 0.9991 |
| `brtd3_raw_apmd_hybrid_rms` | 98.11 | 1.38 | 0.9991 |
| `deeppro_plus_moving` | 99.24 | 1.40 | 0.9991 |
| `brtd3_raw_apmd_hybrid_rms_multiscale_contrast` | 95.09 | 1.35 | 0.9991 |
| `deeppro_plus` | 98.68 | 1.59 | 0.9990 |
| `brtd3_global_align` | 97.92 | 1.88 | 0.9990 |
| `brtd3_bidirectional` | 97.92 | 1.64 | 0.9990 |
| `brtd2` | 99.05 | 2.20 | 0.9990 |
| `brtd3_raw_apmd_hybrid_rms_motion_detrend` | 92.06 | 1.55 | 0.9989 |
| `pointcenter` | 89.04 | 1.39 | 0.9985 |
| `brtd3_raw_apmd_hybrid_rms_scratch_init` | 92.63 | 1.23 | 0.9984 |
| `brtd3_raw_apmd_multiscale_contrast` | 95.84 | 1.22 | 0.9983 |
| `brtd3_raw_apmd` | 92.44 | 1.32 | 0.9981 |
| `brtd3_raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` | 94.71 | 1.66 | 0.9980 |
| `brtd3_raw_apmd_rms` | 97.54 | 1.38 | 0.9973 |
| `brtd3_raw_apmd_motion_detrend` | 94.90 | 1.45 | 0.9965 |
| `brtd3_raw_apmd_hybrid_rms_scratch_bandpass` | 86.58 | 1.55 | 0.9932 |
| `deeppro_plus_tdcsta` | 47.64 | 1.98 | 0.8694 |
| `brtd3_lfp_shallow` | 97.92 | 31958.74 | 0.8358 |
| `feedbacksts` | 3.21 | 131.99 | 0.7228 |
| DeepPro-Plus (paper) | 99.24 | 1.65 | - |

## 可比性边界

- 检测指标、阈值、SNR 分组和单位与 DeepPro-Plus 论文对齐。
- 本地 29 项使用同一套随机初始化训练协议，适合做受控结构消融。
- 当前本地训练使用 learning rate 0.005 和 `f1_calibrated_ohem`；原论文使用 learning rate 0.001、每 10 epoch 乘 0.7，并使用 Soft-IoU。因此本地数值不能冒充论文官方配置复现，应与论文已报告行分开呈现。
- FPS 与硬件和实现高度相关；论文的 224.05 FPS 来自 V100。未经同硬件复测，不做直接速度优越性声明。
- `Pixel IoU/F1` 仅作为诊断补充，不作为与该论文对齐的主结果。
- 本地 Params 是注册参数个数 M；历史 count_parameters 计算的是存储字节 MB。baseline 为0.070913 M、FP32约0.283652 MB，不能把两种单位混比。
- FeedbackSTS 按其训练验证协议用FP32评测，其余模型使用AMP；FP16异常结果不纳入最终表。
