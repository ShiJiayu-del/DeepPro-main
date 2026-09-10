# NUDT-MIRSDT-Noise8.0_FJY 全历史模型对比（2026-09-03）

本实验在固定 Noise8.0_FJY 数据、训练协议和随机种子的条件下，从零训练与干净
NUDT-MIRSDT 实验完全相同的 29 项模型，用于结构排名和抗噪声对比。

## 数据与协议

- 数据：`/home/user/4T_Storage/SJY/CSIG2026/datasets_v1/NUDT-MIRSDT-Noise8.0_FJY`
- 划分：`train.txt` 8,000 项、`test.txt` 2,000 项；SHA256 与干净数据划分一致
- 完整性：120 个序列、12,000 对 image/mask 文件，文件名一一对应
- 输入：40 帧，训练裁块 128×128，采样率 0.1
- 优化：Adam，学习率 0.005，global batch 4，32 epoch，seed 49
- 验证：阈值 0.5，每 2 epoch 一次，主指标为最佳 pixel F1
- 损失：普通模型为 `f1_calibrated_ohem`；PointCenter 为 `center_consistency_f1`
- 初始化：严格随机初始化，三个 checkpoint 参数均为空，不使用预训练权重
- GPU：物理 GPU 0、1、2；每卡一条串行队列
- SwanLab：project `DeepPro-NUDT-MIRSDT`，group `noise8-fjy-all-models-scratch-seed49`

兼容设置与干净数据实验相同：TDCSTA 使用物理 batch 1、梯度累积 4 和显存分配器
参数；FeedbackSTS 使用 FP32；PointCenter 使用 `raw_apmd_hybrid_rms`。

## 独立运行

```bash
DATA_ROOT=/home/user/4T_Storage/SJY/CSIG2026/datasets_v1/NUDT-MIRSDT-Noise8.0_FJY \
DATASET_NAME=NUDT-MIRSDT-Noise8.0_FJY \
MANIFEST=experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03/manifest.tsv \
SAVE_ROOT=log/sem_seg/_queues/nudt_mirsdt_noise8_fjy_all_models_2026-09-03 \
SWANLAB_GROUP=noise8-fjy-all-models-scratch-seed49 \
bash tools/run_nudt_mirsdt_all_models.sh
```

启动器可安全续跑：已有 `.done` 的任务跳过，有 checkpoint 的未完成任务自动恢复，
无 checkpoint 的失败目录归档后重跑。

## 结果文件

- `results.csv` / `RESULTS.md`：pixel F1/IoU 训练筛选指标，不作为论文主表
- `ANALYSIS.md`：基于训练筛选指标的模型族和稳定性诊断
- `paper_metrics.csv` / `PAPER_METRICS.md`：与 DeepPro-Plus 论文对齐的 Pd、Fa、AUC 主结果

训练权重和原始日志位于：

```text
log/sem_seg/<日期>/<数据集>__<开始时间>__<损失标签>-<run_id>_seed49_E32/
```

精确位置见 `manifest.tsv` 的 `log_dir` 列（相对 `log/sem_seg/`）。
日期来自当前训练日志首条记录；失败重试按保留下来的该次训练开始日期归档。
队列状态和启动日志在 `log/sem_seg/_queues/nudt_mirsdt_noise8_fjy_all_models_2026-09-03/`。
旧根目录入口已移除；批量脚本已读取新路径，
手动训练或评测请使用 `--savepath log` / `--logpath log` 和清单中的 `--log_dir`。
