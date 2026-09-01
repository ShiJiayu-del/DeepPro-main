# NUDT-MIRSDT 全历史模型对比（2026-09-01）

本实验将 DeepPro 仓库中保留的全部独立网络，以及
`DeepPro-Plus_BRTD3` 的全部历史结构变体，在 NUDT-MIRSDT 上按统一协议从零训练。

## 数据与比较协议

- 数据：`/home/user/4T_Storage/SJY/CSIG2026/datasets_v1/NUDT-MIRSDT`
- 官方划分：`train.txt`（80 序列、8,000 帧），`test.txt`（20 序列、2,000 帧）
- 输入：40 帧，训练裁块 128×128，采样率 0.1
- 优化：Adam，学习率 0.005，global batch 4，32 epoch，seed 49
- 精度：网络 AMP FP16、损失 FP32
- 验证：阈值 0.5，每 2 epoch 一次，并始终验证最后一轮
- 损失：普通模型统一使用 `f1_calibrated_ohem`；PointCenter 使用其必需的
  `center_consistency_f1`
- 初始化：严格随机初始化，`base_ckpt/spatial_ckpt/st_ckpt` 均为空
- 可视化：SwanLab 项目 `DeepPro-NUDT-MIRSDT`，group
  `all-models-scratch-seed49`
- GPU：只使用物理 GPU 0、1、2；每张卡串行执行一条队列

上述设置固定优化和数据变量，比较的是网络结构；PointCenter 的专用损失是该网络输出
契约的一部分，因此单独记录，不能将其差异仅归因于结构。

## 实验范围

完整的 29 项清单见 [manifest.tsv](manifest.tsv)：9 个非 BRTD3 独立网络，加上
BRTD3 的 20 个结构变体。BRTD3 默认的 `second_order` 已列为一个变体，因此不重复增加
同配置任务。

## 运行

```bash
bash tools/run_nudt_mirsdt_all_models.sh
```

启动器支持安全续跑：成功任务会写入 `.done` 状态；已有 checkpoint 但未完成的任务会用
`--resume auto` 恢复。可用环境变量覆盖非核心运行参数：

```bash
GPU_IDS=0,1,2 SWANLAB_MODE=cloud bash tools/run_nudt_mirsdt_all_models.sh
```

仅查看将要执行的命令：

```bash
DRY_RUN=1 bash tools/run_nudt_mirsdt_all_models.sh
```

训练产物位于：

```text
log/nudt_mirsdt_all_models_2026-09-01/sem_seg/<run_id>/
```

## 汇总

训练期间或训练结束后均可重新生成汇总：

```bash
python tools/summarize_nudt_mirsdt_results.py
```

输出为本目录下的 `results.csv` 和 `RESULTS.md`。排名主指标是每个实验在验证集上达到的
最佳 pixel F1，同时保留同一轮的 IoU、Precision、Recall，以及最终轮指标，避免混用
不同 epoch 的数值。
