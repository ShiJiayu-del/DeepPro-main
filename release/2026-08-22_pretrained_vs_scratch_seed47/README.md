# 2026-08-22 Hybrid-RMS 预训练/随机初始化对照实验发布集

> **历史发布集：** 本目录用于审计 2026-08-22 的 pretrained/scratch 配对结果。
> 当前项目已经执行 scratch-only 策略，并且只允许 GPU 0、1、2；不要直接运行本目录
> 记录的旧 8 卡预训练流程，也不要用这里的 checkpoint 初始化新训练。
>
> **2026-09-04 清理说明：** 比赛结束后，各运行目录中的提交 ZIP、
> `selected_submission.json`、提交格式校验 TXT、ZIP 哈希以及发布集总哈希已删除。
> checkpoint、训练日志、阈值扫描数据与结果汇总仍保留，下面涉及已删除文件的文字
> 仅作为历史流程说明。

本目录是从完整实验目录中抽取的历史审计集。清理后包含 8 组实验的最佳后处理
checkpoint、训练与 SwanLab 侧车日志、阈值扫描结果，但不再包含比赛提交生成物、
逐帧概率缓存与所有中间 epoch 权重。

当前模型与损失说明见
[`docs/MODEL_EVOLUTION_ARCHITECTURE_AND_LOSS_2026-08-26.md`](../../docs/MODEL_EVOLUTION_ARCHITECTURE_AND_LOSS_2026-08-26.md)，
文档导航见 [`docs/README.md`](../../docs/README.md)。完整迁移历史保存在
[`docs/MIGRATION_HANDOFF_2026-08-24.md`](../../docs/MIGRATION_HANDOFF_2026-08-24.md)，其中的预训练建议和旧服务器 GPU 配置仅用于审计。

## 核心结论

- 数据集：SatVideoIRSDT_v1。
- 模型主干：DeepPro-Plus_BRTD3。
- 损失：F1CalibratedOHEM。
- 随机种子：47。
- 4 种结构分别进行 pretrained 与 scratch 配对训练，共 8 个任务，对应 GPU 0--7。
- 全局最佳：`raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast` + pretrained。
- 全局最佳选中 epoch：75。
- 全局最佳后处理：threshold=0.17，min_area=2。
- 全局最佳本地代理指标：Precision=0.933339，Recall=0.694785，F1=0.796586。
- 四个结构上 pretrained 均优于 scratch，F1 增益依次为 +0.013183、+0.016950、+0.014857、+0.026110。

这里的 Precision/Recall/F1 是项目后处理器基于本地验证标签计算的 proxy 指标，不应与比赛服务器的官方 Score 直接等同。

## 8 组最终选优结果

| 结构 | 初始化 | 选中 epoch | threshold | min_area | Precision | Recall | F1 | pretrained 相对 scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_rms | pretrained | 24 | 0.10 | 2 | 0.916369 | 0.690557 | 0.787597 | +0.013183 |
| hybrid_rms | scratch | 86 | 0.22 | 2 | 0.948090 | 0.654517 | 0.774414 | 基线 |
| hybrid_rms + motion_detrend | pretrained | 55 | 0.17 | 2 | 0.929883 | 0.685829 | 0.789424 | +0.016950 |
| hybrid_rms + motion_detrend | scratch | 72 | 0.10 | 2 | 0.934858 | 0.658154 | 0.772474 | 基线 |
| hybrid_rms + multiscale_contrast | pretrained | 24 | 0.10 | 2 | 0.923900 | 0.679033 | 0.782763 | +0.014857 |
| hybrid_rms + multiscale_contrast | scratch | 86 | 0.13 | 2 | 0.923623 | 0.657119 | 0.767906 | 基线 |
| hybrid_rms + motion_detrend + multiscale_contrast | pretrained | 75 | 0.17 | 2 | 0.933339 | 0.694785 | **0.796586** | **+0.026110** |
| hybrid_rms + motion_detrend + multiscale_contrast | scratch | 39 | 0.14 | 2 | 0.931505 | 0.656915 | 0.770476 | 基线 |

同一数据也保存在 `results.csv`，便于脚本读取。

## 目录内容

每个 `runs/<run_slug>/` 包含：

- `checkpoint_epoch_<N>.pth`：由全 epoch/阈值扫描最终选中的权重，不是简单复制训练损失最低权重；
- `training.log`：原始训练日志；
- `swanlab_sidecar.log` 与 `swanlab_sidecar.json`：SwanLab 补传过程及状态；
- `sweep_epoch_<N>.json/.csv`：该 epoch 的完整阈值/面积扫描；
- `eval_epoch_<N>.txt`：选中 epoch 的分割评估输出；
- `pipeline.status`：流水线完成记录；

原有 `selected_submission.json`、`submission_validation.txt`、提交 ZIP 和 `.sha256`
均已在 2026-09-04 清理，不再作为当前目录内容。

`source_snapshot/` 保存了全局最佳实验目录内冻结的模型、适配器与损失源码。仓库根目录中的对应源文件是迁移时的工作版本；快照用于确认当次实验实际运行的代码。

## 历史复现边界

该矩阵最初使用 8 张 GPU 分别运行 4 组 pretrained/scratch 配对实验。当前代码有意
阻止加载预训练权重，并将可用 GPU 限制为 0、1、2，因此旧矩阵不再是可执行的当前
训练入口。

复核历史模型结果时可使用本发布集中的冻结源码、选中 checkpoint、训练日志和扫描
JSON/CSV；提交 ZIP 与哈希已删除。新实验使用仓库根目录的 scratch-only 三卡流水线。

## 未纳入 GitHub 的内容

- 数据集与标注；
- Conda 环境本体；
- 每个实验的全量 epoch checkpoint；
- `postprocess/probabilities/` 等逐帧概率缓存；
- `log/`、`swanlog/` 下的其他历史运行和重复生成物；
- 历史迁移压缩包。

仓库内所有压缩包已于 2026-09-04 清理。
