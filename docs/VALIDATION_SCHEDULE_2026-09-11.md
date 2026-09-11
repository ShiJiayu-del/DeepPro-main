# BC-TPro 验证节奏（2026-09-11）

本文档是 2026-09-11 起 NUDT-MIRSDT 系列数据集上新建 BC-TPro 实验的验证节奏规则。
代码约束覆盖模型名以 `DeepPro-Plus_BCTPro` 开头的当前模型及其后续派生，并要求实验
同时显式提供互不重叠的内部 `train_sequence_list` 和 `val_sequence_list`。

## 新实验的固定设置

所有符合上述范围的新实验必须显式使用：

```text
--eval_interval 1
--skip_inprocess_validation 0
--validation_safe_cudnn 1
--validation_overlap_policy official_window
--eval_chunk_rows 32
--early_stopping_metric eval_iou
--early_stopping_patience 0
--run_test_after_train 0
```

训练进程在每个 epoch 结束后运行一次完整 internal-val16。`train.py` 按官方 `train.py`
逐滑窗累计 intersection/union：overlap 帧在每个窗口中重复计权，再计算 micro pixel
IoU@0.5。该 IoU 是同一次训练内的 checkpoint 选择指标：IoU 变大时覆盖
`best_model.pth`，精确相等时保留较晚 epoch。validation loss、pixel Precision、pixel
Recall 和 pixel F1 只用于观察优化与过拟合趋势；pixel IoU 也只负责选择该 run 的最佳
checkpoint，不能用来比较不同网络结构的最终优劣。

训练固定运行满 32 epochs，不早停。训练完成后，实验 launcher 独立调用一次不带
`--epoch` 的 `test.py`；该默认路径显式加载 `best_model.pth`，在同一 internal-val16 上计算并
保存 Pd@0.5、Fa@0.5 和官方 27 阈值 Pd-Fa AUC。不同结构之间继续按 Pd 越高、Fa 越低、
AUC 越高的三指标 Pareto 关系综合判断，不采用 AUC 优先或未登记的标量分数。
`run_test_after_train=0` 用于避免 `train.py` 另行启动评测，`early_stopping_patience=0` 表示
禁用早停。

`validation_safe_cudnn=1` 只在进程内全分辨率验证期间暂时关闭确定性 cuDNN 算法选择，
验证结束后恢复训练所用的 cuDNN 标志。它不会把内部像素级诊断变成论文检测指标，也不会
改变固定训练轮数。未显式提供内部 train/val 清单时，BC-TPro 不允许开启进程内验证，
以免 loader 默认读取 official test20。

`validation_overlap_policy=official_window` 固定官方验证计分。旧 `sequence_max` 验证会先
按测试阶段的 max 规则合并重叠帧，不等同于官方训练选 best 的口径，不能用于当前确认性
重跑。`eval_chunk_rows=32` 保留数值等价的行分块安全路径：与官方整图输出只有约 `1e-8`
浮点差。整图 smoke 期间物理 GPU1 被另一训练并发占用且新增 Xid 31，原因存在混杂；在
没有干净现场证明其稳定前，正式长跑只使用已通过三卡验证的分块路径。

当前纠正后的七结构重跑入口是
[`experiments/bc_tpro_stage1_noise8_bestval_seed47_2026-09-11`](../experiments/bc_tpro_stage1_noise8_bestval_seed47_2026-09-11/README.md)
及 `tools/run_bc_tpro_bestval.py`。`train.py` 会拒绝不符合上述组合的新 NUDT BC-TPro 命令；
后续实验 launcher 也必须传入这组参数，并在训练成功后对 `best_model.pth` 独立评测一次。

若某次内部验证失败，实验应按失败处理并保留现场。只有 32 轮训练和每轮验证全部成功，
才能接受 `best_model.pth` 并进行后续检测指标评测。

## 被取代的历史实验

以下已完成实验使用冻结的 external-only 验证环境，不追改训练日志、源码快照或原始
结果，但其固定 epoch32 结果已被本协议取代，只能作为历史证据，不能再称为当前结论：

- `bc_tpro_stage1_noise8_upstream_2026-09-09`；
- `bc_tpro_nongate_noise8_seed47_2026-09-10`。

它们的训练命令为 `eval_interval=8`、`skip_inprocess_validation=1`、
`early_stopping_patience=0`、`run_test_after_train=0`；由于跳过进程内验证，
`eval_interval=8` 没有触发内部验证。训练完成后均以独立进程对固定
`epoch_32_model.pth` 做一次 internal-val16 Pd/Fa/AUC 评测。由于它们没有逐 epoch 验证，
也没有保存可供回溯选择的所有中间 checkpoint，不能事后推导真正的 best。旧 launcher
和这些字段只保留用于精确复现及 provenance 审计，不得用于新的模型结论。

冻结启动器通过 `CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION=1` 标记历史重放；代码只允许
该标记放行精确的 `eval_interval=8`、skip、safe-cuDNN-off、无早停且无自动测试组合。
这些历史参数的权威证据是既有训练日志中的完整 `Namespace(...)`；早期 JSON 协议文件
并未逐项收录全部验证节奏字段。

## Final80 暂停

`bc_tpro_final_noise8_2026-09-09` 使用全部 official train80 训练，没有独立内部验证集，
无法按本规则选择 best checkpoint。因此其“固定 epoch32 后测试”的活动方案已经暂停并
废止，当前不得启动。尤其不能为了逐 epoch 选 best 而在未提供内部验证清单时开启验证；
当前 loader 会默认读取 official `test.txt`，使 test20 进入每个 epoch 并破坏测试屏障。

若将来需要 final 阶段的逐 epoch 验证，必须先登记 train-only 内部划分的新协议；该实验
不能再称为使用全部 train80 的 final80 训练。official test20 不参与当前 Stage1 的逐轮
验证、checkpoint 选择或当前 internal-val16 结构比较。

## 启动前检查

历史 Stage1 曾在确定性训练进程的首次全分辨率验证中遇到 CUDA Xid 31 / illegal memory
access，记录见 [CUDA incident](../experiments/bc_tpro_stage1_2026-09-08/CUDA_INCIDENT.md)。
该事件不等同于已确认的显存不足；TPro 行分块也不能覆盖故障所在的前置全分辨率卷积。
2026-09-11 的验收训练段使用 seed47、FP32、batch4、`sample_rate=0.005`、workers0 且关闭
SwanLab，以缩短训练时间；内部验证仍是未抽样的完整 val16、原始全分辨率输入。因此该
验收覆盖逐 epoch 切换和完整验证显存路径，但不代表 32 epoch 正式训练耗时或数据加载负载。

| 物理 GPU | 训练轮数 | 完整 val16 | overlap policy | best checkpoint | 新增 Xid |
|---:|---:|---:|---|---|---|
| 0 | 2 | 两轮均 16/16 | `official_window` | `best_model.pth` 已按 IoU 更新 | 无 |
| 1 | 1 | 16/16 | `official_window` | `best_model.pth` 已按 IoU 更新 | 无 |
| 2 | 1 | 16/16 | `official_window` | `best_model.pth` 已按 IoU 更新 | 无 |

专用 validation DataLoader generator 保证逐 epoch 验证不推进训练随机数状态。GPU0
对照试验中，开启完整验证与跳过验证两组的 epoch2 训练 loss 均为 `0.997440`；checkpoint
中的模型、优化器和 GradScaler state 逐张量完全一致。正式长跑仍应先在目标 GPU 上用
相同数据尺寸、FP32 和确定性训练设置做短 smoke，确认内部验证、checkpoint、显存和
CUDA 状态正常。

GPU0 的 seed47 两轮端到端 smoke 还确认：两轮均按官方逐窗口规则完成 full val16，
`best_model.pth` 随 IoU 更新；训练后以不带 `--epoch` 的 `test.py` 成功加载该文件，指标
JSON 记录 `checkpoint_epoch=2`。checkpoint 记录 `metric=eval_iou`、`mode=max`、
`overlap_policy=official_window`、`best_epoch=2`，并持久化 `validation_metrics`。三卡分块
smoke 之后未产生新增 Xid。
