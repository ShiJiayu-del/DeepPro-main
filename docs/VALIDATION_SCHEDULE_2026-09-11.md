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
--early_stopping_patience 0
--run_test_after_train 0
```

训练进程在每个 epoch 结束后运行一次内部验证。这里的验证只记录 validation loss、
pixel IoU、pixel Precision、pixel Recall 和 pixel F1，用于观察优化与过拟合趋势；这些
像素指标不参与 BC-TPro 检测结论或候选排序。`train.py` 仍会按内部 pixel IoU 维护
`best_model.pth` 作为诊断产物，但论文评测固定使用 `epoch_32_model.pth`，不得用该
best checkpoint 替代。

训练仍固定运行到 epoch 32。epoch 32 的内部验证成功后保存
`epoch_32_model.pth`；随后由实验 launcher 启动一次独立 `test.py --epoch 32`，计算并保存
Pd@0.5、Fa@0.5 和官方 27 阈值 Pd-Fa AUC。`run_test_after_train=0` 用于避免 `train.py`
另行启动默认 checkpoint 评测。`early_stopping_patience=0` 表示禁用早停。

`validation_safe_cudnn=1` 只在进程内全分辨率验证期间暂时关闭确定性 cuDNN 算法选择，
验证结束后恢复训练所用的 cuDNN 标志。它不会把内部像素级诊断变成论文检测指标，也不会
改变固定训练轮数。未显式提供内部 train/val 清单时，BC-TPro 不允许开启进程内验证，
以免 loader 默认读取 official test20。

`train.py` 会拒绝不符合上述组合的新 NUDT BC-TPro 命令。当前仓库没有可套用到任意
新结构的通用批量 launcher；注册下一个结构实验时，实验专用 launcher 必须传入这组参数，
并在训练成功后显式调用一次固定 epoch32 的检测评测。

若某次内部验证失败，实验应按失败处理并保留现场。尤其是 epoch 32 验证必须成功，才能
形成该协议要求的固定 epoch32 checkpoint 和后续外部检测指标。

## 已完成实验保持原样

以下已完成实验使用冻结的 external-only 验证环境，不追改训练日志、源码快照、协议或
结果：

- `bc_tpro_stage1_noise8_upstream_2026-09-09`；
- `bc_tpro_nongate_noise8_seed47_2026-09-10`。

它们的训练命令为 `eval_interval=8`、`skip_inprocess_validation=1`、
`early_stopping_patience=0`、`run_test_after_train=0`；由于跳过进程内验证，
`eval_interval=8` 没有触发内部验证。训练完成后均以独立进程对固定
`epoch_32_model.pth` 做一次 internal-val16 Pd/Fa/AUC 评测。这些字段属于既有结果的
provenance，不能改写成新规则。

冻结启动器通过 `CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION=1` 标记历史重放；代码只允许
该标记放行精确的 `eval_interval=8`、skip、safe-cuDNN-off、无早停且无自动测试组合。
这些历史参数的权威证据是既有训练日志中的完整 `Namespace(...)`；早期 JSON 协议文件
并未逐项收录全部验证节奏字段。

## Final80 例外

`bc_tpro_final_noise8_2026-09-09` 使用全部 official train80 训练，没有独立内部验证集，
因此必须继续使用 `skip_inprocess_validation=1` 和 `run_test_after_train=0`。在没有显式
验证清单时开启进程内验证，当前 loader 会默认读取 official `test.txt`，从而使 test20
在每个 epoch 进入训练流程并破坏测试屏障。

若确实需要 final 阶段的逐 epoch 验证，必须另建使用 train-only 划分的新实验协议；该
实验不能再称为使用全部 train80 的 final80 训练。

## 启动前检查

历史 Stage1 曾在确定性训练进程的首次全分辨率验证中遇到 CUDA Xid 31 / illegal memory
access，记录见 [CUDA incident](../experiments/bc_tpro_stage1_2026-09-08/CUDA_INCIDENT.md)。
该事件不等同于已确认的显存不足；TPro 行分块也不能覆盖故障所在的前置全分辨率卷积。
2026-09-11 的验收训练段使用 seed47、FP32、batch4、`sample_rate=0.005`、workers0 且关闭
SwanLab，以缩短训练时间；内部验证仍是未抽样的完整 val16、原始全分辨率输入。因此该
验收覆盖逐 epoch 切换和完整验证显存路径，但不代表 32 epoch 正式训练耗时或数据加载负载。

| 物理 GPU | 训练轮数 | 完整 val16 | 验证后 reserved memory | 固定轮次 checkpoint | 新增 Xid |
|---:|---:|---:|---:|---|---|
| 0 | 2 | 两轮均 16/16 | 两轮均 0.549 GiB | `epoch_2_model.pth` | 无 |
| 1 | 1 | 16/16 | 0.549 GiB | `epoch_1_model.pth` | 无 |
| 2 | 1 | 16/16 | 0.549 GiB | `epoch_1_model.pth` | 无 |

专用 validation DataLoader generator 保证逐 epoch 验证不推进训练随机数状态。GPU0
对照试验中，开启完整验证与跳过验证两组的 epoch2 训练 loss 均为 `0.997440`；checkpoint
中的模型、优化器和 GradScaler state 逐张量完全一致。正式长跑仍应先在目标 GPU 上用
相同数据尺寸、FP32 和确定性训练设置做短 smoke，确认内部验证、checkpoint、显存和
CUDA 状态正常。
