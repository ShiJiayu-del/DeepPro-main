# 剩余日志入口清理：2026-09-08

## 删除范围（均通过系统回收站，可恢复）

- `log/.clean_then_noise8_all_models.lock`：旧符号链接。
- `log/.paper_metrics_after_noise8.lock`：旧符号链接。
- `log/clean_then_noise8_all_models_2026-09-03.log`：旧符号链接，目标日志保留。
- `log/noise8_resume_offline_2026-09-04.screen.log`：旧符号链接，目标日志保留。
- `log/paper_metrics_after_noise8_2026-09-04.log`：旧符号链接，目标日志保留。
- `log/sem_seg/_pipeline/` 下上述两个空锁：已实际取得非阻塞独占锁，确认未被其他进程持有后清理。
- `log/sem_seg/reproduction/final_submission_score91p30_release_verification_2026-08-31/`
  中三个重复文件，共 1,036,605 字节：`checkpoints/epoch_86_model.pth`、
  `structure_adapters.py`、`DeepPro-Plus_BRTD3.py`。

重复文件分别与 `release/2026-08-29_final_submission_score91.30_scratch/` 下
`checkpoint/epoch_86_model.pth` 和 `source_snapshot/` 下同名源码进行 SHA256 比对，全部一致。
原目录的唯一评测记录 `eval_epoch-86.txt` 已先移动到
`log/sem_seg/_archive/2026-08-31/final_submission_verification/`，随后删除空 reproduction 父目录。
移入回收站不等于立即释放磁盘空间。

## experiments

现有两批实验的 README、manifest、results、ANALYSIS 和论文指标状态表均有独立用途，保留。
不存在权重或训练日志副本。总入口为 [实验索引](../experiments/README.md)。
0/29 的论文指标表是未完成评测的状态证据，不冒充有效论文结果，也不按低分或失败状态删除。

## 脚本与验证

两个跨批次调度脚本直接使用 `log/sem_seg/_pipeline/` 写日志和创建锁，运行前创建目录。
本次未启动训练或推理；历史日志中的原始命令保持原文。
58 个规范路径下的最佳权重及 release 原始权重均保留。
