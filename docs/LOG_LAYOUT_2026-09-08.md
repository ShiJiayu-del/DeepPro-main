# 训练日志目录整理（2026-09-08）

所有原 log/ 下的实际文件已归入 log/sem_seg/；随后移除了四个旧目录入口。
顶层调度日志和锁的旧链接也已移除，脚本直接使用 _pipeline/。
本次是同文件系统移动，不复制权重，不更改训练结果。

## 目录规则

```text
log/sem_seg/
├── 2026-09-01/  # 25 项 NUDT 训练
├── 2026-09-03/  # 12 项 NUDT / Noise8 训练
├── 2026-09-04/  # 21 项 Noise8 训练
├── <已有历史日期>/  # 保留历史实验目录及名称
├── _queues/<批次>/ # status、launcher_logs、failed_attempts、兼容索引
├── _pipeline/      # 跨批次调度日志和锁
└── _archive/       # 原 log/archive 与历史工具归档
```

`reproduction/` 已清理：其权重及两份源码与 release 中的文件 SHA256 一致，
重复副本移入回收站；唯一的 eval_epoch-86.txt 保留在
`_archive/2026-08-31/final_submission_verification/`。
两个确认未被持有的空调度锁已移入回收站，下次运行脚本会在 _pipeline/ 自动创建。
log/ 顶层现在仅保留 sem_seg/。

58 项 NUDT 实验使用：
`<日期>/<数据集>__<YYYY-MM-DD_HH-MM-SS>__<损失标签>-<run_id>_seed49_E32`。
损失标签为 F1OHEM；PointCenter 为 CenterConsistencyF1。
日期取保留下来的训练日志首行时间，不使用迁移日期或目录修改时间。
这解释了为什么 9 月 1 日批次有部分目录在 9 月 3 日，以及 Noise8 部分目录在 9 月 4 日。

## 路径入口

- [干净数据清单](../experiments/nudt_mirsdt_all_models_2026-09-01/manifest.tsv)
- [Noise8 清单](../experiments/nudt_mirsdt_noise8_fjy_all_models_2026-09-03/manifest.tsv)

两份 manifest.tsv 新增 log_dir 列，值为相对 log/sem_seg/ 的路径。
训练、论文评测、结果汇总脚本都读取此列。
SAVE_ROOT 在批次脚本中指 _queues/<批次>（状态和调度产物），
实际传给 train.py/test.py 的根目录为仓库 log/，避免路径逃逸检查拒绝外部符号链接。

旧 log/<批次>/ 入口已移除；_queues/<批次>/sem_seg/<run_id> 仍为兼容索引。
历史日志、SwanLab 配置、checkpoint 元数据中的旧路径原文保持不变。
手动启动 train.py/test.py 时使用规范新路径；不依赖旧链接穿过其路径安全检查。
未带 log_dir 列的外部旧 manifest 保持既有脚本行为；新建批次应显式填写日期命名的 log_dir。

原 experiments/<批次>/paper_metrics/ 整体迁入
log/sem_seg/_queues/<批次>/paper_metrics/，日志位于其中 logs/，不保留旧入口。
论文指标原始 JSON 和状态在运行目录，汇总表继续保存在 experiments。
历史归档已合并为 _archive/2026-09-01/nudt_setup/ 和 _archive/2026-08-31/untracked_tools/。
实验索引见 [experiments/README.md](../experiments/README.md)。

## 核验

- 58 个 best_model.pth 规范路径均可访问；旧根目录链接在后续整理中已移除。
- 移动前后原 log/ 的 4,098 个文件设备号、inode、大小和修改时间逐项一致。
- 两批汇总脚本重新运行成功：各 29/29 done；所有原有 CSV 字段数值不变，增加 log_dir 路径列。
- 58 个训练命令进行了 dry-run，逐项验证路径、loss 和 resume 参数；没有启动训练。
- 四个修改的 shell 脚本通过 bash -n；Python 汇总器通过语法解析。
- 日志目录没有失效符号链接；git diff --check 通过。
- 未重跑模型推理；此前论文指标 JSON 导出错误仍是独立待处理事项。

## 恢复旧布局

使用 manifest 的 run_id 与 log_dir 一一对应即可逆向移动。
先确认无任务运行，再移除对应兼容链接，将规范目录移回批次 sem_seg/<run_id>；
批次及顶层归档同理。不要直接递归删除 log/ 下的链接目标。
本次没有删除原始训练产物，也没有提交或推送 GitHub。
