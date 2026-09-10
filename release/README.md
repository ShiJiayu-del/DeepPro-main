# Reproducible Releases

`release/` 保存历史实验的轻量审计材料。2026-09-04 比赛结束后，提交 ZIP、轨迹
TXT、提交校验记录和对应哈希清单已统一清理；这里继续保留模型权重、源码快照、
训练证据和数值结果，供论文阶段追溯。

| 发布集 | 类型 | 状态 |
|---|---|---|
| [2026-08-29 final submission score 91.30](2026-08-29_final_submission_score91.30_scratch/README.md) | Scratch-only 最终成绩历史快照 | 已清理提交生成物 |
| [2026-08-22 pretrained vs scratch seed47](2026-08-22_pretrained_vs_scratch_seed47/README.md) | 历史配对实验审计 | 历史材料，不得用于新训练初始化 |

发布目录中的源码快照用于复现对应结果，不自动替代仓库根目录的当前实现。
