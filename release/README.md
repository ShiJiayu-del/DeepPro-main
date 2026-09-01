# Reproducible Releases

`release/` 保存已经冻结、可审计且允许进入 Git 的小型发布集。普通训练输出应留在
`log/`，只有同时具备来源说明、运行配置、校验值和验证证据的产物才进入本目录。

| 发布集 | 类型 | 状态 |
|---|---|---|
| [2026-08-29 final submission score 91.30](2026-08-29_final_submission_score91.30_scratch/README.md) | Scratch-only 最终提交完整复现 | 当前首选发布 |
| [2026-08-22 pretrained vs scratch seed47](2026-08-22_pretrained_vs_scratch_seed47/README.md) | 历史配对实验审计 | 历史材料，不得用于新训练初始化 |

发布目录中的源码快照用于复现对应结果，不自动替代仓库根目录的当前实现。
