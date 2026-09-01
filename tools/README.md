# Tools

工具保留在单层目录，避免改变已有启动器之间的相对引用。本索引区分当前入口和历史
脚本；历史脚本可能记录旧 GPU 数量或预训练流程，不能直接复制执行。

## 当前 NUDT-MIRSDT 实验

| 文件 | 用途 |
|---|---|
| `project_runtime_env.sh` | 解释器、数据路径和 GPU 0/1/2 白名单 |
| `run_nudt_mirsdt_all_models.sh` | 29 项实验的三队列调度、锁和安全续跑 |
| `summarize_nudt_mirsdt_results.py` | 汇总同 epoch 的 IoU、Precision、Recall、F1 |

## SatVideoIRSDT 最终提交复现

| 文件 | 用途 |
|---|---|
| `run_final_test_hrms_scratch.sh` | 最终 scratch Hybrid-RMS 推理入口 |
| `validate_submission_zip.py` | 提交 ZIP 结构、帧数和内容校验 |
| `centroid_f1_sweep.py` | 质心阈值和面积扫描 |
| `resume_structure_candidate_postprocess.sh` | 带文件锁的后处理恢复 |

最终 91.30 分版本应优先使用
`release/2026-08-29_final_submission_score91.30_scratch/scripts/` 中冻结的脚本。

## 研究与历史工具

其余 `launch_*`、`run_*`、BRTD 检查、概率融合和 SwanLab 侧车脚本用于历史研究或
消融复现。文件名中包含 `8gpu`、`6gpu`、`pretrain` 的脚本只作为审计材料；当前
`train.py` 会拒绝预训练参数，服务器策略也会阻止 GPU 3 及更高编号。
