# Experiments

本目录只保存可版本控制的实验定义和汇总，不保存大体积 checkpoint、概率图或 SwanLab
缓存。每个实验使用独立子目录，至少包含协议说明和确定的任务清单；运行产物统一写入
被 Git 忽略的 `log/`。

## 当前实验

| 实验 | 数据集 | 范围 | 状态 |
|---|---|---|---|
| [NUDT-MIRSDT 全历史模型对比](nudt_mirsdt_all_models_2026-09-01/README.md) | NUDT-MIRSDT | 9 个独立网络和 20 个 BRTD3 变体 | 运行中 |

新增实验时不要复用其他实验的 `log_dir`，不要加载 `release/` 中的历史 checkpoint，
并确保启动器经过 `tools/project_runtime_env.sh` 的 GPU 白名单检查。
