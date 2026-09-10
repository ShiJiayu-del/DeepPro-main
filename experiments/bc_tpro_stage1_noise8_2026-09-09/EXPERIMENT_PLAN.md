# Noise8 专训 BC-TPro 第一阶段预注册

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-09 (Asia/Shanghai)
- Verification Status: CONFIGURATION_VERIFIED; TRAINING_NOT_STARTED
- Version Label: noise8_bc_tpro_stage1_v1

## 研究问题与比较

在论文的高噪声场景设定下，直接用 `NUDT-MIRSDT-Noise8.0_FJY` 从零训练时，
局部多尺度时间证据和中心—环形背景对比是否能在保持检测率的同时降低虚警？

四个模型只改变 BC-TPro adapter：

| 代号 | variant | 目的 |
|---|---|---|
| B1 | `none` | DeepPro-Plus 直接结构基线 |
| C0 | `temporal_control` | 控制新增普通时间分支容量 |
| C1 | `center_multiscale` | 检验中心多尺度零直流响应 |
| C2 | `center_ring` | 检验环形背景参照的增量价值 |

不加入动态门控、低秩动态 TPro、轨迹后处理、额外损失或预训练权重。本轮只能
支持 B1→C0、C0→C1、C1→C2 的机制归因，不能据此声称完整动态 BC-TPro 已验证。

## 数据与隔离

- 唯一训练/内部验证图像根：
  `/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY`。
- 官方 `train.txt`：8,000 条、80 序列；seed `20260908` 按序列固定划为
  64 train / 16 internal val，二者互斥且并集严格等于官方训练序列集合。
- 官方 `test.txt`：2,000 条、20 序列；只读取路径文本验证与 64/16 集合无交集，
  绝不传给 `train.py`/`test.py`，也不读取其图像用于本实验。
- 不使用 Clean 数据训练或验证。Noise8 缺少 `masks_centroid` 时，现有评测加载器
  会读取相邻 Clean 数据集中的同标签派生质心用于 Pd/Fa 记分；模型输入和像素
  mask 仍全部来自 Noise8。该标签元数据回退需在论文方法中披露。
- 不生成、保存或要求 SHA256。启动器使用绝对路径、数据集名、条目/序列计数、
  集合关系、PCG64 划分复算和 manifest 精确行做 fail-closed 校验。

## 固定训练协议

- seeds 47/49/51 分别映射物理 GPU 0/1/2；每个运行单 GPU、global batch 4。
- 32 epochs；Adam；初始学习率 0.001；weight decay 0.0001；每 10 epoch
  学习率乘 0.7。
- 输入 40 帧、128×128 随机裁块、sample rate 0.1；不开序列几何增强。
- Soft-IoU；训练/评测 AMP；确定性训练。
- 因本机已确认的 full-frame deterministic cuDNN Xid 31，训练期间跳过进程内
  全幅验证；固定评测 `epoch_32_model.pth`，并由 fresh、串行 `test.py` 进程
  对 16 个 Noise8 internal-val 序列评测。
- SwanLab cloud 必开；不续训、不自动重试。
- `base_ckpt`、`spatial_ckpt`、`st_ckpt` 均为空，`resume=never`，所有模型
  从随机初始化开始。

## 指标与分析顺序

主指标按 DeepPro-Plus 论文口径：Pd、Fa、Pd-Fa AUC；同时给出固定低虚警预算
下的 `Pd @ fixed Fa` 和固定检测率下的 `Fa @ fixed Pd`。Pixel IoU/F1 只作
诊断。报告每 seed 原始结果、相对同 seed B1 的配对差、三 seeds 均值和样本
标准差；三 seeds 不足以把非显著解释为“无效果”。参数、推理时间和峰值显存
作为资源指标。

在全部 12 个 checkpoint 和 12 个固定 internal-val 结果产生之前不选择赢家，
不访问官方 test，也不自动启动下一阶段结构。
