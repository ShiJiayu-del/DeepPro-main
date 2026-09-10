# CSIG2026 最终提交复现包：Scratch Hybrid-RMS，91.30

本目录记录了 CSIG2026 赛道一最终提交（网站 ID `907655`）的历史结果。
网站于 2026-08-29 22:26 完成评分，最终分数为 **91.30**。上传文件名在网站中
显示为 `submit_hrms_scratch_epoch86_adaptiv.zip`。

> **2026-09-04 清理说明：** 比赛结束后，提交 ZIP、轨迹 TXT 哈希、提交格式校验、
> 轨迹生成日志和发布哈希清单均已删除。最终 checkpoint、源码快照、训练证据、
> 阈值扫描结果和网站成绩记录仍保留。下文中的提交格式仅用于记录历史结果。

## 最终结论

- 模型：`DeepPro-Plus_BRTD3`，结构 `raw_apmd_hybrid_rms`；
- checkpoint：epoch 86，seed 47；
- 初始化：完全随机初始化；
- 预训练权重：**未使用**；
- 普通序列：阈值 `0.16`、最短轨迹 `3` 帧；
- 1280×1024 序列 `000204`、`000205`：阈值 `0.96`、最短轨迹 `4` 帧；
- 最小连通区域：`2`；
- 全验证集自适应轨迹 F1：`0.780460890`；
- 网站最终分数：`91.30`。

## 目录内容

```text
checkpoint/      scratch-only epoch-86 checkpoint
environment/     sjyPID Conda 环境快照
evidence/        训练、推理、历史审计和来源证据
scripts/         已停用的历史提交复现脚本，仅供阅读
source_snapshot/ 训练和提交关键代码的逐文件快照
validation/      全验证集 AMP 阈值扫描及大分辨率专项扫描
```

`scripts/verify_release.sh` 与 `scripts/reproduce_submission.sh` 依赖已删除的提交产物，
现仅作为历史实现参考，不再是可执行验收入口。2026-08-31 的原复现结论仍记录在
`evidence/reproduction_verification.md`。

## 训练来源与可复现边界

`evidence/training.log` 保存了原始训练全过程。日志明确包含：

```text
base_ckpt=''
spatial_ckpt=''
st_ckpt=''
resume='never'
Initialized ... from random weights; no base checkpoint loaded.
Starting a new experiment from scratch.
```

checkpoint 内同时保存了模型配置、模型状态、优化器状态和 early-stopping 状态。
原训练设置 `deterministic=0`，因此重新训练可复现训练方法和配置，但不承诺不同 GPU
上的 checkpoint 逐字节一致；使用已发布 checkpoint 重新生成提交是本目录的主要复现路径。

## 最终 ZIP 不变量

- 220 个顶层 TXT，无子目录；
- 21,285 帧；
- 48,673 个检测点、1,766 条轨迹；
- 帧号连续，字段数合法；
- 坐标顺序为 `x y`，全部位于图像范围；
- 普通轨迹至少 3 帧，大分辨率轨迹至少 4 帧；
- ASCII、无 BOM、LF 换行；
- 单帧最大目标数 12；
- checkpoint SHA256：`63d620dedfeab5a58610b90f7a912176368d3e9e48f402364982a052f70373f4`。

网站分数来自比赛结束时保存的提交页面记录；隐藏测试标签和网站评分器不在本仓库中。
