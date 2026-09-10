# Noise8 专训 raw-logit 固定工作点评测协议

状态：训练尚在进行，尚未产生候选模型的 raw-logit 结果。本文件在揭盲前固定
评测顺序与继续门槛。

## 范围与顺序

- 仅评测本实验 12 个随机初始化、Noise8 直接训练的 `epoch_32_model.pth`。
- 仅使用固定 16 个 internal-val 序列；不读取官方 test 清单或图像。
- 模型输入与像素 mask 均来自 Noise8；目标级 Pd/Fa 使用主实验文档已披露的
  `masks_centroid` 标签元数据回退，不引入额外图像输入。
- 串行执行：先完成三个 seed 的 B1，再按 C0、C1、C2 顺序完成候选。
- B1 与 C2 每个 seed 独立重复一次，共 12 个主评测和 6 个重复评测。
- 共享 `log/sem_seg/_queues/bc_tpro_stage1_noise8_2026-09-09/.evaluation.lock`，
  不与现有全幅评测并发。

## 精确工作点

对重叠窗口先在 raw-logit 域逐像素取最大值。一次 GPU 推理所得 float32 序列
只暂存于评测目录，随后进行两遍 CPU 事件计算：第一遍保留全部目标峰值与低虚警
背景尾部，第二遍在全部事件阈值上计算逐序列整数 TP/FP。两遍复用同一临时数组，
完成后删除临时预测，不受 sigmoid 在 1.0 饱和影响。

- `Pd@B1-Fa`：候选在 FP 不超过同 seed B1 raw-logit=0 的 FP 预算时可达到的
  最大 Pd。
- `Fa@B1-Pd`：候选在 TP 不低于同 seed B1 raw-logit=0 的 TP 目标时可达到的
  最小 Fa。
- 并列时以整数计数决定；阈值只作可追溯说明。

重复评测逐字段比较阈值、逐序列 TP/FP、目标数和像素数，不采用文件指纹。

## C2 门槛

以下条件必须同时满足：

1. C2 三 seed 平均 `Fa@B1-Pd` 相对 B1 下降至少 20%；
2. C2 三 seed 平均 `Pd@B1-Fa` 相对 B1 下降不超过 1 pp；
3. 任一 seed 的 Fa 相对下降不得低于 -20%，Pd 下降不得超过 3 pp；
4. 任一 seed 的推理时间比不得超过 B1 的 1.3 倍；
5. C2 必须在每个配对 seed 上 Pareto 优于 C1，且至少一个主指标严格更好；
6. 六个 B1/C2 重复评测的精确计数必须逐字段一致；
7. 概率网格门槛与 raw-logit 门槛必须同时通过。

任一条件失败或两个阈值域结论不一致时，结论为 `STOP`，不自动启动 C3。

## 执行

```bash
DRY_RUN=1 bash tools/run_bc_tpro_noise8_exact_logit.sh
bash tools/run_bc_tpro_noise8_exact_logit.sh
```

启动器要求 12 个 checkpoint、12 个概率网格结果和 12 个 `.done` 状态全部存在；
因此训练未完成时正式执行会 fail closed。当前配置阶段只运行 dry-run 和 CPU 单测。
