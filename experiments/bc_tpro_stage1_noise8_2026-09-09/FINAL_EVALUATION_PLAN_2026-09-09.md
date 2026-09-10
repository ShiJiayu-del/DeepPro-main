# Noise8 候选锁定与论文测试计划

登记时间：2026-09-09 14:18 CST。

登记状态：B1 三个随机种子已完成；C0 正在训练；C0/C1/C2 尚无任何内部验证
结果，raw-logit 精确评测尚未运行，官方 test 图像尚未访问。本文件只冻结候选选择和
最终论文测试规则，不修改第一阶段模型、数据划分或训练预算。

## 第一阶段候选锁定

先完成 B1/C0/C1/C2 的 12 个 epoch-32 checkpoint、概率网格评测和 raw-logit
精确工作点评测。每个候选相对同 seed B1 计算配对差。

候选进入最终训练必须同时满足：

1. raw-logit 三 seed 平均 `Pd @ B1-Fa` 下降不超过 1 个百分点；
2. raw-logit 三 seed 平均 `Fa @ B1-Pd` 相对下降大于 0；
3. 至少两个 seed 同时满足 Pd 不下降且 Fa 不增加；
4. 三 seed 平均论文 27 阈值 AUC 不低于 B1；
5. 每个 seed 的验证耗时不超过同 seed B1 的 1.3 倍；
6. checkpoint、训练日志、split 和 scratch-only 身份校验全部通过。

若多个候选合格，依次按平均 AUC、平均 `Fa @ B1-Pd` 相对下降、平均
`Pd @ B1-Fa` 排序；仍相同时选择参数更少的结构。若没有候选合格，B1 保持为
锁定模型，不因单 seed、pixel F1 或单一阈值的偶然优势访问官方 test 选模。

`PROTOCOL_AMENDMENT_2026-09-09.md` 的 C2 门槛只决定是否训练 C3。只有概率网格
和 raw-logit 都通过该门槛，才在同一 64/16 split 上训练 C3 × seeds 47/49/51，
并用上述同一资格与排序规则和 B1/C0/C1/C2 比较。门槛失败则不实现 C3/C4。

## 最终论文对齐训练

候选锁定后，才允许读取官方 test 图像，并执行最终对照：

- B1 和锁定候选分别使用官方 `train.txt` 的全部 80 个序列从随机权重训练；若
  B1 本身胜出，则只训练 B1。
- 每个结构固定 seeds 47/49/51，物理 GPU 分别为 0/1/2；不使用 GPU 3。
- 沿用论文训练协议：32 epochs、global batch 4、Adam、lr 0.001、weight decay
  0.0001、每 10 epochs 乘 0.7、T=40、crop=128、Soft-IoU、无额外几何增强。
- `resume=never`，`base_ckpt/spatial_ckpt/st_ckpt` 为空，禁止任何预训练权重。
- SwanLab cloud 必开；固定使用 epoch 32，不根据官方 test 选择 checkpoint、结构或
  阈值。
- 正式训练开始前排他创建不可覆盖的 `FINAL_PROTOCOL_LOCK.json`；每次官方 test
  前重新计算候选选择和 split，并与该锁逐字段比较，禁止换候选后再次访问 test。
- 官方 `test.txt` 的 20 个序列每个 checkpoint 只评测一次，报告阈值 0.5 的 Pd/Fa、
  显式使用 `split=test`，报告论文 27 阈值 AUC、三 seed mean ± sample SD；pixel
  IoU/F1 仅作补充诊断。
- 本协议不生成或要求 SHA256、MD5 或其他文件内容哈希；身份由绝对路径、精确清单、
  集合关系、训练参数、checkpoint metadata 和逐序列整数计数核对。

## 解释边界

- 第一阶段 64/16 结果用于选模，最终 80/20 结果才可与论文 HiNo 表格并列。
- 本地 `Noise8.0_FJY` 与论文 HiNo 的配置证据相符，但仍应在论文中注明本地数据来源
  和质心标签由 Clean 同名标注回退的实现事实。
- 三个训练 seed 只能量化初步训练波动，不能把不显著差异写成等效性结论。
