# Noise8 Stage1 继续门槛补充登记

登记时间：2026-09-09 13:49 CST。

登记状态：三组 B1 正在训练，C0/C1/C2 尚未开始训练，任何候选的 internal-val
结果均尚未产生。本文件只把 `EXPERIMENT_PLAN.md` 中尚未数值化的“保持检测率、
降低虚警”转成可执行门槛，不修改模型、训练预算、数据划分或指标原始值。

## 为什么需要补充

旧 Clean→Noise8 补充实验已经暴露出 float32 sigmoid 在大正 logit 处可能精确
饱和到 1，导致有限概率阈值网格无法可靠解析极低虚警工作点。因此，本批先保留
论文概率阈值口径用于可比报告，但不允许仅凭该网格授权更复杂的 C3。

## 固定工作点

每个训练 seed 单独以 B1 的 logit `0`（sigmoid 概率 `0.5`）定义两个参考量：

- `F_ref`：B1 在该点的 Fa，用于比较候选 `Pd @ Fa <= F_ref`；
- `P_ref`：B1 在该点的 Pd，用于比较候选 `Fa @ Pd >= P_ref`。

候选和同 seed B1 使用相同的 16 个 internal-val 视频。概率网格结果只标为
`PROVISIONAL`；raw-logit 事件阈值评测必须直接在未 sigmoid 的拼接 logits 上
求精确工作点，并复算完整背景计数。官方 test 不参与门槛计算。

## C2 进入 C3 的必要条件

只有 12/12 B1/C0/C1/C2 训练与评测全部通过身份及 scratch-only 检查，并且 C2
同时满足以下全部条件，才允许实现和训练 C3：

1. 相对同 seed B1，三 seed 平均 `Fa @ fixed Pd` 相对下降至少 20%；
2. 相对同 seed B1，三 seed 平均 `Pd @ fixed Fa` 下降不超过 1 个百分点；
3. 任一 seed 的 Fa 相对下降不得低于 -20%，Pd 差不得低于 -3 个百分点；
4. 任一 seed 的端到端验证耗时不得超过同 seed B1 的 1.3 倍；
5. 相对同 seed C1，C2 在每个 seed 的两个固定工作点均不得变差，并须至少
   严格改善其中一个；具体执行为 `Fa_C2 <= Fa_C1` 且 `Pd_C2 >= Pd_C1`；
6. 概率网格与 raw-logit 精确评测对上述门槛给出一致的通过结论。

若任一条件失败，本轮不继续堆叠动态门控或低秩 TPro。可以报告 C0/C1/C2 的
机制结果，但不能把局部最优单 seed、pixel F1 或完整 AUC 单项提升替代上述门槛。

## 复现性与报告边界

- B1 与 C2 的 raw-logit 关键计数各重复一次，并逐字段比较；不使用哈希判断。
- 三 seed 报告 mean ± sample SD；按视频配对 bootstrap 只反映固定训练 seeds 下
  的视频抽样不确定性。
- 论文 27 个概率阈值 AUC、阈值 0.5 的 Pd/Fa、pixel F1 和显存继续保留，但
  pixel F1 只作诊断。
- 本补充登记不生成或要求 SHA256。
