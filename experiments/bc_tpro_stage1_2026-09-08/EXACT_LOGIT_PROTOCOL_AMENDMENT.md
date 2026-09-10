# BC-TPro raw-logit 精确低虚警评测补充协议（草案）

## 协议地位

这是在查看 109 个 sigmoid 概率阈值的第一阶段结果后新增的 **post-hoc
protocol amendment**，不是原预注册协议的一部分。原
`metrics/*__clean_val.json` 和 `metrics/*__noise8_val.json` 仍是主实验的冻结
结果，不会被本工具修改、覆盖或静默替代。论文中若使用这里的结果，必须将其
明确标为补充的精确工作点评估，并同时披露新增时间和动机。

新增评测解决两个离散阈值问题：固定 109 点可能跳过低虚警区的关键跃迁；在
Noise8 上，B1 的 logit=0（probability=0.5）虚警预算可远高于低虚警 pAUC 的
`5e-5` 上限，不能只凭低虚警 top-K 背景分数声称完整高虚警 ROC 是精确的。

## 精确定义

- 模型、固定 16 序列划分、40 帧输入、epoch 32、AMP 和 Shooting Rules 与
  第一阶段一致；不重新训练，不加载新权重。
- 在 sigmoid 之前取 float32 raw logits。重叠窗口逐元素取最大值；缓冲区以
  `-inf` 初始化，因而负 logit 不会被错误截到 0。
- 目标和虚警区域直接调用 `ShootingRules._prepare_target`：目标检测分数是每个
  连通目标所有像素 3×3 邻域的最大 logit；虚警像素来自原规则定义的
  `box2_map`。
- 比较规则固定为 `score >= threshold`。因此阈值并列时，Pass 2 会把全部并列
  分数一次性计入，不以 top-K 中恰好保留了几个并列元素近似。
- Pass 1 收集全部 target peaks，并仅保留全局最高的
  `floor(5e-5 * P) + 1` 个背景 logits；候选阈值是全部 target peaks、保留的
  background logits、0，以及严格大于全局最大 logit 的有限 sentinel。
- Pass 2 重新推理，并输出每个 threshold × sequence 的整数 TP、FP、targets、
  pixels。每个 stitched sequence 的 float32 字节 SHA256 必须在两遍之间完全
  一致，否则评测立即失败且不写结果。
- 全程最多只在内存中保留一个序列的 stitched logits；不保存预测图或整套数据的
  score maps。

## 精确覆盖边界

默认 `target-events` 模式对下列量是精确的：

1. `Fa <= 5e-5` 内的经验 Pd-Fa 曲线和边界外第一个背景跃迁；
2. `Pd @ B1(logit=0) Fa` 的离散最大值；
3. `Fa @ B1(logit=0) Pd` 的离散最小值。

第 2、3 项无需保留直到 B1 高 Fa 预算的全部背景分数。原因是 TP 只会在某个
target peak 被跨过时改变，而候选集合包含了每一个 target peak；Pass 2 又会在
这些阈值上对所有背景像素精确计数。任何不跨 target peak 的背景阈值变化都不能
改善最大 TP，也不能改变“达到指定 TP”所需的最小 FP。该模式**不覆盖完整
high-Fa ROC 或完整 AUC**，输出元数据会明确记录这一限制。

可选 `full-reference-grid` 模式令
`K=max(floor(cap*P)+1, F_ref+1)`，额外保留直到 B1 虚警预算的每个背景跃迁。
Noise8 的 `F_ref` 可能达到两千至三千万，threshold × sequence 密集整数矩阵会
达到数 GiB。工具会在推理前报告保守内存估算；超过显式安全上限时拒绝运行，
不得通过交换内存或隐藏降采样伪称 exact。

## 两阶段运行和产物

第一阶段先为每个 seed/condition 的 B1 生成 logit=0 的 `F_ref`、`D_ref`；第二
阶段的 C0/C1/C2 必须读取相同 seed、相同 condition 的 B1 JSON，并验证数据内容
身份。运行器默认串行评测，并与原 Stage-1 launcher 共用同一 evaluation lock，
避免本服务器已确认的并发/确定性 cuDNN Conv3d 故障。

只打印命令，不占用 GPU：

```bash
DRY_RUN=1 bash tools/run_bc_tpro_exact_logit.sh
```

所有 12 个 epoch-32 checkpoint 完成后正式执行：

```bash
bash tools/run_bc_tpro_exact_logit.sh
```

输出目录为 `experiments/bc_tpro_stage1_2026-09-08/exact_logit_metrics/`。每次
评测生成一对文件：

- JSON：协议地位、checkpoint/SHA256、代码依赖 SHA、数据与 split 身份、
  AMP/cuDNN flags、模型/variant/seed、内存估算、精确覆盖范围和工作点摘要；
  同时解析训练日志中的 Namespace，核对 seed、variant、epoch 和三个空 checkpoint
  参数，从而留下 scratch-only 的可审计证据；
- NPZ：阈值以及 threshold × sequence 的整数 TP/FP/targets/pixels，并保留
  target peaks 和低虚警 top-K 背景 logits 供独立复核。

CPU 算法测试（不会初始化 CUDA）：

```bash
/home/user/anaconda3/envs/sjyPID/bin/python -m unittest -v \
  tests.test_evaluate_bc_tpro_exact_logit
```

测试覆盖目标峰值与原 Shooting Rules 等价性、全负 logit 的 max stitching、
top-K 分块归并、阈值并列、无目标/sentinel 端点、逐序列两遍整数计数，以及
两遍输出不一致时的强制失败；另用全背景阈值穷举作 oracle，验证默认的
target-event 网格在高 B1-Fa 预算下仍给出相同的两个固定工作点。
