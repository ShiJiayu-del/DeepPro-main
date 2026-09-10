# Upstream8fa1a68-FP32 Stage1 候选与 raw-logit 冻结计划

> **2026-09-10 指标修订：** 用户在三个 B1 完成、但任何 C0/C1/C2 候选完成训练或
> 产生结果之前，要求检测评价改为官方 DeepPro 指标。候选门槛、排序与锁定现由
> `OFFICIAL_METRIC_AMENDMENT_2026-09-10.md` 管理，仅使用 Pd@0.5、Fa@0.5 和官方
> 27 阈值 Pd-Fa AUC。下文原 raw-logit 门槛仅保留为历史预注册记录，不再授权或
> 选择模型。

登记日期：2026-09-09。登记状态：upstream profile 尚未启动任何训练或评测，尚无
任何该 profile 的指标。本文件在结果产生前冻结继续、停止和候选排序规则。

## 身份与数据边界

- profile 必须为 `upstream8fa1a68_fp32`，上游锚点为
  `https://github.com/TinaLRJ/DeepPro.git` commit
  `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`。
- 仅接受本目录 manifest 的 B1/C0/C1/C2 × seeds 47/49/51，以及本目录固定
  64/16 split；训练与所有评测均为 FP32。
- checkpoint 必须为 scratch-only 固定 epoch 32，训练日志必须记录
  `upstream_compat=1`、`train_amp=0`、`eval_amp=0`。
- Stage1 只使用 official train80 派生的 internal-val16；official test 清单和图像
  不得用于训练、评测、阈值选择或模型选择。

## 概率网格与 raw-logit 工作点

每个 seed 以同 seed B1 的 logit 0（sigmoid 0.5）定义：

- `F_ref`：B1 的 Fa；候选报告在 `Fa <= F_ref` 下可达到的最大 Pd；
- `P_ref`：B1 的 Pd；候选报告在 `Pd >= P_ref` 下可达到的最小 Fa。

概率结果同时保留论文 27 阈值 AUC和阈值 0.5 的 Pd/Fa。raw-logit 评测必须在
重叠窗口逐像素取最大 logit 后，以完整目标峰值和背景事件求精确工作点，不得因
sigmoid=1 饱和丢失阈值。B1 和 C2 每 seed 各重复一次，重复评测的阈值、逐序列
TP/FP、目标数、像素数必须逐字段一致。

## C2 是否授权 C3

概率网格和 raw-logit 必须一致满足全部条件，才允许在相同 upstream profile、64/16
split、FP32、三 seeds 和固定 epoch-32 下实现并训练 C3：

1. C2 相对同 seed B1 的三-seed平均 `Fa@B1-Pd` 相对下降至少 20%；
2. 三-seed平均 `Pd@B1-Fa` 下降不超过 1 个百分点；
3. 任一 seed 的 Fa 相对下降不得低于 -20%，Pd 差不得低于 -3 个百分点；
4. 任一 seed 的端到端验证耗时不得超过同 seed B1 的 1.3 倍；
5. 每个配对 seed 上均满足 `Fa_C2 <= Fa_C1`、`Pd_C2 >= Pd_C1`，且至少一个指标
   严格改善；
6. B1/C2 的六份重复 raw-logit 计数逐字段一致。

任一条件失败、身份不完整、重复不一致或两个阈值域结论不一致时，结论为 `STOP`，
不实现 C3/C4。不得以单 seed、pixel F1 或单一 AUC 提升替代门槛。

## 最终候选资格和排序

C3 未获授权时，只在 B1/C0/C1/C2 中选择；获授权后把 C3 加入同一规则。候选必须
同时满足：

1. raw-logit 三-seed平均 `Pd@B1-Fa` 下降不超过 1 个百分点；
2. raw-logit 三-seed平均 `Fa@B1-Pd` 相对下降大于 0；
3. 至少两个 seed 同时满足 Pd 不下降且 Fa 不增加；
4. 三-seed平均论文 27 阈值 AUC 不低于 B1；
5. 每个 seed 的验证耗时不超过同 seed B1 的 1.3 倍；
6. split、日志、checkpoint、FP32、upstream-compatible 和 scratch-only 身份全部通过。

多个合格候选依次按平均 AUC 降序、平均 Fa 相对下降降序、平均 Pd 差降序、参数量
升序排序。完全并列则停止并要求预先登记的新规则；无候选合格则锁定 B1。官方 test
在候选唯一锁定前保持封闭。

## 独立输出路径与已冻结入口

upstream exact-logit profile 必须使用：

- 输入：本目录 manifest、splits、metrics 和对应 upstream log_dir；
- 输出：本目录 `exact_logit_metrics/`；
- 队列锁：`log/sem_seg/_queues/bc_tpro_stage1_noise8_upstream_2026-09-09/`；
- 推理：FP32，命令不得含 `--amp`；
- 分析：必须显式接收 `--profile upstream8fa1a68_fp32` 并拒绝现代化 AMP 产物。

入口分别为 `tools/run_bc_tpro_noise8_exact_logit_upstream.sh`、带显式 profile 的
`tools/evaluate_bc_tpro_noise8_exact_logit.py`、
`tools/analyze_bc_tpro_noise8_exact_logit.py` 和
`tools/analyze_bc_tpro_noise8_paper.py`。CPU fail-closed 测试必须在正式训练前保持通过；
不得直接用旧 modernized AMP 入口评测 upstream checkpoint，也不得跳过 raw-logit
阶段选模。

本计划不生成、要求或使用任何文件内容哈希。
