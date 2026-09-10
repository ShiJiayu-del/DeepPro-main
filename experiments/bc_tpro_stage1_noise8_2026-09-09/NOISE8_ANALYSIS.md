# BC-TPro Noise8 专训 Stage1 分析

验证状态：12/12 run 已通过语义身份、scratch、固定划分、checkpoint、评测日志与原始整数计数重算。

概率网格继续门槛：**PROVISIONAL_FAIL**。该结论仅为 `PROVISIONAL`；raw-logit 排序敏感性评测给出一致结论前，不授权 C3。

## 三随机种子 mean ± sample-SD

| Variant | Pd@fixedFa (%) | Fa@fixedPd (×1e-5) | Paper AUC | Pd@0.5 (%) | Fa@0.5 (×1e-5) | Pixel F1 (%) | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 DeepPro-Plus | 78.151 ± 3.370 | 1.277 ± 0.429 | 0.941288 ± 0.013489 | 78.151 ± 3.370 | 1.292 ± 0.442 | 66.382 ± 0.771 | 8.398 ± 0.082 |
| C0 temporal control | 75.444 ± 5.100 | 2.179 ± 1.964 | 0.949142 ± 0.030951 | 80.065 ± 4.035 | 1.679 ± 0.858 | 66.596 ± 0.890 | 8.744 ± 0.126 |
| C1 center multiscale | 79.342 ± 0.789 | 1.625 ± 1.390 | 0.974305 ± 0.003898 | 80.089 ± 0.651 | 1.710 ± 0.558 | 67.667 ± 0.429 | 8.374 ± 0.190 |
| C2 center+ring | 77.801 ± 1.457 | 2.264 ± 1.979 | 0.972304 ± 0.004549 | 78.315 ± 1.496 | 1.637 ± 0.722 | 66.703 ± 0.300 | 8.359 ± 0.250 |

## Noise8 专训继续门槛

该门槛不声称来自原 `EXPERIMENT_PLAN.md`；其权威来源必须是候选结果产生前另行冻结的协议修订。

- C2 mean Fa@fixedPd relative reduction >=20%: FAIL
- C2 mean Pd@fixedFa delta >=-1pp: PASS
- Every seed Fa reduction >=-20%: FAIL
- Every seed Pd delta >=-3pp: PASS
- Every seed latency ratio <=1.3: PASS
- C2 Pareto-improves C1 within every paired seed: FAIL

## 配对视频 Bootstrap 95% CI

- 重采样单位：固定 16 个验证视频；候选与同 seed B1、三个训练 seed 共用同一组抽样权重。
- 重复：10000；随机种子：20260909；区间仅反映给定三个训练 seed 条件下的视频抽样不确定性。
- Pixel F1 缺少逐视频交并计数，latency 不是视频级可加统计量，因此二者只报告三 seed mean ± SD，不伪造视频 bootstrap CI。

| Variant | Metric | Paired estimate | 95% percentile CI | Valid |
|---|---|---:|---:|---:|
| C0 temporal control | `delta_pd_at_fixed_fa` | -0.027077498 | [-0.05, 0.018581807] | 10000/10000 |
| C0 temporal control | `delta_fa_at_fixed_pd` | 9.0114727e-06 | [-2.1007157e-06, 0.0029424368] | 10000/10000 |
| C0 temporal control | `fa_at_fixed_pd_relative_reduction` | -0.50520576 | [-173.82331, 0.14394846] | 10000/10000 |
| C0 temporal control | `delta_auc` | 0.0078542287 | [-0.0031526707, 0.019633242] | 10000/10000 |
| C0 temporal control | `delta_pd_at_0_5` | 0.01914099 | [-0.0016025641, 0.047143091] | 10000/10000 |
| C0 temporal control | `delta_fa_at_0_5` | 3.872699e-06 | [2.3307587e-06, 5.3172733e-06] | 10000/10000 |
| C1 center multiscale | `delta_pd_at_fixed_fa` | 0.011904762 | [-0.025524606, 0.030812325] | 10000/10000 |
| C1 center multiscale | `delta_fa_at_fixed_pd` | 3.4794711e-06 | [-3.453159e-06, 1.1540173e-05] | 10000/10000 |
| C1 center multiscale | `fa_at_fixed_pd_relative_reduction` | -0.14019353 | [-0.86458063, 0.24169269] | 10000/10000 |
| C1 center multiscale | `delta_auc` | 0.033017057 | [0.010326371, 0.060986438] | 10000/10000 |
| C1 center multiscale | `delta_pd_at_0_5` | 0.019374416 | [0.0029858849, 0.039543825] | 10000/10000 |
| C1 center multiscale | `delta_fa_at_0_5` | 4.1795359e-06 | [1.47181e-06, 7.1497422e-06] | 10000/10000 |
| C2 center+ring | `delta_pd_at_fixed_fa` | -0.0035014006 | [-0.030953004, 0.0047690763] | 10000/10000 |
| C2 center+ring | `delta_fa_at_fixed_pd` | 9.8664455e-06 | [4.9965964e-06, 0.00022427564] | 10000/10000 |
| C2 center+ring | `fa_at_fixed_pd_relative_reduction` | -0.55998523 | [-18.903571, -0.21708256] | 10000/10000 |
| C2 center+ring | `delta_auc` | 0.031015798 | [0.0085152636, 0.061030808] | 10000/10000 |
| C2 center+ring | `delta_pd_at_0_5` | 0.0016339869 | [-0.0051622419, 0.0080952381] | 10000/10000 |
| C2 center+ring | `delta_fa_at_0_5` | 3.4556391e-06 | [1.4917138e-06, 5.8896225e-06] | 10000/10000 |

## 解释边界

- 这是 Noise8 专门训练的 in-domain internal-validation 实验，不是 Clean 权重的 zero-shot Noise8 测试。
- Paper AUC 保留论文的 27 个 sigmoid 概率阈值；固定工作点也先由 109 点概率网格计算，因此可能受 sigmoid=1 饱和影响。
- 必须另做 raw-logit 事件阈值敏感性评测；只在概率与 raw-logit 门槛一致时，才把 `PROVISIONAL_PASS` 升级为 C3 授权。
- 三个训练 seed 支持稳定性筛查，不支持等效性证明；bootstrap CI 也不覆盖训练 seed 总体不确定性。
- 本分析器不要求或生成 SHA256；身份来自规范化路径、精确 split 内容、run manifest、日志、checkpoint metadata 和逐序列整数计数。
