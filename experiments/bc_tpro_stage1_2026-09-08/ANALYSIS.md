# BC-TPro 第一阶段结果分析

状态：已获得 24/24 份 Clean/Noise8 验证指标；C2 继续门槛：**FAIL**。

主表使用 `epoch_32_model.pth`。`Pd@fixedFa` 和 `Fa@fixedPd` 的工作点由同 seed、同条件 B1 在阈值 0.5 处预先定义；低 Fa pAUC 在 `Fa<=5.0e-05` 内归一化。

## 身份与完整性检查

- manifest 必须精确匹配 12 个计划 run 的 wave/model/variant/seed/GPU/log_dir 映射。
- Clean/Noise8 `train.txt`、固定 train split、固定 validation split 的 SHA256 均已钉住。
- 每份已加载指标均通过训练 Namespace、checkpoint metadata、匹配 `metrics_json` 的独立评测 Namespace，以及 counts 重算检查。

## 三随机种子汇总

| Condition | Variant | n | Pd@fixedFa (%) | Fa@fixedPd (×1e-5) | low-Fa pAUC | Paper/Dense AUC | Pd@0.5 (%) | Fa@0.5 (×1e-5) | Pixel IoU/F1 (%) | Time (s) | Peak alloc/reserved (GiB) | Params (M) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_val | B1 DeepPro-Plus | 3 | 100.000 ± 0.000 | 1.639 ± 0.050 | 0.8361 ± 0.0050 | 0.9989 ± 0.0000 / 0.9989 ± 0.0000 | 100.000 ± 0.000 | 2.180 ± 0.144 | 96.719 ± 0.825 / 98.331 ± 0.427 | 8.4 ± 0.1 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.070913 |
| noise8_val | B1 DeepPro-Plus | 3 | 93.231 ± 1.848 | 22272.918 ± 4496.025 | 0.0002 ± 0.0001 | 0.8923 ± 0.0199 / 0.8923 ± 0.0199 | 93.231 ± 1.848 | 22339.758 ± 4471.182 | 0.085 ± 0.012 / 0.169 ± 0.024 | 7.8 ± 0.1 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.070913 |
| clean_val | C0 temporal control | 3 | 100.000 ± 0.000 | 1.977 ± 0.028 | 0.8483 ± 0.0116 | 0.9990 ± 0.0000 / 0.9990 ± 0.0000 | 100.000 ± 0.000 | 2.091 ± 0.090 | 96.891 ± 0.463 / 98.420 ± 0.239 | 8.5 ± 0.1 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.071233 |
| noise8_val | C0 temporal control | 3 | 94.304 ± 1.314 | 22029.303 ± 5396.348 | 0.0003 ± 0.0000 | 0.9163 ± 0.0074 / 0.9163 ± 0.0074 | 89.332 ± 2.815 | 16046.435 ± 1184.710 | 0.099 ± 0.003 / 0.198 ± 0.007 | 8.1 ± 0.2 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.071233 |
| clean_val | C1 center multiscale | 3 | 99.953 ± 0.040 | 44.174 ± 57.036 | 0.8857 ± 0.0134 | 0.9990 ± 0.0000 / 0.9990 ± 0.0000 | 99.953 ± 0.040 | 2.172 ± 0.097 | 96.966 ± 0.330 / 98.459 ± 0.170 | 8.5 ± 0.2 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.071233 |
| noise8_val | C1 center multiscale | 3 | 80.065 ± 17.342 | 22399.241 ± 5704.418 | 0.0003 ± 0.0001 | 0.9139 ± 0.0155 / 0.9139 ± 0.0155 | 98.669 ± 0.875 | 28747.880 ± 5765.868 | 0.082 ± 0.010 / 0.164 ± 0.019 | 7.8 ± 0.1 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.071233 |
| clean_val | C2 center+ring | 3 | 100.000 ± 0.000 | 2.062 ± 0.126 | 0.8839 ± 0.0175 | 0.9990 ± 0.0000 / 0.9990 ± 0.0000 | 100.000 ± 0.000 | 2.212 ± 0.187 | 96.901 ± 0.519 / 98.426 ± 0.268 | 8.6 ± 0.1 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.071281 |
| noise8_val | C2 center+ring | 3 | 96.499 ± 2.324 | 18436.892 ± 3447.642 | 0.0003 ± 0.0001 | 0.9181 ± 0.0087 / 0.9181 ± 0.0088 | 97.619 ± 1.195 | 24141.637 ± 2933.838 | 0.086 ± 0.006 / 0.172 ± 0.011 | 7.8 ± 0.0 | 1.496 ± 0.000 / 1.604 ± 0.000 | 0.071281 |

## 配对差值（相对同 seed B1）

| Condition | Variant | ΔPd@fixedFa (pp) | Fa@fixedPd 相对下降 | ΔpAUC | Latency ratio |
|---|---|---:|---:|---:|---:|
| clean_val | C0 temporal control | 0.000 ± 0.000 | -20.72% ± 4.77% | 0.0122 ± 0.0131 | 1.008 ± 0.011 |
| noise8_val | C0 temporal control | 1.074 ± 1.063 | 1.05% ± 17.06% | 0.0001 ± 0.0000 | 1.043 ± 0.028 |
| clean_val | C1 center multiscale | -0.047 ± 0.040 | -2643.40% ± 3594.31% | 0.0496 ± 0.0166 | 1.008 ± 0.007 |
| noise8_val | C1 center multiscale | -13.165 ± 15.834 | -0.27% ± 10.20% | 0.0001 ± 0.0000 | 0.999 ± 0.019 |
| clean_val | C2 center+ring | 0.000 ± 0.000 | -25.93% ± 10.37% | 0.0479 ± 0.0209 | 1.020 ± 0.022 |
| noise8_val | C2 center+ring | 3.268 ± 1.191 | 17.07% ± 1.79% | 0.0001 ± 0.0000 | 1.000 ± 0.009 |

## 按视频配对 Bootstrap 95% CI

固定 seed `20260908`、10000 次有放回序列重采样；每个 replicate 在候选/B1、三个训练 seed 及 Clean/Noise8 间使用同一组 16 个序列索引。区间是给定这三个训练 seed 条件下的视频抽样不确定性，不代表训练随机性的总体置信区间。

| Condition | Variant | Metric | Paired estimate | Bootstrap median | 95% percentile CI | Valid |
|---|---|---|---:|---:|---:|---:|
| clean_val | C0 temporal control | `delta_pd_at_fixed_fa` | 0.000 pp | 0.000 pp | [-0.067 pp, 0.000 pp] | 10000/10000 (100.0%) |
| clean_val | C0 temporal control | `delta_fa_at_fixed_pd` | 0.338 ×1e-5 | 0.217 ×1e-5 | [-0.127 ×1e-5, 0.612 ×1e-5] | 10000/10000 (100.0%) |
| clean_val | C0 temporal control | `fa_at_fixed_pd_relative_reduction` | -20.72% | -15.76% | [-26.60%, 8.84%] | 10000/10000 (100.0%) |
| clean_val | C0 temporal control | `delta_low_fa_pauc` | 0.0122 | 0.0120 | [0.0039, 0.0237] | 10000/10000 (100.0%) |
| noise8_val | C0 temporal control | `delta_pd_at_fixed_fa` | 1.074 pp | 0.978 pp | [-0.280 pp, 2.289 pp] | 10000/10000 (100.0%) |
| noise8_val | C0 temporal control | `delta_fa_at_fixed_pd` | -243.615 ×1e-5 | -829.797 ×1e-5 | [-2863.905 ×1e-5, 1048.691 ×1e-5] | 10000/10000 (100.0%) |
| noise8_val | C0 temporal control | `fa_at_fixed_pd_relative_reduction` | 1.05% | 3.25% | [-5.35%, 12.87%] | 10000/10000 (100.0%) |
| noise8_val | C0 temporal control | `delta_low_fa_pauc` | 0.0001 | 0.0001 | [0.0001, 0.0001] | 10000/10000 (100.0%) |
| clean_val | C1 center multiscale | `delta_pd_at_fixed_fa` | -0.047 pp | -0.046 pp | [-0.611 pp, 0.000 pp] | 10000/10000 (100.0%) |
| clean_val | C1 center multiscale | `delta_fa_at_fixed_pd` | 42.534 ×1e-5 | 37.169 ×1e-5 | [0.161 ×1e-5, 61.498 ×1e-5] | 10000/10000 (100.0%) |
| clean_val | C1 center multiscale | `fa_at_fixed_pd_relative_reduction` | -2643.40% | -1952.05% | [-4295.77%, -16.45%] | 10000/10000 (100.0%) |
| clean_val | C1 center multiscale | `delta_low_fa_pauc` | 0.0496 | 0.0479 | [0.0212, 0.0966] | 10000/10000 (100.0%) |
| noise8_val | C1 center multiscale | `delta_pd_at_fixed_fa` | -13.165 pp | -13.176 pp | [-17.590 pp, -8.312 pp] | 10000/10000 (100.0%) |
| noise8_val | C1 center multiscale | `delta_fa_at_fixed_pd` | 126.323 ×1e-5 | 179.982 ×1e-5 | [-548.817 ×1e-5, 912.913 ×1e-5] | 10000/10000 (100.0%) |
| noise8_val | C1 center multiscale | `fa_at_fixed_pd_relative_reduction` | -0.27% | -0.50% | [-3.52%, 3.00%] | 10000/10000 (100.0%) |
| noise8_val | C1 center multiscale | `delta_low_fa_pauc` | 0.0001 | 0.0001 | [0.0001, 0.0001] | 10000/10000 (100.0%) |
| clean_val | C2 center+ring | `delta_pd_at_fixed_fa` | 0.000 pp | 0.000 pp | [0.000 pp, 0.000 pp] | 10000/10000 (100.0%) |
| clean_val | C2 center+ring | `delta_fa_at_fixed_pd` | 0.422 ×1e-5 | 0.409 ×1e-5 | [-0.091 ×1e-5, 0.774 ×1e-5] | 10000/10000 (100.0%) |
| clean_val | C2 center+ring | `fa_at_fixed_pd_relative_reduction` | -25.93% | -25.21% | [-39.95%, 9.02%] | 10000/10000 (100.0%) |
| clean_val | C2 center+ring | `delta_low_fa_pauc` | 0.0479 | 0.0461 | [0.0204, 0.0932] | 10000/10000 (100.0%) |
| noise8_val | C2 center+ring | `delta_pd_at_fixed_fa` | 3.268 pp | 3.289 pp | [1.821 pp, 4.786 pp] | 10000/10000 (100.0%) |
| noise8_val | C2 center+ring | `delta_fa_at_fixed_pd` | -3836.026 ×1e-5 | -3646.434 ×1e-5 | [-3934.231 ×1e-5, -3001.858 ×1e-5] | 10000/10000 (100.0%) |
| noise8_val | C2 center+ring | `fa_at_fixed_pd_relative_reduction` | 17.07% | 16.07% | [12.96%, 18.25%] | 10000/10000 (100.0%) |
| noise8_val | C2 center+ring | `delta_low_fa_pauc` | 0.0001 | 0.0001 | [0.0001, 0.0001] | 10000/10000 (100.0%) |

所有指标只有在至少 95% bootstrap replicate 有效时才报告区间；否则区间显示 `NA`。相对 Fa 下降还要求基线 Fa 非零，此时应优先解释绝对 `ΔFa@fixedPd`。

## 工作点与低 Fa 支持诊断

| Condition | Variant | Seed | Pd@fixedFa selected threshold | Pd plateau [min,max] / n | Endpoint | Fa@fixedPd selected threshold | Fa plateau [min,max] / n | Endpoint | Internal unique Fa points | Min empirical Fa | Support |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| clean_val | B1 DeepPro-Plus | 47 | 0.500000 | [0.500000, 1.000000] / 51 | no | 1.000000 | [1.000000, 1.000000] / 1 | yes | 102 | 0.00001594 | `empirical-internal-support` |
| clean_val | B1 DeepPro-Plus | 49 | 0.500000 | [0.500000, 1.000000] / 51 | no | 1.000000 | [1.000000, 1.000000] / 1 | yes | 66 | 0.00001694 | `empirical-internal-support` |
| clean_val | B1 DeepPro-Plus | 51 | 0.500000 | [0.500000, 1.000000] / 51 | no | 1.000000 | [1.000000, 1.000000] / 1 | yes | 96 | 0.00001630 | `empirical-internal-support` |
| clean_val | C0 temporal control | 47 | 0.010000 | [0.010000, 0.990000] / 99 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 92 | 0.00001610 | `empirical-internal-support` |
| clean_val | C0 temporal control | 49 | 0.390000 | [0.390000, 0.990000] / 61 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 85 | 0.00001550 | `empirical-internal-support` |
| clean_val | C0 temporal control | 51 | 0.020000 | [0.020000, 0.990000] / 98 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 76 | 0.00001386 | `empirical-internal-support` |
| clean_val | C1 center multiscale | 47 | 0.230000 | [0.230000, 0.990000] / 77 | no | 0.000010 | [0.000010, 0.000010] / 1 | no | 88 | 0.00001255 | `empirical-internal-support` |
| clean_val | C1 center multiscale | 49 | 0.760000 | [0.760000, 0.990000] / 24 | no | 0.000100 | [0.000100, 0.000100] / 1 | no | 101 | 0.00001088 | `empirical-internal-support` |
| clean_val | C1 center multiscale | 51 | 0.210000 | [0.210000, 0.990000] / 79 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 85 | 0.00000971 | `empirical-internal-support` |
| clean_val | C2 center+ring | 47 | 0.890000 | [0.890000, 0.990000] / 11 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 90 | 0.00001329 | `empirical-internal-support` |
| clean_val | C2 center+ring | 49 | 0.910000 | [0.910000, 0.980000] / 8 | no | 0.980000 | [0.980000, 0.980000] / 1 | no | 92 | 0.00001096 | `empirical-internal-support` |
| clean_val | C2 center+ring | 51 | 0.210000 | [0.210000, 0.990000] / 79 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 85 | 0.00000996 | `empirical-internal-support` |
| noise8_val | B1 DeepPro-Plus | 47 | 0.500000 | [0.500000, 0.550000] / 6 | no | 0.550000 | [0.550000, 0.550000] / 1 | no | 0 | 0.08537092 | `origin-interpolation-only` |
| noise8_val | B1 DeepPro-Plus | 49 | 0.500000 | [0.500000, 0.500000] / 1 | no | 0.500000 | [0.500000, 0.500000] / 1 | no | 0 | 0.08946366 | `origin-interpolation-only` |
| noise8_val | B1 DeepPro-Plus | 51 | 0.500000 | [0.500000, 0.510000] / 2 | no | 0.510000 | [0.510000, 0.510000] / 1 | no | 0 | 0.05252898 | `origin-interpolation-only` |
| noise8_val | C0 temporal control | 47 | 0.001000 | [0.001000, 0.001000] / 1 | no | 0.000100 | [0.000100, 0.000100] / 1 | no | 0 | 0.05160596 | `origin-interpolation-only` |
| noise8_val | C0 temporal control | 49 | 0.000100 | [0.000100, 0.000100] / 1 | no | 0.000100 | [0.000100, 0.000100] / 1 | no | 0 | 0.04467839 | `origin-interpolation-only` |
| noise8_val | C0 temporal control | 51 | 0.300000 | [0.300000, 0.310000] / 2 | no | 0.870000 | [0.870000, 0.870000] / 1 | no | 0 | 0.03768100 | `origin-interpolation-only` |
| noise8_val | C1 center multiscale | 47 | 0.930000 | [0.930000, 0.930000] / 1 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 0 | 0.05774402 | `origin-interpolation-only` |
| noise8_val | C1 center multiscale | 49 | 1.000000 | [1.000000, 1.000000] / 1 | yes | 0.990000 | [0.990000, 0.990000] / 1 | no | 0 | 0.08049201 | `origin-interpolation-only` |
| noise8_val | C1 center multiscale | 51 | 1.000000 | [1.000000, 1.000000] / 1 | yes | 0.990000 | [0.990000, 0.990000] / 1 | no | 0 | 0.03662871 | `origin-interpolation-only` |
| noise8_val | C2 center+ring | 47 | 0.820000 | [0.820000, 0.820000] / 1 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 0 | 0.06781167 | `origin-interpolation-only` |
| noise8_val | C2 center+ring | 49 | 0.580000 | [0.580000, 0.670000] / 10 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 0 | 0.05240291 | `origin-interpolation-only` |
| noise8_val | C2 center+ring | 51 | 0.920000 | [0.920000, 0.920000] / 1 | no | 0.990000 | [0.990000, 0.990000] / 1 | no | 0 | 0.04213523 | `origin-interpolation-only` |

`origin-interpolation-only` 表示 `0 < Fa < cap` 没有任何经验曲线点；该 pAUC 仅由人为补入的原点与第一个经验点线性插值得到。尤其在 Noise8 出现此标记时，不能把数值解释为实测低虚警性能。
工作点并列时，`Pd@fixedFa` 选择满足最优 Pd 的最低阈值，`Fa@fixedPd` 选择满足最优 Fa 的最高阈值；表中同时给出完整并列 plateau 范围及点数。

## C2 继续门槛

原计划写明“无单 seed 严重反向异常”但未给数值；本分析修正（非原始预注册）将其操作化为 Fa 相对下降不低于 -20%、Pd 差不低于 -3 pp。

- Clean mean Fa reduction >=20%: FAIL
- Clean mean Pd loss <=1pp: PASS
- Noise8 direction does not reverse: PASS
- No severe reverse seed (analysis amendment: Fa >=-20%, Pd >=-3pp): FAIL
- Every latency ratio <=1.3x: PASS
- 结论：C2 未通过预注册门槛，不应在同一证据基础上直接堆叠 C3/C4。

## 解释边界

- 三个训练 seed 只支持稳定性筛查，不支持“证明等效”或强显著性结论。
- Noise8 是 Clean 权重的外部分布测试，不是第二套训练重复。
- pAUC 使用本次加密阈值网格；完整 AUC 同时保留论文阈值协议。
- Pixel F1/IoU 只用于定位训练退化，不替代目标级 Pd/Fa。
- 本阶段没有使用官方 test，最终方法确定后才允许进行一次锁定测试。
