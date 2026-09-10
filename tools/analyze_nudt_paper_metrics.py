#!/usr/bin/env python3
"""Build a complete, source-checked comparison of the two NUDT evaluations."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCHES = (
    ("NUDT-MIRSDT", "nudt_mirsdt_all_models_2026-09-01"),
    ("Noise8.0_FJY", "nudt_mirsdt_noise8_fjy_all_models_2026-09-03"),
)


def load_batch(batch):
    control = ROOT / "experiments" / batch
    runtime = ROOT / "log/sem_seg/_queues" / batch / "paper_metrics"
    with (control / "manifest.tsv").open() as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    assert len(jobs) == 29
    rows = {}
    for job in jobs:
        name = job["run_id"]
        assert (runtime / "status" / (name + ".done")).is_file(), name
        data = json.loads((runtime / "raw" / (name + ".json")).read_text())
        assert data["sequence_count"] == 20 and data["sequence_length"] == 40
        assert data["operating_threshold"] == 0.5
        assert Path(data["checkpoint"]).resolve() == (
            ROOT / "log/sem_seg" / job["log_dir"] / "checkpoints/best_model.pth"
        ).resolve()
        for group in ("all", "low_snr", "high_snr"):
            m = data[group]
            assert all(math.isfinite(m[k]) for k in ("pd_percent", "fa_x1e5", "auc"))
            assert abs(m["pd_percent"] - 100 * m["true_targets"] / m["total_targets"]) < 1e-9
            assert abs(m["fa_x1e5"] - 1e5 * m["false_pixels"] / m["pixel_count"]) < 1e-9
        rows[name] = data
    assert len({(r["all"]["total_targets"], r["all"]["pixel_count"]) for r in rows.values()}) == 1
    return rows


def dominates(a, b):
    va = (a["pd_percent"], -a["fa_x1e5"], a["auc"])
    vb = (b["pd_percent"], -b["fa_x1e5"], b["auc"])
    return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))


def describe(label, batch, rows):
    base = rows["deeppro_plus"]["all"]
    front = [n for n, r in rows.items() if not any(
        dominates(q["all"], r["all"]) for q in rows.values()
    )]
    better = [n for n, r in rows.items() if dominates(r["all"], base)]
    ordered = sorted(rows, key=lambda n: rows[n]["all"]["auc"], reverse=True)
    lines = [f"## {label}", "", "29/29 评测成功；固定工作点 0.5。Pd 单位为 %，Fa 单位为 10^-5。",
             "按 AUC 排列仅便于阅读，不构造 Pd/Fa/AUC 综合分数。", "",
             "| 模型 | Epoch | Pd ↑ | Fa ↓ | AUC ↑ | ΔPd (百分点) | ΔFa | ΔAUC | 参数量 M |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ordered:
        row = rows[name]; m = row["all"]
        lines.append(f"| {name} | {row['checkpoint_epoch']} | {m['pd_percent']:.4f} | {m['fa_x1e5']:.6f} | {m['auc']:.8f} | {m['pd_percent']-base['pd_percent']:+.4f} | {m['fa_x1e5']-base['fa_x1e5']:+.6f} | {m['auc']-base['auc']:+.8f} | {row['parameters_m']:.6f} |")
    lines += ["", "三指标同时不差且至少一个更优的 baseline 改进项：" + ("、".join(better) or "无") + "。",
              "三指标 Pareto 前沿（不存在另一模型三项同时不差且至少一项更好）：" + "、".join(front) + "。",
              "Pareto 前沿只是取舍集合，不代表每个成员都适合实际使用。", "",
              "### 低 SNR 分组", "",
              "原始 NUDT 使用论文指定的 8 条低 SNR 序列；Noise8 此处仅沿用同一序列分组，不能解释为重新测定的 SNR≤3 子集。", "",
              "| 模型 | Pd ↑ | Fa ↓ | AUC ↑ |", "|---|---:|---:|---:|"]
    for name in ordered:
        m = rows[name]["low_snr"]
        lines.append(f"| {name} | {m['pd_percent']:.4f} | {m['fa_x1e5']:.6f} | {m['auc']:.8f} |")
    lines += ["", f"原始指标与计数：`log/sem_seg/_queues/{batch}/paper_metrics/raw/`。"]
    lines += ["", "### 结果解释", ""]
    if label == "NUDT-MIRSDT":
        lines += [
            "MovingScenes 是唯一在三项总体指标上同时不劣于 baseline 且至少一项更好的模型：多命中 2 个目标（1724 对 1722），误警像素减少 199（3189 对 3388）。这是本次测试集的微小改善，不能声称稳定显著。",
            "BRTD1 总体 Pd 最高（99.8265%），比 baseline 多命中 4 个目标，同时增加 53 个误警像素；它更偏向提高检出率。低 SNR 子集也多检出 4 个目标（526 对 522），误警像素由 838 降到 759，低 SNR 的三项指标均改善。",
            "旧 pixel F1 冠军 Hybrid-RMS 多尺度对比，在论文工作点下总体少检出 19 个目标（1703 对 1722），误警像素多 70。其低 SNR Pd 为 95.0851%，低于 baseline 的 98.6767%。因此它不是干净数据上全面优于 baseline 的论文指标冠军。",
            "原始 Raw-APMD 在干净数据的 Pd、Fa、AUC 三项均差于 baseline；其 pixel F1 提升不能证明目标级检测能力提高。像素分割重合度与中心邻域命中、保护区外误警衡量的对象不同，排名反转并不必然说明格式错误。",
            "最高 AUC 的 DeepPro、TDC dual-stream 和 BRTD1 差距不到 0.000001；baseline 与这些模型的差距也只有约 0.000036。该接近饱和的全区间 AUC 不能单独承担模型筛选。",
        ]
    else:
        lines += [
            "没有候选同时在 Pd、Fa、AUC 三项上优于 baseline。有效检测模型中，baseline 的 Fa 最低（1.408066），应保留为低误警参照。",
            "raw_apmd_motion_detrend 的 AUC 最高（0.98203060）：比 baseline 多命中 47 个目标，但增加 1752 个误警像素；Fa 从 1.408066 增至 2.732238，约增加94%。这是明确的误警代价，不宜只写 AUC 提升。",
            "Hybrid-RMS 多尺度对比比 baseline 多检出 70 个目标（1369 对 1299），Pd 增加 4.0486 个百分点，误警像素增加 1099。它的 Pd 高于 motion_detrend，Fa 也更低，但 AUC 略低，适合继续验证固定工作点的取舍。",
            "local_align 的 Pd 达到 80.4511%，相对 baseline 多检出 92 个目标、增加 1395 个误警像素；这是排除明显海量误警的 lfp_shallow 后的最高 Pd。",
            "旧 pixel F1 冠军 Raw-APMD 与 baseline 都命中1299个目标，Pd 完全相同，却多出909个误警像素；优势在 AUC，而不是0.5阈值下的检出/误警取舍。",
            "Hybrid-RMS motion_detrend 是较温和的取舍：多检出4个目标，误警像素增加291，AUC提高约0.001629；还需验证其改善是否可重复。",
            "TDCSTA 的 Pd=0、Fa=0 是空检测，AUC约0.4995不能作为可用性证据；lfp_shallow 的 Pd=88.6640% 伴随约38%的全图误警像素，不能作为最高检出率的有效方案。",
        ]
    return lines


def main():
    batches = [(label, batch, load_batch(batch)) for label, batch in BATCHES]
    lines = ["## Material Passport", "", "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: validate", "- Origin Date: 2026-09-08",
             "- Verification Status: ANALYZED", "- Version Label: paper_metrics_comparison_v1", "",
             "# 两套 NUDT 实验：论文指标对比", "",
             "58 项训练均完成 32 epoch。本报告基于重新执行 test.py 后成功导出的 58 个 JSON；每个模型评估完整 20 条测试序列。",
             "修复包括提前保存 checkpoint epoch、拒绝非有限预测，以及 FeedbackSTS 按训练验证时的 FP32 设置评测。此前导出失败日志保留在 previous_export_failure_2026-09-04。",
             "FeedbackSTS 的 FP16 结果另行归档且不进入主表；主表使用重新执行的 FP32 结果，其他模型使用 AMP。70 组有限概率阈值检查与原始 ShootingRules 计数一致。",
             "运行记录：首轮调度收尾因运行中更新 shell 脚本出现解析错误；所有模型结果产生后已重新执行完整调度入口，跳过有效完成项、重建汇总并以退出码0和最终 PIPELINE_COMPLETE 验收。",
             "验证包括完成标记、20 条序列、40 帧、阈值、checkpoint 路径、有限数值以及目标数/误警计数与 Pd/Fa 的一致性。",
             "ANALYZED 表示已分析本次重新评测结果；未独立重训、未重复随机种子，不将一次运行宣称为完整复现验证。", "",
             "## 指标与可比性", "",
             "Pd=命中目标数/目标总数，Fa=保护区域外误警像素数/全部像素数；AUC 为既有 ShootingRules 和预定义阈值组上的 Pd-Fa 曲线面积。",
             "保留原实现的目标中心邻域、保护区和窗口 max 融合规则。此 AUC 不能与像素分类 ROC-AUC 混用。",
             "checkpoint 由训练期最佳像素 IoU 选择（同一汇总下与像素 F1 单调对应），并非逐模型按 AUC 再挑最优 checkpoint。", "",
             "论文 Table 2 的 DeepPro-Plus：干净数据 Pd=99.71%、Fa=2.69、AUC=0.9978；低 SNR Pd=99.24%、Fa=1.65；HiNo Pd=76.23%、Fa=1.69、AUC=0.9171。",
             "这些是论文已报告参照值；当前本地训练 lr=0.005、f1_calibrated_ohem（PointCenter 为专用损失），论文为 lr=0.001、Soft-IoU。",
             "论文 HiNo 指定 σn=8.0、σg=0.15、σo=1.3 和逐像素非均匀噪声；当前 Noise8.0_FJY 文件夹名称不足以证明生成协议及样本与其完全相同。",
             "因此主要比较本地同协议 deeppro_plus；不声称严格复现论文或直接刷新论文 HiNo 分数。",
             "论文参数量报告为 0.284 M；本地实际注册参数按 sum(numel) 统计，与论文不一致时必须单列，不能据此宣称压缩成功。",
             "本地 baseline 实际为 0.070913 M 参数，FP32 参数字节数约 0.283652 MB；test.py 的历史 count_parameters 函数统计的是字节 MB。",
             "该数值关系提示存在参数个数/存储大小口径混淆，尚不能据此断言论文采用了同一统计方式，也不能解读为模型压缩了四倍。",
             "不把 elapsed_seconds 换算成可与论文 V100 FPS 对比的速度；本次含数据处理且三卡并发，未执行独立硬件性能基准。", ""]
    for label, batch, rows in batches:
        lines += describe(label, batch, rows) + [""]
    clean, noise = batches[0][2], batches[1][2]
    lines += ["## 两数据集的性能变化", "",
              "各数据集分别训练后在对应测试集评估，非干净模型直接迁移到噪声数据。", "",
              "| 模型 | 干净 Pd | Noise8 Pd | ΔPd | 干净 AUC | Noise8 AUC | ΔAUC |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(clean, key=lambda n: noise[n]["all"]["auc"], reverse=True):
        a,b = clean[name]["all"],noise[name]["all"]
        lines.append(f"| {name} | {a['pd_percent']:.4f} | {b['pd_percent']:.4f} | {b['pd_percent']-a['pd_percent']:+.4f} | {a['auc']:.8f} | {b['auc']:.8f} | {b['auc']-a['auc']:+.8f} |")
    lines += ["", "## 统计解释边界", "",
              "单 seed=49，无重复训练方差、置信区间或显著性检验。29 个结构及多 epoch 选择带来选择偏差；小数位多不等于差异稳定。",
              "若 test.txt 同时用于 checkpoint/结构选择和最终报告，最终测试并非独立保留集，论文应明确这一局限并另设验证流程。",
              "AUC 较高但固定阈值 Pd 较低，表示曲线整体与选定工作点不同；Fa 接近零但 Pd 也接近零是可能的空预测，不能只按 Fa 排名。",
              "11/11 统计谬误已检查：Simpson（缺少完整分层验证）；生态谬误（不外推逐序列优势）；Berkson（不筛除低分模型）；",
              "Collider（无条件回归，不适用）；基率忽视（联合报告 Pd/Fa）；均值回归（最佳 checkpoint 乐观风险）；",
              "幸存者偏差（覆盖全部29项并保留失败证据）；多处寻找（结构/epoch筛选风险）；分叉路径（探索性）；",
              "因果误推（不将不同损失与兼容配置下的差值归因于单模块）；反向因果（不适用）。总体置信等级 CAUTION。", "",
              "来源：仓库 paper/ 下 Li 等 2026 本地 PDF（实验设置与 Table 2）、两批 manifest、训练日志、本次评测 JSON。",
              "生成命令：`python3 tools/analyze_nudt_paper_metrics.py`。"]
    output = ROOT / "experiments/PAPER_COMPARISON_2026-09-08.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
