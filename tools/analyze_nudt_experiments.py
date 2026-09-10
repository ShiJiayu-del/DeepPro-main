#!/usr/bin/env python3
"""Create a detailed, reproducible analysis from an all-model results CSV."""

import argparse
import csv
import statistics
from pathlib import Path


NUMERIC_FIELDS = (
    "best_iou",
    "best_precision",
    "best_recall",
    "best_f1",
    "latest_f1",
    "delta_best_iou",
    "delta_best_precision",
    "delta_best_recall",
    "delta_best_f1",
    "delta_latest_f1",
    "elapsed_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-run-id", default="deeppro_plus")
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--reference-label", default="NUDT-MIRSDT clean")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in NUMERIC_FIELDS:
            value = row.get(field, "")
            row[field] = float(value) if value else None
        for field in ("rank", "evaluations", "best_epoch", "latest_epoch"):
            value = row.get(field, "")
            row[field] = int(value) if value else None
    return rows


def metric(value, signed=False):
    if value is None:
        return "-"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def family(row):
    return "BRTD3 variants" if "BRTD3" in row["model"] else "standalone networks"


def ranked_complete(rows):
    return sorted(
        (row for row in rows if row["best_f1"] is not None),
        key=lambda row: row["best_f1"],
        reverse=True,
    )


def build_single_dataset_analysis(rows, label, baseline_id):
    completed = [row for row in rows if row["status"] == "done"]
    failed = [row for row in rows if row["status"] == "failed"]
    running = [row for row in rows if row["status"] == "running"]
    ranked = ranked_complete(rows)
    baseline = next(row for row in rows if row["run_id"] == baseline_id)
    lines = [
        f"# {label} 训练筛选分析",
        "",
        "> 本文档分析 pixel F1/IoU，仅作为训练诊断和结构初筛；论文主表使用 `PAPER_METRICS.md` 的 Pd/Fa/AUC。",
        "",
        "## 完整性与口径",
        "",
        f"- 清单共 {len(rows)} 项：完成 {len(completed)}，运行中 {len(running)}，失败 {len(failed)}。",
        "- 筛选顺序采用验证期间最佳 pixel F1；IoU、Precision、Recall 均来自该最佳 F1 的同一轮。",
        "- `Final F1` 是最后一次验证结果，`Best-Final gap` 用来观察训练末期回落。",
        f"- 同协议 baseline 是 `{baseline_id}`；所有模型均为随机初始化，不加载预训练权重。",
        "",
        "## 总排名",
        "",
        "| Rank | Run | Model / Variant | Best epoch | IoU | Precision | Recall | Best F1 | Final F1 | Best-Final gap | ΔF1 vs baseline |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        model = row["model"]
        if row["structure_variant"] != "-":
            model += " / " + row["structure_variant"]
        gap = row["best_f1"] - row["latest_f1"]
        lines.append(
            f"| {index} | `{row['run_id']}` | {model} | {row['best_epoch']} | "
            f"{metric(row['best_iou'])} | {metric(row['best_precision'])} | "
            f"{metric(row['best_recall'])} | {metric(row['best_f1'])} | "
            f"{metric(row['latest_f1'])} | {metric(gap)} | "
            f"{metric(row['best_f1'] - baseline['best_f1'], signed=True)} |"
        )

    top = ranked[0]
    gains = [row for row in ranked if row["best_f1"] > baseline["best_f1"]]
    best_precision = max(ranked, key=lambda row: row["best_precision"])
    best_recall = max(ranked, key=lambda row: row["best_recall"])
    most_stable = min(
        ranked,
        key=lambda row: abs(row["best_f1"] - row["latest_f1"]),
    )
    lines.extend([
        "",
        "## 关键结论",
        "",
        f"- 最佳模型为 `{top['run_id']}`，Best F1={metric(top['best_f1'])}，"
        f"相对 baseline 提升 {metric(top['best_f1'] - baseline['best_f1'], signed=True)}。",
        f"- 共 {len(gains)}/{len(ranked)} 个有有效指标的模型超过 baseline。",
        f"- 最高 Precision 来自 `{best_precision['run_id']}`（{metric(best_precision['best_precision'])}）；"
        f"最高 Recall 来自 `{best_recall['run_id']}`（{metric(best_recall['best_recall'])}）。",
        f"- 最佳与最终 F1 差距最小的是 `{most_stable['run_id']}`"
        f"（{metric(abs(most_stable['best_f1'] - most_stable['latest_f1']))}）。",
        "",
        "## 模型族统计",
        "",
        "| Family | Count | Mean Best F1 | Median Best F1 | Best run | Best F1 |",
        "|---|---:|---:|---:|---|---:|",
    ])
    for name in ("standalone networks", "BRTD3 variants"):
        members = [row for row in ranked if family(row) == name]
        winner = max(members, key=lambda row: row["best_f1"])
        values = [row["best_f1"] for row in members]
        lines.append(
            f"| {name} | {len(members)} | {metric(statistics.mean(values))} | "
            f"{metric(statistics.median(values))} | `{winner['run_id']}` | "
            f"{metric(winner['best_f1'])} |"
        )

    anchor = next((row for row in rows if row["run_id"] == "brtd3_second_order"), None)
    variants = [row for row in ranked if family(row) == "BRTD3 variants"]
    if anchor and anchor["best_f1"] is not None:
        lines.extend([
            "",
            "## BRTD3 结构消融",
            "",
            "`brtd3_second_order` 作为 BRTD3 默认二阶结构锚点；下表按相对它的 F1 变化排序。",
            "",
            "| Run | Variant | Best F1 | ΔF1 vs second_order | Precision | Recall |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in sorted(
            variants,
            key=lambda item: item["best_f1"] - anchor["best_f1"],
            reverse=True,
        ):
            lines.append(
                f"| `{row['run_id']}` | `{row['structure_variant']}` | "
                f"{metric(row['best_f1'])} | "
                f"{metric(row['best_f1'] - anchor['best_f1'], signed=True)} | "
                f"{metric(row['best_precision'])} | {metric(row['best_recall'])} |"
            )

    gaps = sorted(
        (
            (row["best_f1"] - row["latest_f1"], row)
            for row in ranked
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    lines.extend([
        "",
        "## 收敛与选择建议",
        "",
        f"- 最大训练末期回落为 `{gaps[0][1]['run_id']}`："
        f"{metric(gaps[0][0])}。最终部署应使用保存的 best checkpoint，而不是机械采用 latest checkpoint。",
        f"- 若以 F1 为唯一主指标，首选 `{top['run_id']}` 的 best checkpoint（epoch {top['best_epoch']}）。",
        "- Precision 与 Recall 的极值来自不同模型时，应避免只看单项指标；F1 排名仍是本轮模型选择依据。",
        "- 单 seed 结果适合筛选结构，但很小的差距仍需多 seed 复验后才能视为稳定改进。",
        "",
    ])
    if failed or running:
        unresolved = ", ".join(
            f"{row['run_id']}({row['status']})" for row in failed + running
        )
        lines.extend(["## 未完成项", "", unresolved, ""])
    return lines


def append_cross_dataset(lines, rows, reference_rows, label, reference_label):
    current = {row["run_id"]: row for row in rows if row["best_f1"] is not None}
    reference = {
        row["run_id"]: row for row in reference_rows if row["best_f1"] is not None
    }
    common = sorted(set(current) & set(reference))
    rank_current = {
        row["run_id"]: index
        for index, row in enumerate(ranked_complete(rows), start=1)
    }
    rank_reference = {
        row["run_id"]: index
        for index, row in enumerate(ranked_complete(reference_rows), start=1)
    }
    comparisons = []
    for run_id in common:
        noisy = current[run_id]
        clean = reference[run_id]
        comparisons.append({
            "run_id": run_id,
            "clean_f1": clean["best_f1"],
            "current_f1": noisy["best_f1"],
            "delta": noisy["best_f1"] - clean["best_f1"],
            "retention": noisy["best_f1"] / clean["best_f1"],
            "rank_shift": rank_reference[run_id] - rank_current[run_id],
        })
    comparisons.sort(key=lambda item: item["current_f1"], reverse=True)
    most_robust = max(comparisons, key=lambda item: item["retention"])
    smallest_drop = max(comparisons, key=lambda item: item["delta"])
    mean_delta = statistics.mean(item["delta"] for item in comparisons)
    lines.extend([
        "## 跨数据集抗噪对比",
        "",
        f"以下比较 `{reference_label}` 与 `{label}` 的同名模型；共匹配 {len(common)} 项。",
        "",
        "| Noise rank | Run | Clean Best F1 | Noise Best F1 | Noise-Clean ΔF1 | F1 retention | Rank shift |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for item in comparisons:
        lines.append(
            f"| {rank_current[item['run_id']]} | `{item['run_id']}` | "
            f"{metric(item['clean_f1'])} | {metric(item['current_f1'])} | "
            f"{metric(item['delta'], signed=True)} | {item['retention']:.2%} | "
            f"{item['rank_shift']:+d} |"
        )
    lines.extend([
        "",
        "### 抗噪结论",
        "",
        f"- 全模型平均 Noise-Clean ΔF1 为 {metric(mean_delta, signed=True)}。",
        f"- F1 保持率最高的是 `{most_robust['run_id']}`（{most_robust['retention']:.2%}）。",
        f"- F1 绝对下降最小的是 `{smallest_drop['run_id']}`"
        f"（{metric(smallest_drop['delta'], signed=True)}）。",
        "- `Rank shift` 为正表示在噪声数据中的相对名次上升；它反映相对鲁棒性，不等同于绝对 F1 提高。",
        "",
    ])


def main():
    args = parse_args()
    rows = read_rows(args.results)
    incomplete = [row for row in rows if row["status"] != "done"]
    if incomplete and not args.allow_incomplete:
        names = ", ".join(f"{row['run_id']}:{row['status']}" for row in incomplete)
        raise SystemExit(f"Refusing final analysis with incomplete runs: {names}")
    lines = build_single_dataset_analysis(
        rows,
        args.dataset_label,
        args.baseline_run_id,
    )
    if args.reference_results:
        reference_rows = read_rows(args.reference_results)
        append_cross_dataset(
            lines,
            rows,
            reference_rows,
            args.dataset_label,
            args.reference_label,
        )
    output = args.output or args.results.with_name("ANALYSIS.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"ANALYSIS rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
