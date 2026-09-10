#!/usr/bin/env python3
"""Build DeepPro-Plus-paper-aligned Pd/Fa/AUC comparison tables."""

import argparse
import csv
import json
from pathlib import Path


PUBLISHED_DEEPPRO_PLUS = {
    "clean": {
        "low_snr": {"pd_percent": 99.24, "fa_x1e5": 1.65},
        "all": {"pd_percent": 99.71, "fa_x1e5": 2.69, "auc": 0.9978},
    },
    "noise8": {
        "all": {"pd_percent": 76.23, "fa_x1e5": 1.69, "auc": 0.9171},
    },
    "parameters_m": 0.284,
    "gflops_per_frame_256": 3.89,
    "fps_v100": 224.05,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, help="Batch runtime directory under log/sem_seg/_queues")
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--paper-scenario", choices=("clean", "noise8"), required=True)
    parser.add_argument("--baseline-run-id", default="deeppro_plus")
    return parser.parse_args()


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def nested(payload, group, metric):
    return payload.get(group, {}).get(metric)


def main():
    args = parse_args()
    manifest_path = args.control_root / "manifest.tsv"
    save_root = args.save_root or (
        Path(__file__).resolve().parents[1] / "log" / "sem_seg"
        / "_queues" / args.control_root.name
    )
    metrics_root = save_root / "paper_metrics" / "raw"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for job in manifest:
        path = metrics_root / f"{job['run_id']}.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        rows.append({
            **job,
            "status": "done" if payload else "missing",
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "low_snr_pd": nested(payload, "low_snr", "pd_percent"),
            "low_snr_fa": nested(payload, "low_snr", "fa_x1e5"),
            "low_snr_auc": nested(payload, "low_snr", "auc"),
            "all_pd": nested(payload, "all", "pd_percent"),
            "all_fa": nested(payload, "all", "fa_x1e5"),
            "all_auc": nested(payload, "all", "auc"),
            "parameters_m": payload.get("parameters_m"),
            "pixel_iou": payload.get("pixel_iou_at_0_5"),
            "pixel_f1": payload.get("pixel_f1_at_0_5"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })

    completed = [row for row in rows if row["status"] == "done"]
    ranked = sorted(
        completed,
        key=lambda row: (
            row["all_auc"],
            row["all_pd"],
            -row["all_fa"],
        ),
        reverse=True,
    )
    baseline = next(
        (row for row in rows if row["run_id"] == args.baseline_run_id),
        None,
    )

    csv_path = args.control_root / "paper_metrics.csv"
    fieldnames = [
        "auc_order", "run_id", "model", "structure_variant", "status",
        "checkpoint_epoch", "low_snr_pd", "low_snr_fa", "low_snr_auc",
        "all_pd", "all_fa", "all_auc", "parameters_m", "pixel_iou",
        "pixel_f1", "elapsed_seconds", "delta_pd_vs_baseline",
        "delta_fa_vs_baseline", "delta_auc_vs_baseline",
    ]
    order = {row["run_id"]: index for index, row in enumerate(ranked, start=1)}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["auc_order"] = order.get(row["run_id"], "")
            if baseline and baseline["all_auc"] is not None and row["all_auc"] is not None:
                output["delta_pd_vs_baseline"] = row["all_pd"] - baseline["all_pd"]
                output["delta_fa_vs_baseline"] = row["all_fa"] - baseline["all_fa"]
                output["delta_auc_vs_baseline"] = row["all_auc"] - baseline["all_auc"]
            else:
                output["delta_pd_vs_baseline"] = ""
                output["delta_fa_vs_baseline"] = ""
                output["delta_auc_vs_baseline"] = ""
            writer.writerow({name: output.get(name, "") for name in fieldnames})

    published = PUBLISHED_DEEPPRO_PLUS[args.paper_scenario]
    lines = [
        f"# {args.dataset_label} 论文对齐指标",
        "",
        f"完成 {len(completed)}/{len(rows)}。以下 `Pd` 使用 sigmoid 后阈值 0.5，单位为百分数；"
        "`Fa` 单位为 10^-5；`AUC` 来自论文代码采用的预定义阈值组扫描。",
        "",
        "排序列 `AUC order` 只是便于阅读；Pd、Fa、AUC 是多目标评价，不定义虚构的综合分数。",
        "",
        "## 本地同协议模型",
        "",
        "| AUC order | Run | Epoch | Pd | Fa | AUC | Params (M) | Pixel IoU (supp.) | Pixel F1 (supp.) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | `{row['run_id']}` | {row['checkpoint_epoch']} | "
            f"{fmt(row['all_pd'], 2)} | {fmt(row['all_fa'], 2)} | "
            f"{fmt(row['all_auc'])} | {fmt(row['parameters_m'], 3)} | "
            f"{fmt(row['pixel_iou'], 6)} | {fmt(row['pixel_f1'], 6)} |"
        )

    lines.extend([
        "",
        "## 论文已报告的 DeepPro-Plus 参照值",
        "",
        "| Source | Pd | Fa | AUC | Params (M) | GFLOPs/frame (256x256) | FPS (V100) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Li et al., TPAMI 2026 | {pd:.2f} | {fa:.2f} | {auc:.4f} | {params:.3f} | {gflops:.2f} | {fps:.2f} |".format(
            pd=published["all"]["pd_percent"],
            fa=published["all"]["fa_x1e5"],
            auc=published["all"]["auc"],
            params=PUBLISHED_DEEPPRO_PLUS["parameters_m"],
            gflops=PUBLISHED_DEEPPRO_PLUS["gflops_per_frame_256"],
            fps=PUBLISHED_DEEPPRO_PLUS["fps_v100"],
        ),
        "",
    ])
    if args.paper_scenario == "clean":
        lines.extend([
            "## SNR <= 3 子集",
            "",
            "| Run | Pd | Fa | AUC (supp.) |",
            "|---|---:|---:|---:|",
        ])
        for row in ranked:
            lines.append(
                f"| `{row['run_id']}` | {fmt(row['low_snr_pd'], 2)} | "
                f"{fmt(row['low_snr_fa'], 2)} | {fmt(row['low_snr_auc'])} |"
            )
        low = published["low_snr"]
        lines.extend([
            f"| DeepPro-Plus (paper) | {low['pd_percent']:.2f} | {low['fa_x1e5']:.2f} | - |",
            "",
        ])

    lines.extend([
        "## 可比性边界",
        "",
        "- 检测指标、阈值、SNR 分组和单位与 DeepPro-Plus 论文对齐。",
        "- 本地 29 项使用同一套随机初始化训练协议，适合做受控结构消融。",
        "- 当前本地训练使用 learning rate 0.005 和 `f1_calibrated_ohem`；原论文使用 learning rate 0.001、每 10 epoch 乘 0.7，并使用 Soft-IoU。"
        "因此本地数值不能冒充论文官方配置复现，应与论文已报告行分开呈现。",
        "- FPS 与硬件和实现高度相关；论文的 224.05 FPS 来自 V100。未经同硬件复测，不做直接速度优越性声明。",
        "- `Pixel IoU/F1` 仅作为诊断补充，不作为与该论文对齐的主结果。",
        "- 本地 Params 是注册参数个数 M；历史 count_parameters 计算的是存储字节 MB。baseline 为0.070913 M、FP32约0.283652 MB，不能把两种单位混比。",
        "- FeedbackSTS 按其训练验证协议用FP32评测，其余模型使用AMP；FP16异常结果不纳入最终表。",
        "",
    ])
    if args.paper_scenario == "noise8":
        lines.append("- 论文参照来自 NUDT-MIRSDT-HiNo；尚未核实本地 Noise8.0_FJY 的噪声生成参数和样本与其完全一致，不作严格同数据集复现声明。")
    markdown_path = args.control_root / "PAPER_METRICS.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"PAPER_METRICS completed={len(completed)}/{len(rows)} "
        f"csv={csv_path} markdown={markdown_path}"
    )


if __name__ == "__main__":
    main()
