#!/usr/bin/env python3
"""Summarize comparable NUDT-MIRSDT validation metrics from training logs."""

import argparse
import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_ROOT = (
    REPO_ROOT / "experiments" / "nudt_mirsdt_all_models_2026-09-01"
)
DEFAULT_SAVE_ROOT = REPO_ROOT / "log" / "sem_seg" / "_queues" / "nudt_mirsdt_all_models_2026-09-01"
DEFAULT_BASELINE_RUN_ID = "deeppro_plus"
METRIC_PATTERNS = {
    "loss": re.compile(r"Eval mean loss: ([0-9.eE+-]+)"),
    "iou": re.compile(r"Eval avg class IoU of prediction: ([0-9.eE+-]+)"),
    "precision": re.compile(r"Eval pixel precision: ([0-9.eE+-]+)"),
    "recall": re.compile(r"Eval pixel recall: ([0-9.eE+-]+)"),
    "f1": re.compile(r"Eval pixel F1: ([0-9.eE+-]+)"),
}
EPOCH_PATTERN = re.compile(r"---- EPOCH ([0-9]+) EVALUATION ----")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--dataset-label", default="NUDT-MIRSDT")
    parser.add_argument(
        "--baseline-run-id",
        default=DEFAULT_BASELINE_RUN_ID,
        help="Completed run used for same-protocol metric deltas.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_evaluations(log_path):
    if not log_path.is_file():
        return []
    evaluations = []
    current = None
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current = {"epoch": int(epoch_match.group(1))}
                evaluations.append(current)
                continue
            if current is None:
                continue
            for name, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    current[name] = float(match.group(1))
    return [item for item in evaluations if all(key in item for key in METRIC_PATTERNS)]


def read_status(status_root, run_id):
    for state in ("done", "failed", "running"):
        path = status_root / f"{run_id}.{state}"
        if path.is_file():
            values = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    values[key] = value
            return state, values
    return "pending", {}


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def fmt_delta(value):
    return "" if value is None else f"{value:+.6f}"


def main():
    args = parse_args()
    manifest_path = args.control_root / "manifest.tsv"
    rows = []
    for job in read_manifest(manifest_path):
        run_id = job["run_id"]
        experiment_dir = (
            REPO_ROOT / "log" / "sem_seg" / job["log_dir"]
            if job.get("log_dir") else args.save_root / "sem_seg" / run_id
        )
        log_path = experiment_dir / "logs" / f"{job['model']}.txt"
        evaluations = parse_evaluations(log_path)
        status, status_values = read_status(args.save_root / "status", run_id)
        best = max(evaluations, key=lambda item: item["f1"]) if evaluations else None
        latest = evaluations[-1] if evaluations else None
        rows.append({
            **job,
            "status": status,
            "evaluations": len(evaluations),
            "best_epoch": best["epoch"] if best else None,
            "best_iou": best["iou"] if best else None,
            "best_precision": best["precision"] if best else None,
            "best_recall": best["recall"] if best else None,
            "best_f1": best["f1"] if best else None,
            "latest_epoch": latest["epoch"] if latest else None,
            "latest_f1": latest["f1"] if latest else None,
            "elapsed_seconds": status_values.get("elapsed_seconds", ""),
        })

    baseline = next(
        (row for row in rows if row["run_id"] == args.baseline_run_id),
        None,
    )
    if baseline is None:
        raise ValueError(
            f"Baseline run is absent from manifest: {args.baseline_run_id}"
        )
    if baseline["best_f1"] is None or baseline["latest_f1"] is None:
        raise ValueError(
            f"Baseline run has no complete metrics: {args.baseline_run_id}"
        )
    comparison_metrics = (
        "best_iou",
        "best_precision",
        "best_recall",
        "best_f1",
        "latest_f1",
    )
    for row in rows:
        row["baseline_run_id"] = args.baseline_run_id
        for name in comparison_metrics:
            value = row[name]
            row[f"delta_{name}"] = (
                value - baseline[name] if value is not None else None
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["best_f1"] is not None,
            row["best_f1"] if row["best_f1"] is not None else -1.0,
        ),
        reverse=True,
    )
    csv_path = args.control_root / "results.csv"
    fieldnames = [
        "rank", "run_id", "model", "structure_variant", "loss", "status",
        "evaluations", "best_epoch", "best_iou", "best_precision",
        "best_recall", "best_f1", "latest_epoch", "latest_f1",
        "baseline_run_id", "delta_best_iou", "delta_best_precision",
        "delta_best_recall", "delta_best_f1", "delta_latest_f1",
        "elapsed_seconds", "log_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(ranked, start=1):
            output = dict(row)
            output["rank"] = index if row["best_f1"] is not None else ""
            for name in (
                "best_iou", "best_precision", "best_recall", "best_f1",
                "latest_f1",
            ):
                output[name] = fmt(output[name])
            for name in (
                "delta_best_iou", "delta_best_precision",
                "delta_best_recall", "delta_best_f1", "delta_latest_f1",
            ):
                output[name] = fmt_delta(output[name])
            writer.writerow({name: output.get(name, "") for name in fieldnames})

    completed = sum(row["status"] == "done" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    running = sum(row["status"] == "running" for row in rows)
    markdown = [
        f"# {args.dataset_label} 训练筛选指标",
        "",
        "> 本表的 pixel IoU/F1 只用于训练诊断和结构初筛。论文主结果请使用同目录的 `PAPER_METRICS.md`（Pd/Fa/AUC）。",
        "",
        f"进度：完成 {completed}/{len(rows)}，运行中 {running}，失败 {failed}。",
        "",
        "筛选顺序按验证集最佳 pixel F1；IoU、Precision、Recall 均取自同一个最佳 F1 epoch。",
        "",
        "| Rank | Run | Model / Variant | Loss | Status | Epoch | IoU | Precision | Recall | F1 | Final F1 |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        model_label = row["model"]
        if row["structure_variant"] != "-":
            model_label += " / " + row["structure_variant"]
        rank = str(index) if row["best_f1"] is not None else "-"
        markdown.append(
            "| {rank} | `{run}` | {model} | `{loss}` | {status} | {epoch} | "
            "{iou} | {precision} | {recall} | {f1} | {latest_f1} |".format(
                rank=rank,
                run=row["run_id"],
                model=model_label,
                loss=row["loss"],
                status=row["status"],
                epoch=row["best_epoch"] or "-",
                iou=fmt(row["best_iou"]) or "-",
                precision=fmt(row["best_precision"]) or "-",
                recall=fmt(row["best_recall"]) or "-",
                f1=fmt(row["best_f1"]) or "-",
                latest_f1=fmt(row["latest_f1"]) or "-",
            )
        )
    markdown.extend([
        "",
        "## 与 baseline 对比",
        "",
        "baseline 固定为 `deeppro_plus`（DeepPro-Plus）：它是 BRTD 系列的直接父网络。",
        "对比继续使用上述同一口径；`Δ` 为当前模型减 baseline，正值表示提高。",
        "",
        "| Rank | Run | ΔIoU | ΔPrecision | ΔRecall | ΔF1 | ΔFinal F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for index, row in enumerate(ranked, start=1):
        rank = str(index) if row["best_f1"] is not None else "-"
        markdown.append(
            "| {rank} | `{run}` | {iou} | {precision} | {recall} | "
            "{f1} | {latest_f1} |".format(
                rank=rank,
                run=row["run_id"],
                iou=fmt_delta(row["delta_best_iou"]) or "-",
                precision=fmt_delta(row["delta_best_precision"]) or "-",
                recall=fmt_delta(row["delta_best_recall"]) or "-",
                f1=fmt_delta(row["delta_best_f1"]) or "-",
                latest_f1=fmt_delta(row["delta_latest_f1"]) or "-",
            )
        )
    best = ranked[0]
    markdown.extend([
        "",
        "### 当前最佳模型相对 baseline",
        "",
        "| Run | IoU | Precision | Recall | F1 | Final F1 |",
        "|---|---:|---:|---:|---:|---:|",
        "| `{run}` | {iou} | {precision} | {recall} | {f1} | {latest_f1} |".format(
            run=baseline["run_id"],
            iou=fmt(baseline["best_iou"]),
            precision=fmt(baseline["best_precision"]),
            recall=fmt(baseline["best_recall"]),
            f1=fmt(baseline["best_f1"]),
            latest_f1=fmt(baseline["latest_f1"]),
        ),
        "| `{run}` | {iou} | {precision} | {recall} | {f1} | {latest_f1} |".format(
            run=best["run_id"],
            iou=fmt(best["best_iou"]),
            precision=fmt(best["best_precision"]),
            recall=fmt(best["best_recall"]),
            f1=fmt(best["best_f1"]),
            latest_f1=fmt(best["latest_f1"]),
        ),
        "| 绝对变化 | {iou} | {precision} | {recall} | {f1} | {latest_f1} |".format(
            iou=fmt_delta(best["delta_best_iou"]),
            precision=fmt_delta(best["delta_best_precision"]),
            recall=fmt_delta(best["delta_best_recall"]),
            f1=fmt_delta(best["delta_best_f1"]),
            latest_f1=fmt_delta(best["delta_latest_f1"]),
        ),
    ])
    markdown.extend([
        "",
        "## 解释边界",
        "",
        f"- 这些是 {args.dataset_label} 的 `test.txt` 划分上的本地 pixel 指标，不是原比赛网站分数。",
        "- baseline 为相同数据、seed、训练轮数、损失和阈值协议下的 `DeepPro-Plus`，不是 SatVideoIRSDT_v1 的历史网站 baseline。",
        "- 所有任务均从零初始化；PointCenter 使用专用中心损失，跨损失比较需谨慎。",
        "- 未完成任务不参与排名；完整配置见 `manifest.tsv` 与训练目录中的日志。",
        "",
    ])
    markdown_path = args.control_root / "RESULTS.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    if not args.quiet:
        print(
            f"SUMMARY completed={completed}/{len(rows)} running={running} "
            f"failed={failed} csv={csv_path} markdown={markdown_path}"
        )


if __name__ == "__main__":
    main()
