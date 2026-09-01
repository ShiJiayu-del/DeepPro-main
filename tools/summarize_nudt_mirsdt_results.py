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
DEFAULT_SAVE_ROOT = REPO_ROOT / "log" / "nudt_mirsdt_all_models_2026-09-01"
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


def main():
    args = parse_args()
    manifest_path = args.control_root / "manifest.tsv"
    rows = []
    for job in read_manifest(manifest_path):
        run_id = job["run_id"]
        experiment_dir = args.save_root / "sem_seg" / run_id
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
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(ranked, start=1):
            output = dict(row)
            output["rank"] = index if row["best_f1"] is not None else ""
            for name in (
                "best_iou", "best_precision", "best_recall", "best_f1",
                "latest_f1",
            ):
                output[name] = fmt(output[name])
            writer.writerow({name: output.get(name, "") for name in fieldnames})

    completed = sum(row["status"] == "done" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    running = sum(row["status"] == "running" for row in rows)
    markdown = [
        "# NUDT-MIRSDT 全历史模型结果",
        "",
        f"进度：完成 {completed}/{len(rows)}，运行中 {running}，失败 {failed}。",
        "",
        "排名按验证集最佳 pixel F1；IoU、Precision、Recall 均取自同一个最佳 F1 epoch。",
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
        "## 解释边界",
        "",
        "- 这些是 NUDT-MIRSDT 官方 `test.txt` 划分上的本地 pixel 指标，不是原比赛网站分数。",
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
