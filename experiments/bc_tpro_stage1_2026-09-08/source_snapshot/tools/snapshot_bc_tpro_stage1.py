#!/usr/bin/env python3
"""Freeze a completed BC-TPro stage-1 experiment without copying datasets.

The command is deliberately read-only with respect to training outputs.  It
refuses to freeze a partial experiment, verifies every run-local source
snapshot, copies the small reproducibility sources into ``source_snapshot``,
and records hashes for checkpoints and metrics without copying either.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = 1
EXPECTED_RUN_COUNT = 12
EXPECTED_METRIC_COUNT = 24
EXPECTED_EPOCH = 32
EXPECTED_SEQUENCE_LENGTH = 40
EXPECTED_VALIDATION_SEQUENCE_COUNT = 16
EXPECTED_MODEL = "DeepPro-Plus_BCTPro"
EXPECTED_CONDITIONS = {
    "clean_val": "NUDT-MIRSDT",
    "noise8_val": "NUDT-MIRSDT-Noise8.0_FJY",
}
MANIFEST_COLUMNS = (
    "run_id",
    "wave",
    "model",
    "structure_variant",
    "seed",
    "gpu",
    "log_dir",
)

# These files are sufficient to reconstruct the exact stage-1 train/evaluate
# path and its analysis.  Dataset images and checkpoint payloads are explicitly
# excluded from the source archive.
CRITICAL_SOURCE_PATHS = (
    "train.py",
    "test.py",
    "runtime_utils.py",
    "sequence_utils.py",
    "ShootingRules.py",
    "write_results.py",
    "data_utils/TrainDataLoader.py",
    "data_utils/TestDataLoader.py",
    "data_utils/loader_utils.py",
    "networks/__init__.py",
    "networks/models/__init__.py",
    "networks/models/DeepPro-Plus_BCTPro.py",
    "networks/layers/__init__.py",
    "networks/layers/basic.py",
    "networks/layers/TPro.py",
    "networks/layers/bc_tpro_adapter.py",
    "networks/losses/__init__.py",
    "networks/losses/segmentation_losses.py",
    "tools_forSatVideoIRSTD/seg2centroid_txt.py",
    "tools/project_runtime_env.sh",
    "tools/create_nudt_sequence_split.py",
    "tools/run_bc_tpro_stage1.sh",
    "tools/analyze_bc_tpro_stage1.py",
    "tools/snapshot_bc_tpro_stage1.py",
    "tests/test_bc_tpro_stage1.py",
    "tests/test_analyze_bc_tpro_stage1.py",
    "tests/test_snapshot_bc_tpro_stage1.py",
    "docs/environment_sjyPID_2026-08-24.yml",
    "experiments/bc_tpro_stage1_2026-09-08/manifest.tsv",
    "experiments/bc_tpro_stage1_2026-09-08/splits/split_manifest.json",
    "experiments/bc_tpro_stage1_2026-09-08/splits/train_sequences.txt",
    "experiments/bc_tpro_stage1_2026-09-08/splits/val_sequences.txt",
    "experiments/bc_tpro_stage1_2026-09-08/EXPERIMENT_PLAN.md",
    "experiments/bc_tpro_stage1_2026-09-08/PRECHECK.md",
    "experiments/bc_tpro_stage1_2026-09-08/CUDA_INCIDENT.md",
    "experiments/bc_tpro_stage1_2026-09-08/PROTOCOL_AMENDMENT_2026-09-09.md",
    "experiments/bc_tpro_stage1_2026-09-08/REPRODUCIBILITY_README.md",
)

RUN_SNAPSHOT_SOURCES = {
    "model": "networks/models/DeepPro-Plus_BCTPro.py",
    "adapter": "networks/layers/bc_tpro_adapter.py",
    "loss": "networks/losses/segmentation_losses.py",
}


class SnapshotError(RuntimeError):
    """Raised when the experiment is not safe to freeze."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(str(path.resolve()), str(root.resolve()))


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SnapshotError("Missing %s: %s" % (label, path))
    if path.is_symlink():
        raise SnapshotError("Refusing symlink for %s: %s" % (label, path))


def file_record(path: Path, repo_root: Path) -> Dict[str, Any]:
    _require_regular_file(path, "artifact")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "repo_relative_path": _relative_path(path, repo_root),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def read_manifest(path: Path) -> List[Dict[str, str]]:
    _require_regular_file(path, "experiment manifest")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise SnapshotError(
                "Unexpected manifest columns: %r; expected %r"
                % (reader.fieldnames, MANIFEST_COLUMNS)
            )
        rows = [dict(row) for row in reader]

    if len(rows) != EXPECTED_RUN_COUNT:
        raise SnapshotError(
            "Expected %d manifest runs, found %d"
            % (EXPECTED_RUN_COUNT, len(rows))
        )
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise SnapshotError("Manifest run_id values are not unique.")
    log_dirs = [row["log_dir"] for row in rows]
    if len(set(log_dirs)) != len(log_dirs):
        raise SnapshotError("Manifest log_dir values are not unique.")

    expected_variants = {
        "none": {47, 49, 51},
        "temporal_control": {47, 49, 51},
        "center_multiscale": {47, 49, 51},
        "center_ring": {47, 49, 51},
    }
    observed: Dict[str, set] = {}
    for row in rows:
        if row["model"] != EXPECTED_MODEL:
            raise SnapshotError(
                "Run %s has unexpected model %s."
                % (row["run_id"], row["model"])
            )
        log_dir = PurePosixPath(row["log_dir"])
        if log_dir.is_absolute() or ".." in log_dir.parts:
            raise SnapshotError(
                "Run %s has unsafe log_dir %s."
                % (row["run_id"], row["log_dir"])
            )
        try:
            seed = int(row["seed"])
            wave = int(row["wave"])
            gpu = int(row["gpu"])
        except ValueError as error:
            raise SnapshotError(
                "Run %s has a non-integer seed/wave/gpu."
                % row["run_id"]
            ) from error
        if wave not in {1, 2, 3, 4} or gpu not in {0, 1, 2}:
            raise SnapshotError(
                "Run %s has unexpected wave=%d or gpu=%d."
                % (row["run_id"], wave, gpu)
            )
        observed.setdefault(row["structure_variant"], set()).add(seed)
    if observed != expected_variants:
        raise SnapshotError(
            "Manifest variant/seed matrix differs from the preregistration: %r"
            % observed
        )
    return rows


def _parse_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise SnapshotError(
                "Malformed status line %s:%d" % (path, line_number)
            )
        key, value = raw.split("=", 1)
        if not key or key in values:
            raise SnapshotError(
                "Duplicate/empty status key %s:%d" % (path, line_number)
            )
        values[key] = value
    return values


def audit_completion(
    jobs: Sequence[Mapping[str, str]], queue_root: Path, repo_root: Path
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    status_root = queue_root / "status"
    completion_records: List[Dict[str, Any]] = []
    stable_paths: List[Path] = []
    for job in jobs:
        run_id = job["run_id"]
        done_path = status_root / (run_id + ".done")
        running_path = status_root / (run_id + ".running")
        failed_path = status_root / (run_id + ".failed")
        if running_path.exists() or failed_path.exists():
            raise SnapshotError(
                "Run %s is not cleanly complete (running/failed marker exists)."
                % run_id
            )
        _require_regular_file(done_path, "done marker for %s" % run_id)
        values = _parse_key_value_file(done_path)
        for key in ("run_id", "wave", "gpu", "seed", "finished_at"):
            if key not in values:
                raise SnapshotError(
                    "Done marker %s lacks %s." % (done_path, key)
                )
        for key in ("run_id", "wave", "gpu", "seed"):
            if values[key] != job[key]:
                raise SnapshotError(
                    "Done marker identity mismatch for %s: %s=%r, expected %r"
                    % (run_id, key, values[key], job[key])
                )
        completion_records.append({
            "run_id": run_id,
            "status": "done",
            "marker": file_record(done_path, repo_root),
            "status_fields": values,
        })
        stable_paths.append(done_path)
    return completion_records, stable_paths


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotError("Invalid JSON in %s: %s" % (path, error)) from error
    if not isinstance(payload, dict):
        raise SnapshotError("Expected JSON object in %s." % path)
    return payload


def _read_sequence_list(path: Path) -> List[str]:
    _require_regular_file(path, "sequence list")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names or len(names) != len(set(names)):
        raise SnapshotError("Sequence list is empty or contains duplicates: %s" % path)
    return names


def _sequence_names_from_frame_list(path: Path) -> List[str]:
    _require_regular_file(path, "dataset frame list")
    names: List[str] = []
    seen = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        entry = raw.strip().replace("\\", "/")
        if not entry:
            continue
        parsed = PurePosixPath(entry)
        if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
            raise SnapshotError(
                "Unsafe dataset-list entry %s:%d" % (path, line_number)
            )
        name = parsed.parts[0]
        if name not in seen:
            seen.add(name)
            names.append(name)
    if not names:
        raise SnapshotError("Dataset frame list is empty: %s" % path)
    return names


def audit_data_lists(
    experiment_root: Path,
    clean_data_root: Path,
    noise_data_root: Path,
    repo_root: Path,
) -> Tuple[Dict[str, Any], List[Path], List[str]]:
    split_root = experiment_root / "splits"
    train_split = split_root / "train_sequences.txt"
    val_split = split_root / "val_sequences.txt"
    split_manifest_path = split_root / "split_manifest.json"
    train_names = _read_sequence_list(train_split)
    val_names = _read_sequence_list(val_split)
    if len(train_names) != 64 or len(val_names) != 16:
        raise SnapshotError(
            "Expected a 64/16 train/validation split, got %d/%d."
            % (len(train_names), len(val_names))
        )
    if set(train_names) & set(val_names):
        raise SnapshotError("Fixed train and validation sequence lists overlap.")

    data_paths = {
        "clean_train_index": clean_data_root / "train.txt",
        "clean_test_index": clean_data_root / "test.txt",
        "noise8_train_index": noise_data_root / "train.txt",
        "noise8_test_index": noise_data_root / "test.txt",
        "fixed_train_split": train_split,
        "fixed_validation_split": val_split,
        "split_manifest": split_manifest_path,
    }
    records = {
        name: file_record(path, repo_root) for name, path in data_paths.items()
    }
    clean_train_names = _sequence_names_from_frame_list(
        data_paths["clean_train_index"]
    )
    noise_train_names = _sequence_names_from_frame_list(
        data_paths["noise8_train_index"]
    )
    if clean_train_names != noise_train_names:
        raise SnapshotError("Clean and Noise8 train indices have different sequence order.")
    if set(train_names) | set(val_names) != set(clean_train_names):
        raise SnapshotError(
            "The fixed 64/16 split does not partition the clean training sequences."
        )
    if records["clean_train_index"]["sha256"] != records["noise8_train_index"]["sha256"]:
        raise SnapshotError("Clean and Noise8 train.txt hashes differ.")

    split_manifest = _load_json(split_manifest_path, "split manifest")
    expected_hashes = {
        "source_sha256": records["clean_train_index"]["sha256"],
        "train_sha256": records["fixed_train_split"]["sha256"],
        "validation_sha256": records["fixed_validation_split"]["sha256"],
    }
    for key, expected in expected_hashes.items():
        if split_manifest.get(key) != expected:
            raise SnapshotError(
                "split_manifest.json %s=%r, expected %r"
                % (key, split_manifest.get(key), expected)
            )
    return ({
        "passed": True,
        "train_sequence_count": len(train_names),
        "validation_sequence_count": len(val_names),
        "clean_train_sequence_count": len(clean_train_names),
        "files": records,
        "split_manifest_payload": split_manifest,
    }, list(data_paths.values()), val_names)


def load_checkpoint_identity(path: Path) -> Dict[str, Any]:
    """Load only identity fields on CPU; no CUDA API is called."""
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise SnapshotError(
            "PyTorch is required to audit checkpoint metadata. Run this tool "
            "inside the training environment."
        ) from error
    try:
        checkpoint = torch.load(str(path), map_location="cpu")
    except Exception as error:
        raise SnapshotError("Cannot load checkpoint %s: %s" % (path, error)) from error
    if not isinstance(checkpoint, dict):
        raise SnapshotError("Checkpoint is not a mapping: %s" % path)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise SnapshotError("Checkpoint lacks model_config mapping: %s" % path)
    return {
        "stored_epoch_zero_based": checkpoint.get("epoch"),
        "model_name": checkpoint.get("model_name"),
        "structure_variant": config.get("structure_variant"),
        "model_config": config,
    }


def audit_run_artifacts(
    jobs: Sequence[Mapping[str, str]],
    experiment_root: Path,
    log_root: Path,
    repo_root: Path,
    validation_names: Sequence[str],
    checkpoint_identity_loader: Callable[[Path], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    metrics_root = experiment_root / "metrics"
    expected_metric_names = {
        "%s__%s.json" % (job["run_id"], condition)
        for job in jobs
        for condition in EXPECTED_CONDITIONS
    }
    actual_metric_names = (
        {path.name for path in metrics_root.glob("*.json")}
        if metrics_root.is_dir() else set()
    )
    if actual_metric_names != expected_metric_names:
        missing = sorted(expected_metric_names - actual_metric_names)
        extra = sorted(actual_metric_names - expected_metric_names)
        raise SnapshotError(
            "Metrics set is incomplete or ambiguous: missing=%r extra=%r"
            % (missing, extra)
        )
    if len(actual_metric_names) != EXPECTED_METRIC_COUNT:
        raise SnapshotError(
            "Expected %d metric JSON files, found %d."
            % (EXPECTED_METRIC_COUNT, len(actual_metric_names))
        )

    run_records: List[Dict[str, Any]] = []
    stable_paths: List[Path] = []
    for job in jobs:
        run_dir = (log_root / Path(job["log_dir"])).resolve()
        expected_root = log_root.resolve()
        try:
            run_dir.relative_to(expected_root)
        except ValueError as error:
            raise SnapshotError(
                "Run directory escapes log root: %s" % run_dir
            ) from error
        if not run_dir.is_dir():
            raise SnapshotError("Missing run directory: %s" % run_dir)

        checkpoint_path = run_dir / "checkpoints" / "epoch_32_model.pth"
        checkpoint_record = file_record(checkpoint_path, repo_root)
        checkpoint_identity = checkpoint_identity_loader(checkpoint_path)
        expected_identity = {
            "stored_epoch_zero_based": EXPECTED_EPOCH - 1,
            "model_name": job["model"],
            "structure_variant": job["structure_variant"],
        }
        for key, expected in expected_identity.items():
            if checkpoint_identity.get(key) != expected:
                raise SnapshotError(
                    "Checkpoint identity mismatch for %s: %s=%r, expected %r"
                    % (
                        job["run_id"], key,
                        checkpoint_identity.get(key), expected,
                    )
                )

        metric_records = []
        for condition, expected_dataset in EXPECTED_CONDITIONS.items():
            metric_path = metrics_root / (
                "%s__%s.json" % (job["run_id"], condition)
            )
            payload = _load_json(metric_path, "metric JSON")
            expected_fields = {
                "schema_version": 2,
                "dataset": expected_dataset,
                "model": job["model"],
                "checkpoint_epoch": EXPECTED_EPOCH,
                "sequence_length": EXPECTED_SEQUENCE_LENGTH,
                "sequence_count": EXPECTED_VALIDATION_SEQUENCE_COUNT,
            }
            for key, expected in expected_fields.items():
                if payload.get(key) != expected:
                    raise SnapshotError(
                        "Metric identity mismatch in %s: %s=%r, expected %r"
                        % (metric_path, key, payload.get(key), expected)
                    )
            payload_checkpoint = payload.get("checkpoint")
            if not isinstance(payload_checkpoint, str) or (
                Path(payload_checkpoint).expanduser().resolve()
                != checkpoint_path.resolve()
            ):
                raise SnapshotError(
                    "Metric %s does not reference its epoch-32 checkpoint."
                    % metric_path
                )
            curve_counts = payload.get("curve_counts")
            if not isinstance(curve_counts, dict) or (
                curve_counts.get("sequence_names") != list(validation_names)
            ):
                raise SnapshotError(
                    "Metric %s has the wrong validation sequence list/order."
                    % metric_path
                )
            metric_records.append({
                "condition": condition,
                "dataset": expected_dataset,
                "file": file_record(metric_path, repo_root),
            })
            stable_paths.append(metric_path)

        train_log_path = run_dir / "logs" / (job["model"] + ".txt")
        train_log_record = file_record(train_log_path, repo_root)
        stable_paths.extend([checkpoint_path, train_log_path])
        run_records.append({
            "run_id": job["run_id"],
            "wave": int(job["wave"]),
            "model": job["model"],
            "structure_variant": job["structure_variant"],
            "seed": int(job["seed"]),
            "gpu": int(job["gpu"]),
            "log_dir": job["log_dir"],
            "checkpoint": checkpoint_record,
            "checkpoint_identity": checkpoint_identity,
            "training_log": train_log_record,
            "metrics": metric_records,
        })
    return run_records, stable_paths


def audit_run_source_snapshots(
    jobs: Sequence[Mapping[str, str]],
    run_records: List[Dict[str, Any]],
    log_root: Path,
    repo_root: Path,
) -> Tuple[Dict[str, Any], List[Path]]:
    canonical: Dict[str, Dict[str, Any]] = {}
    for label, relative in RUN_SNAPSHOT_SOURCES.items():
        source = repo_root / relative
        canonical[label] = file_record(source, repo_root)

    record_by_id = {record["run_id"]: record for record in run_records}
    audit_rows = []
    stable_paths: List[Path] = []
    hashes_by_label: Dict[str, set] = {
        label: set() for label in RUN_SNAPSHOT_SOURCES
    }
    for job in jobs:
        run_dir = log_root / Path(job["log_dir"])
        files: Dict[str, Dict[str, Any]] = {}
        for label, canonical_relative in RUN_SNAPSHOT_SOURCES.items():
            snapshot_path = run_dir / Path(canonical_relative).name
            snapshot_record = file_record(snapshot_path, repo_root)
            expected_hash = canonical[label]["sha256"]
            matches = snapshot_record["sha256"] == expected_hash
            if not matches:
                raise SnapshotError(
                    "Run snapshot differs from canonical %s for %s."
                    % (label, job["run_id"])
                )
            snapshot_record["matches_canonical"] = True
            files[label] = snapshot_record
            hashes_by_label[label].add(snapshot_record["sha256"])
            stable_paths.append(snapshot_path)
        row = {"run_id": job["run_id"], "files": files, "passed": True}
        audit_rows.append(row)
        record_by_id[job["run_id"]]["run_source_snapshot"] = row

    unique_hash_count = {
        label: len(values) for label, values in hashes_by_label.items()
    }
    if any(count != 1 for count in unique_hash_count.values()):
        raise SnapshotError(
            "Run-local source snapshots are not identical across 12 runs."
        )
    return ({
        "passed": True,
        "run_count": len(audit_rows),
        "canonical_sources": canonical,
        "unique_hash_count_by_source": unique_hash_count,
        "runs": audit_rows,
    }, stable_paths)


def _run_command(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command), cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=30,
        )
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": "%s: %s" % (type(error).__name__, error),
        }


def _require_git(command: Sequence[str], repo_root: Path) -> bytes:
    completed = subprocess.run(
        ["git"] + list(command), cwd=str(repo_root), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise SnapshotError(
            "git %s failed: %s"
            % (" ".join(command), completed.stderr.decode("utf-8", "replace"))
        )
    return completed.stdout


def collect_git_state(repo_root: Path) -> Dict[str, Any]:
    head = _require_git(("rev-parse", "HEAD"), repo_root).decode().strip()
    branch_name = _require_git(
        ("rev-parse", "--abbrev-ref", "HEAD"), repo_root
    ).decode().strip()
    branch = None if branch_name == "HEAD" else branch_name
    status_bytes = _require_git(
        ("status", "--porcelain=v1", "--untracked-files=all"), repo_root
    )
    diff_bytes = _require_git(
        ("diff", "--binary", "--no-ext-diff", "HEAD", "--"), repo_root
    )
    status = status_bytes.decode("utf-8", errors="surrogateescape").splitlines()
    return {
        "head": head,
        "branch": branch,
        "is_dirty": bool(status_bytes),
        "status_porcelain_v1": status,
        "status_sha256": sha256_bytes(status_bytes),
        "diff_head_binary_sha256": sha256_bytes(diff_bytes),
        "diff_head_binary_size_bytes": len(diff_bytes),
    }


def _package_versions(names: Iterable[str]) -> Dict[str, Any]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python 3.7 fallback
        import importlib_metadata as metadata  # type: ignore
    versions: Dict[str, Any] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment_state(repo_root: Path) -> Dict[str, Any]:
    pip_freeze = _run_command(
        (sys.executable, "-m", "pip", "freeze", "--all"), repo_root
    )
    nvidia_smi = _run_command((
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ), repo_root)
    nvcc = _run_command(("nvcc", "--version"), repo_root)

    torch_info: Dict[str, Any]
    try:
        torch = importlib.import_module("torch")
        torch_info = {
            "version": getattr(torch, "__version__", None),
            "compiled_cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_runtime_was_queried": False,
        }
    except Exception as error:
        torch_info = {"error": "%s: %s" % (type(error).__name__, error)}

    selected_env = {}
    for key in (
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
    ):
        if key in os.environ:
            selected_env[key] = os.environ[key]
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": list(platform.uname()),
        },
        "selected_environment_variables": selected_env,
        "selected_packages": _package_versions((
            "numpy", "scipy", "scikit-image", "scikit-learn",
            "opencv-python", "Pillow", "torch", "torchvision", "swanlab",
        )),
        "pip_freeze": pip_freeze,
        "torch_build": torch_info,
        "gpu_driver_inventory": nvidia_smi,
        "cuda_compiler": nvcc,
    }


def _copy_sources_atomically(
    repo_root: Path, experiment_root: Path
) -> Tuple[List[Dict[str, Any]], bool]:
    snapshot_root = experiment_root / "source_snapshot"
    source_records: List[Dict[str, Any]] = []
    for relative in CRITICAL_SOURCE_PATHS:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise SnapshotError("Unsafe critical-source path: %s" % relative)
        source = repo_root / Path(relative)
        _require_regular_file(source, "critical source")
        source_records.append({
            "path": relative,
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        })

    reused = False
    if snapshot_root.exists():
        if not snapshot_root.is_dir() or snapshot_root.is_symlink():
            raise SnapshotError(
                "source_snapshot exists but is not a regular directory: %s"
                % snapshot_root
            )
        actual_files = sorted(
            path.relative_to(snapshot_root).as_posix()
            for path in snapshot_root.rglob("*") if path.is_file()
        )
        expected_files = sorted(record["path"] for record in source_records)
        if actual_files != expected_files:
            raise SnapshotError(
                "Existing source_snapshot file set differs; refusing overwrite."
            )
        for record in source_records:
            frozen = snapshot_root / Path(record["path"])
            if frozen.is_symlink() or sha256_file(frozen) != record["sha256"]:
                raise SnapshotError(
                    "Existing frozen source differs; refusing overwrite: %s"
                    % frozen
                )
        reused = True
    else:
        temporary = Path(tempfile.mkdtemp(
            prefix=".source_snapshot.", dir=str(experiment_root)
        ))
        try:
            for record in source_records:
                source = repo_root / Path(record["path"])
                destination = temporary / Path(record["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(destination))
                if sha256_file(destination) != record["sha256"]:
                    raise SnapshotError("Copy verification failed for %s" % source)
            # Detect source edits that raced with the copy.
            for record in source_records:
                source = repo_root / Path(record["path"])
                if sha256_file(source) != record["sha256"]:
                    raise SnapshotError("Source changed during snapshot: %s" % source)
            os.replace(str(temporary), str(snapshot_root))
        except Exception:
            if temporary.exists():
                shutil.rmtree(str(temporary))
            raise
    return source_records, reused


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _assert_paths_stable(paths: Sequence[Path], initial: Mapping[str, str]) -> None:
    for path in paths:
        key = str(path.resolve())
        if not path.is_file() or sha256_file(path) != initial[key]:
            raise SnapshotError("Artifact changed while freezing: %s" % path)


def create_reproducibility_snapshot(
    repo_root: Path,
    experiment_root: Path,
    log_root: Path,
    queue_root: Path,
    clean_data_root: Path,
    noise_data_root: Path,
    checkpoint_identity_loader: Callable[[Path], Dict[str, Any]] = load_checkpoint_identity,
    git_state_collector: Callable[[Path], Dict[str, Any]] = collect_git_state,
    environment_collector: Callable[[Path], Dict[str, Any]] = collect_environment_state,
) -> Dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    experiment_root = experiment_root.expanduser().resolve()
    log_root = log_root.expanduser().resolve()
    queue_root = queue_root.expanduser().resolve()
    clean_data_root = clean_data_root.expanduser().resolve()
    noise_data_root = noise_data_root.expanduser().resolve()
    if not experiment_root.is_dir():
        raise SnapshotError("Missing experiment root: %s" % experiment_root)

    jobs = read_manifest(experiment_root / "manifest.tsv")
    completion, completion_paths = audit_completion(jobs, queue_root, repo_root)
    data_audit, data_paths, validation_names = audit_data_lists(
        experiment_root, clean_data_root, noise_data_root, repo_root
    )
    runs, artifact_paths = audit_run_artifacts(
        jobs, experiment_root, log_root, repo_root, validation_names,
        checkpoint_identity_loader,
    )
    source_audit, run_snapshot_paths = audit_run_source_snapshots(
        jobs, runs, log_root, repo_root
    )

    stable_paths = list(dict.fromkeys(
        completion_paths + data_paths + artifact_paths + run_snapshot_paths
    ))
    initial_hashes = {
        str(path.resolve()): sha256_file(path) for path in stable_paths
    }

    git_state = git_state_collector(repo_root)
    environment = environment_collector(repo_root)
    source_records, reused = _copy_sources_atomically(
        repo_root, experiment_root
    )
    _assert_paths_stable(stable_paths, initial_hashes)

    sums_payload = "".join(
        "%s  %s\n" % (record["sha256"], record["path"])
        for record in sorted(source_records, key=lambda item: item["path"])
    ).encode("utf-8")
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bc_tpro_stage1_reproducibility_freeze",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root_at_freeze": str(repo_root),
        "experiment_root_at_freeze": str(experiment_root),
        "completion_audit": {
            "passed": True,
            "expected_run_count": EXPECTED_RUN_COUNT,
            "completed_run_count": len(completion),
            "expected_metric_count": EXPECTED_METRIC_COUNT,
            "metric_count": sum(len(run["metrics"]) for run in runs),
            "runs": completion,
        },
        "source_snapshot": {
            "path": "source_snapshot",
            "file_count": len(source_records),
            "reused_existing_identical_snapshot": reused,
            "files": source_records,
            "sha256s_file": "SOURCE_SHA256SUMS",
            "sha256s_file_sha256": sha256_bytes(sums_payload),
        },
        "run_snapshot_consistency_audit": source_audit,
        "data_list_audit": data_audit,
        "artifact_policy": {
            "datasets_copied": False,
            "checkpoints_copied": False,
            "metrics_copied": False,
            "checkpoint_and_metric_files_are_hashed_by_reference": True,
        },
        "runs": runs,
        "git": git_state,
        "environment": environment,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")

    # Write the checksum list first: the JSON records its digest.  Each replace
    # is atomic, so readers never observe a partially written file.
    _atomic_write(experiment_root / "SOURCE_SHA256SUMS", sums_payload)
    _atomic_write(
        experiment_root / "REPRODUCIBILITY_MANIFEST.json", manifest_payload
    )
    return manifest


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[1]
    default_experiment = (
        default_repo / "experiments" / "bc_tpro_stage1_2026-09-08"
    )
    parser = argparse.ArgumentParser(
        description="Freeze a completed BC-TPro stage-1 experiment (CPU only)."
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--experiment-root", type=Path, default=default_experiment)
    parser.add_argument(
        "--log-root", type=Path, default=default_repo / "log" / "sem_seg"
    )
    parser.add_argument(
        "--queue-root", type=Path,
        default=(
            default_repo / "log" / "sem_seg" / "_queues"
            / "bc_tpro_stage1_2026-09-08"
        ),
    )
    parser.add_argument(
        "--clean-data-root", type=Path,
        default=default_repo.parent / "datasets" / "NUDT-MIRSDT",
    )
    parser.add_argument(
        "--noise-data-root", type=Path,
        default=(
            default_repo.parent / "datasets" / "NUDT-MIRSDT-Noise8.0_FJY"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    try:
        manifest = create_reproducibility_snapshot(
            repo_root=args.repo_root,
            experiment_root=args.experiment_root,
            log_root=args.log_root,
            queue_root=args.queue_root,
            clean_data_root=args.clean_data_root,
            noise_data_root=args.noise_data_root,
        )
    except SnapshotError as error:
        print("SNAPSHOT REFUSED: %s" % error, file=sys.stderr)
        return 2
    print(
        "Snapshot complete: runs=%d metrics=%d sources=%d"
        % (
            manifest["completion_audit"]["completed_run_count"],
            manifest["completion_audit"]["metric_count"],
            manifest["source_snapshot"]["file_count"],
        )
    )
    print(
        "Manifest: %s"
        % (args.experiment_root / "REPRODUCIBILITY_MANIFEST.json")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
