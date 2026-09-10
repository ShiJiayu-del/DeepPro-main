"""CPU-only contract tests for the BC-TPro reproducibility freezer."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import snapshot_bc_tpro_stage1 as snapshot


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SnapshotFixture:
    variants = (
        (1, "b1_none", "none"),
        (2, "c0_temporal_control", "temporal_control"),
        (3, "c1_center_multiscale", "center_multiscale"),
        (4, "c2_center_ring", "center_ring"),
    )

    def __init__(self, root):
        self.root = Path(root)
        self.repo = self.root / "DeepPro-main"
        self.experiment = (
            self.repo / "experiments" / "bc_tpro_stage1_2026-09-08"
        )
        self.log_root = self.repo / "log" / "sem_seg"
        self.queue = (
            self.log_root / "_queues" / "bc_tpro_stage1_2026-09-08"
        )
        self.clean = self.root / "datasets" / "NUDT-MIRSDT"
        self.noise = self.root / "datasets" / "NUDT-MIRSDT-Noise8.0_FJY"
        self.jobs = []
        self.by_log_dir = {}
        self.critical_sources = tuple(snapshot.RUN_SNAPSHOT_SOURCES.values())
        self._build()

    def _write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _build(self):
        canonical_content = {
            "networks/models/DeepPro-Plus_BCTPro.py": "MODEL_SOURCE\n",
            "networks/layers/bc_tpro_adapter.py": "ADAPTER_SOURCE\n",
            "networks/losses/segmentation_losses.py": "LOSS_SOURCE\n",
        }
        for relative, content in canonical_content.items():
            self._write(self.repo / relative, content)

        manifest_lines = ["\t".join(snapshot.MANIFEST_COLUMNS)]
        seeds = (47, 49, 51)
        for wave, prefix, variant in self.variants:
            for gpu, seed in enumerate(seeds):
                run_id = "%s_seed%d" % (prefix, seed)
                log_dir = "2026-09-08/%s" % run_id
                job = {
                    "run_id": run_id,
                    "wave": str(wave),
                    "model": snapshot.EXPECTED_MODEL,
                    "structure_variant": variant,
                    "seed": str(seed),
                    "gpu": str(gpu),
                    "log_dir": log_dir,
                }
                self.jobs.append(job)
                self.by_log_dir[log_dir] = job
                manifest_lines.append("\t".join(
                    job[column] for column in snapshot.MANIFEST_COLUMNS
                ))
                self._build_run(job, canonical_content)
        self._write(
            self.experiment / "manifest.tsv",
            "\n".join(manifest_lines) + "\n",
        )
        self._build_data_lists()

    def _build_run(self, job, canonical_content):
        run_id = job["run_id"]
        done = self.queue / "status" / (run_id + ".done")
        self._write(done, (
            "run_id={run_id}\nwave={wave}\ngpu={gpu}\nseed={seed}\n"
            "started_at=2026-09-09T00:00:00+08:00\n"
            "finished_at=2026-09-09T00:20:00+08:00\n"
            "elapsed_seconds=1200\n"
        ).format(**job))

        run_dir = self.log_root / job["log_dir"]
        checkpoint = run_dir / "checkpoints" / "epoch_32_model.pth"
        self._write(checkpoint, "FAKE CHECKPOINT %s\n" % run_id)
        self._write(
            run_dir / "logs" / (snapshot.EXPECTED_MODEL + ".txt"),
            "scratch-only fixture for %s\n" % run_id,
        )
        for relative, content in canonical_content.items():
            self._write(run_dir / Path(relative).name, content)

        validation_names = ["Sequence%d" % value for value in range(65, 81)]
        for condition, dataset in snapshot.EXPECTED_CONDITIONS.items():
            payload = {
                "schema_version": 2,
                "dataset": dataset,
                "model": snapshot.EXPECTED_MODEL,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_epoch": 32,
                "sequence_length": 40,
                "sequence_count": 16,
                "curve_counts": {"sequence_names": validation_names},
            }
            metric = (
                self.experiment / "metrics"
                / ("%s__%s.json" % (run_id, condition))
            )
            self._write(metric, json.dumps(payload, sort_keys=True) + "\n")

    def _build_data_lists(self):
        train_names = ["Sequence%d" % value for value in range(1, 65)]
        val_names = ["Sequence%d" % value for value in range(65, 81)]
        train_split = self.experiment / "splits" / "train_sequences.txt"
        val_split = self.experiment / "splits" / "val_sequences.txt"
        self._write(train_split, "\n".join(train_names) + "\n")
        self._write(val_split, "\n".join(val_names) + "\n")

        frame_lines = "".join(
            "%s/Mix/00001.mat\n" % name for name in train_names + val_names
        )
        clean_train = self.clean / "train.txt"
        noise_train = self.noise / "train.txt"
        self._write(clean_train, frame_lines)
        self._write(noise_train, frame_lines)
        self._write(self.clean / "test.txt", "Sequence85/Mix/00001.mat\n")
        self._write(self.noise / "test.txt", "Sequence85/Mix/00001.mat\n")
        split_manifest = {
            "source_sha256": _sha(clean_train),
            "train_sha256": _sha(train_split),
            "validation_sha256": _sha(val_split),
        }
        self._write(
            self.experiment / "splits" / "split_manifest.json",
            json.dumps(split_manifest, sort_keys=True) + "\n",
        )

    def checkpoint_identity(self, path):
        log_dir = path.parents[1].relative_to(self.log_root).as_posix()
        job = self.by_log_dir[log_dir]
        return {
            "stored_epoch_zero_based": 31,
            "model_name": snapshot.EXPECTED_MODEL,
            "structure_variant": job["structure_variant"],
            "model_config": {
                "structure_variant": job["structure_variant"],
                "eval_chunk_rows": 32,
            },
        }

    @staticmethod
    def git_state(_repo):
        return {
            "head": "a" * 40,
            "branch": "fixture",
            "is_dirty": True,
            "status_porcelain_v1": ["?? fixture"],
            "status_sha256": "b" * 64,
            "diff_head_binary_sha256": "c" * 64,
            "diff_head_binary_size_bytes": 7,
        }

    @staticmethod
    def environment(_repo):
        return {
            "python": {"version": "fixture"},
            "torch_build": {"version": "fixture", "cudnn_version": 0},
            "gpu_driver_inventory": {"returncode": 0, "stdout": "fixture"},
        }

    def freeze(self):
        with mock.patch.object(
            snapshot, "CRITICAL_SOURCE_PATHS", self.critical_sources
        ):
            return snapshot.create_reproducibility_snapshot(
                repo_root=self.repo,
                experiment_root=self.experiment,
                log_root=self.log_root,
                queue_root=self.queue,
                clean_data_root=self.clean,
                noise_data_root=self.noise,
                checkpoint_identity_loader=self.checkpoint_identity,
                git_state_collector=self.git_state,
                environment_collector=self.environment,
            )


class ReproducibilitySnapshotTests(unittest.TestCase):
    def test_complete_fixture_is_frozen_without_large_artifact_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SnapshotFixture(temporary)
            manifest = fixture.freeze()

            self.assertEqual(
                manifest["completion_audit"]["completed_run_count"], 12
            )
            self.assertEqual(manifest["completion_audit"]["metric_count"], 24)
            self.assertTrue(
                manifest["run_snapshot_consistency_audit"]["passed"]
            )
            self.assertEqual(
                manifest["run_snapshot_consistency_audit"]["run_count"], 12
            )
            self.assertFalse(manifest["artifact_policy"]["datasets_copied"])
            self.assertFalse(manifest["artifact_policy"]["checkpoints_copied"])

            frozen_root = fixture.experiment / "source_snapshot"
            actual_frozen = sorted(
                path.relative_to(frozen_root).as_posix()
                for path in frozen_root.rglob("*") if path.is_file()
            )
            self.assertEqual(actual_frozen, sorted(fixture.critical_sources))
            self.assertFalse(list(frozen_root.rglob("*.pth")))
            self.assertFalse(list(frozen_root.rglob("*__clean_val.json")))

            output_manifest = json.loads(
                (fixture.experiment / "REPRODUCIBILITY_MANIFEST.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(output_manifest["schema_version"], 1)
            sums = (
                fixture.experiment / "SOURCE_SHA256SUMS"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(sums), len(fixture.critical_sources))
            self.assertEqual(sums, sorted(sums, key=lambda line: line.split("  ", 1)[1]))
            self.assertFalse(list(fixture.experiment.glob(".source_snapshot.*")))
            self.assertFalse(list(fixture.experiment.glob(".REPRODUCIBILITY_MANIFEST.json.*")))

    def test_partial_run_refuses_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SnapshotFixture(temporary)
            missing = (
                fixture.queue / "status"
                / (fixture.jobs[-1]["run_id"] + ".done")
            )
            missing.unlink()
            with self.assertRaises(snapshot.SnapshotError):
                fixture.freeze()
            self.assertFalse((fixture.experiment / "source_snapshot").exists())
            self.assertFalse(
                (fixture.experiment / "REPRODUCIBILITY_MANIFEST.json").exists()
            )

    def test_missing_metric_refuses_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SnapshotFixture(temporary)
            missing = (
                fixture.experiment / "metrics"
                / (fixture.jobs[-1]["run_id"] + "__noise8_val.json")
            )
            missing.unlink()
            with self.assertRaises(snapshot.SnapshotError):
                fixture.freeze()
            self.assertFalse((fixture.experiment / "source_snapshot").exists())

    def test_run_local_source_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SnapshotFixture(temporary)
            run_dir = fixture.log_root / fixture.jobs[3]["log_dir"]
            (run_dir / "bc_tpro_adapter.py").write_text(
                "CHANGED\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                snapshot.SnapshotError, "differs from canonical"
            ):
                fixture.freeze()
            self.assertFalse((fixture.experiment / "source_snapshot").exists())

    def test_identical_existing_source_snapshot_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SnapshotFixture(temporary)
            first = fixture.freeze()
            second = fixture.freeze()
            self.assertFalse(
                first["source_snapshot"]["reused_existing_identical_snapshot"]
            )
            self.assertTrue(
                second["source_snapshot"]["reused_existing_identical_snapshot"]
            )


if __name__ == "__main__":
    unittest.main()
