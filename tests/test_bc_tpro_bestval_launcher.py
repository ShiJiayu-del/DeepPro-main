"""CPU-only checks for the seven-run best-validation launcher."""

import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'bestval_launcher_test_subject', REPO_ROOT / 'tools/run_bc_tpro_bestval.py'
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def option(command, flag):
    return command[command.index(flag) + 1]


class BestValidationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'repo'
        self.root.mkdir()
        self.runner = launcher.Launcher(self.root, sys.executable)
        self.jobs = launcher.expected_manifest_rows()
        self.runner.experiment.mkdir(parents=True)
        self.manifest = self.runner.experiment / 'manifest.tsv'
        with self.manifest.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, launcher.COLUMNS, delimiter='\t')
            writer.writeheader()
            writer.writerows(self.jobs)

    def tree(self):
        return {
            str(path.relative_to(self.root)): (
                path.read_bytes() if path.is_file() else None
            )
            for path in self.root.rglob('*')
        }

    def test_registered_manifest_matches_repository_and_gpu_allocation(self):
        repository_manifest = (
            REPO_ROOT / 'experiments' / launcher.EXPERIMENT_NAME / 'manifest.tsv'
        )
        self.assertEqual(launcher.read_manifest(repository_manifest), self.jobs)
        self.assertEqual(launcher.read_manifest(self.manifest), self.jobs)
        self.assertEqual(Counter(job['gpu'] for job in self.jobs), {'0': 3, '1': 2, '2': 2})
        self.assertEqual({job['seed'] for job in self.jobs}, {'47'})
        self.assertEqual(
            [job['structure_variant'] for job in self.jobs],
            [item[2] for item in launcher.REGISTERED_VARIANTS],
        )

        original = self.manifest.read_text(encoding='utf-8')
        self.manifest.write_text(original.replace('\t47\t0\t', '\t47\t3\t', 1))
        with self.assertRaisesRegex(ValueError, 'registered seven-run design'):
            launcher.read_manifest(self.manifest)

    def test_repository_protocol_locks_official_validation_semantics(self):
        protocol_path = (
            REPO_ROOT / 'experiments' / launcher.EXPERIMENT_NAME
            / 'PROTOCOL.json'
        )
        payload = launcher.validate_protocol(protocol_path)
        self.assertEqual(
            payload['validation']['checkpoint_selection_metric'],
            'official_window_micro_pixel_iou_at_0.5',
        )
        changed = dict(payload)
        changed['validation'] = dict(payload['validation'])
        changed['validation']['eval_chunk_rows'] = 0
        local_protocol = self.runner.experiment / 'PROTOCOL.json'
        local_protocol.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'eval_chunk_rows'):
            launcher.validate_protocol(local_protocol)

    def test_dry_run_is_read_only_and_prints_correct_bestval_protocol(self):
        self.assertEqual(self.runner.environment['CSIG_ALLOWED_GPU_IDS'], '0,1,2')
        self.assertNotIn(
            'CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION', self.runner.environment
        )
        before = self.tree()
        stream = io.StringIO()
        with redirect_stdout(stream), \
             mock.patch.object(self.runner, 'validate_setup', return_value=self.jobs), \
             mock.patch.object(launcher.legacy.subprocess, 'Popen') as popen:
            self.assertEqual(self.runner.run(dry_run=True), 0)
        popen.assert_not_called()
        self.assertEqual(self.tree(), before)
        self.assertEqual(stream.getvalue().count('\nTRAIN '), 7)
        self.assertEqual(stream.getvalue().count('\nEVAL '), 7)

        for job in self.jobs:
            command = self.runner.train_command(job)
            expected = {
                '--gpu': job['gpu'],
                '--gpu_num': '1',
                '--seed': '47',
                '--epoch': '32',
                '--batch_size': '4',
                '--train_amp': '0',
                '--eval_amp': '0',
                '--eval_chunk_rows': '32',
                '--eval_interval': '1',
                '--skip_inprocess_validation': '0',
                '--validation_safe_cudnn': '1',
                '--validation_overlap_policy': 'official_window',
                '--early_stopping_metric': 'eval_iou',
                '--early_stopping_patience': '0',
                '--run_test_after_train': '0',
                '--resume': 'never',
                '--use_swanlab': '0',
                '--base_ckpt': '',
                '--spatial_ckpt': '',
                '--st_ckpt': '',
                '--upstream_compat': '1',
            }
            for flag, value in expected.items():
                self.assertEqual(option(command, flag), value)

            evaluation = self.runner.eval_command(job)
            self.assertNotIn('--epoch', evaluation)
            self.assertIn(
                str(self.runner.run_dir(job) / 'source_snapshot/test.py'), evaluation
            )
            self.assertEqual(option(evaluation, '--gpu'), job['gpu'])
            self.assertEqual(option(evaluation, '--eval_chunk_rows'), '32')
            self.assertEqual(
                option(evaluation, '--sequence_list'),
                str(self.runner.split('val_sequences.txt')),
            )
            self.assertEqual(
                option(evaluation, '--metrics_json'), str(self.runner.metrics_path(job))
            )

    def test_per_gpu_lock_serializes_only_jobs_sharing_a_gpu(self):
        self.runner.queue.mkdir(parents=True)
        active = defaultdict(int)
        maximum = defaultdict(int)
        total = {'active': 0, 'maximum': 0}
        state_lock = threading.Lock()

        def fake_base_run_job(_runner, job):
            with state_lock:
                active[job['gpu']] += 1
                maximum[job['gpu']] = max(maximum[job['gpu']], active[job['gpu']])
                total['active'] += 1
                total['maximum'] = max(total['maximum'], total['active'])
            time.sleep(0.04)
            with state_lock:
                active[job['gpu']] -= 1
                total['active'] -= 1
            return True

        with mock.patch.object(self.runner, 'wait_for_gpu_idle'), mock.patch.object(
            launcher.legacy.Launcher,
            'run_job',
            autospec=True,
            side_effect=fake_base_run_job,
        ):
            with ThreadPoolExecutor(max_workers=7) as pool:
                results = list(pool.map(self.runner.run_job, self.jobs))
        self.assertEqual(results, [True] * 7)
        self.assertEqual(dict(maximum), {'0': 1, '1': 1, '2': 1})
        self.assertGreaterEqual(total['maximum'], 2)
        self.assertEqual(
            {path.name for path in self.runner.queue.glob('.gpu-*.lock')},
            {'.gpu-0.lock', '.gpu-1.lock', '.gpu-2.lock'},
        )

    def test_busy_physical_gpu_waits_before_starting(self):
        readings = iter((10812, 25))
        with mock.patch.object(
            self.runner,
            'gpu_memory_used_mib',
            side_effect=lambda _gpu: next(readings),
        ), mock.patch.object(
            self.runner.cancel, 'wait', return_value=False
        ) as wait:
            self.runner.wait_for_gpu_idle('1')
        wait.assert_called_once_with(10.0)

    def create_valid_artifacts(self, job, best_epoch=19):
        run_dir = self.runner.run_dir(job)
        training_log = run_dir / 'logs' / (launcher.MODEL + '.txt')
        training_log.parent.mkdir(parents=True, exist_ok=True)
        training_log.write_text(
            ''.join(
                (
                    '2026-09-11 - INFO - ---- EPOCH %03d EVALUATION ----\n'
                    'Eval mean loss: 0.375000\n'
                    'Eval avg class IoU of prediction: 0.625000\n'
                    'Eval pixel precision: 0.700000\n'
                    'Eval pixel recall: 0.600000\n'
                    'Eval pixel F1: 0.646154\n'
                    'Best validation pixel IoU: 0.625000 at epoch %d\n'
                ) % (epoch, epoch)
                for epoch in range(1, launcher.EPOCHS + 1)
            ),
            encoding='utf-8',
        )
        best_path = (run_dir / 'checkpoints' / 'best_model.pth').resolve()
        best_path.parent.mkdir(exist_ok=True)
        torch.save(
            {
                'epoch': best_epoch - 1,
                'model_name': launcher.MODEL,
                'model_config': {'structure_variant': job['structure_variant']},
                'checkpoint_selection': {
                    'metric': 'eval_iou',
                    'mode': 'max',
                    'overlap_policy': 'official_window',
                    'best_value': 0.625,
                    'best_epoch': best_epoch,
                },
                'validation_metrics': {
                    'epoch': best_epoch,
                    'overlap_policy': 'official_window',
                    'loss': 0.375,
                    'iou': 0.625,
                    'precision': 0.7,
                    'recall': 0.6,
                    'f1': 0.646153846,
                },
            },
            best_path,
        )
        metrics_path = self.runner.metrics_path(job).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'dataset': launcher.DATASET,
            'model': launcher.MODEL,
            'inference_amp': False,
            'checkpoint': str(best_path),
            'checkpoint_epoch': best_epoch,
            'sequence_count': launcher.VAL_SEQUENCE_COUNT,
            'evaluation_windows': launcher.VAL_EVALUATION_WINDOWS,
            'all': {'pd': 0.8, 'fa': 0.00003, 'auc': 0.95},
        }
        metrics_path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
        (run_dir / 'eval.txt').write_text(
            'Paper-aligned metrics saved to %s.\n' % metrics_path,
            encoding='utf-8',
        )
        return training_log, best_path, metrics_path, payload

    def test_artifacts_require_all_epochs_and_best_checkpoint_metadata(self):
        job = self.jobs[0]
        training_log, best_path, _metrics_path, _payload = self.create_valid_artifacts(job)
        self.runner.validate_artifacts(job)

        training_log.write_text(
            training_log.read_text(encoding='utf-8').replace(
                '---- EPOCH 017 EVALUATION ----\n', ''
            ),
            encoding='utf-8',
        )
        with self.assertRaisesRegex(ValueError, 'epochs 1..32'):
            self.runner.validate_artifacts(job)

        # Restore the log, then prove selection metadata is independently checked.
        self.create_valid_artifacts(job)
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=True)
        checkpoint['checkpoint_selection']['metric'] = 'eval_f1'
        torch.save(checkpoint, best_path)
        with self.assertRaisesRegex(ValueError, 'official-window eval_iou'):
            self.runner.validate_artifacts(job)

    def test_artifacts_bind_detection_metrics_to_selected_best_epoch(self):
        job = self.jobs[1]
        _training_log, _best_path, metrics_path, payload = self.create_valid_artifacts(
            job, best_epoch=7
        )
        self.runner.validate_artifacts(job)

        payload['checkpoint_epoch'] = 32
        metrics_path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not match best_epoch'):
            self.runner.validate_artifacts(job)

        payload['checkpoint_epoch'] = 7
        payload['all'].pop('pd')
        metrics_path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'all.pd'):
            self.runner.validate_artifacts(job)


if __name__ == '__main__':
    unittest.main()
