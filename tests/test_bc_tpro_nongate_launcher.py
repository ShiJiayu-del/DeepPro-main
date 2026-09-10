"""CPU-only launcher checks: isolation, no overwrite, completion and cancellation."""

import csv
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'nongate_launcher_test_subject', REPO_ROOT / 'tools/run_bc_tpro_nongate.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'repo'
        self.root.mkdir()
        self.runner = launcher.Launcher(self.root, sys.executable)
        self.jobs = launcher.expected_manifest_rows()
        self.runner.experiment.mkdir(parents=True)
        self.manifest = self.runner.experiment / 'manifest.tsv'
        with self.manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, launcher.COLUMNS, delimiter='\t')
            writer.writeheader()
            writer.writerows(self.jobs)
        self.setup_patch = mock.patch.object(self.runner, 'validate_setup', return_value=self.jobs)
        self.setup_patch.start()
        self.addCleanup(self.setup_patch.stop)

    def tree(self):
        return {str(path.relative_to(self.root)): path.read_bytes() if path.is_file() else None
                for path in self.root.rglob('*')}

    def test_dry_run_has_no_writes_and_uses_fixed_protocol(self):
        before = self.tree()
        stream = io.StringIO()
        with redirect_stdout(stream), mock.patch.object(launcher.subprocess, 'Popen') as popen:
            self.assertEqual(self.runner.run(dry_run=True), 0)
        popen.assert_not_called()
        self.assertEqual(self.tree(), before)
        self.assertEqual(stream.getvalue().count('\nTRAIN '), 3)
        for job in self.jobs:
            command = self.runner.train_command(job)
            for flag, expected in {'--gpu': job['gpu'], '--gpu_num': '1', '--seed': '47',
                                   '--train_amp': '0', '--eval_amp': '0', '--upstream_compat': '1',
                                   '--epoch': '32', '--batch_size': '4', '--resume': 'never',
                                   '--base_ckpt': '', '--st_ckpt': '', '--spatial_ckpt': ''}.items():
                self.assertEqual(command[command.index(flag) + 1], expected)
            evaluation = self.runner.eval_command(job)
            self.assertIn(str(self.runner.run_dir(job) / 'source_snapshot/test.py'), evaluation)
            self.assertNotIn('--amp', evaluation)
            self.assertEqual(evaluation[evaluation.index('--threshold_grid_step') + 1], '0.01')
            self.assertEqual(evaluation[evaluation.index('--sequence_list') + 1],
                             str(self.runner.split('val_sequences.txt')))

    def test_manifest_rejects_gpu3_or_changed_identity(self):
        self.assertEqual(launcher.read_manifest(self.manifest), self.jobs)
        original = self.manifest.read_text()
        self.manifest.write_text(original.replace('\t47\t0\t', '\t47\t3\t'))
        with self.assertRaisesRegex(ValueError, 'registered three-run design'):
            launcher.read_manifest(self.manifest)

    def test_dirty_and_failed_runs_refused_without_log_truncation(self):
        for evidence in ('running', 'failed', 'log', 'directory', 'metrics'):
            with self.subTest(evidence=evidence):
                job = self.jobs[0]
                if evidence in ('running', 'failed'):
                    path = self.runner.status / (job['run_id'] + '.' + evidence)
                elif evidence == 'log':
                    path = self.runner.queue / 'launcher_logs' / (job['run_id'] + '.log')
                elif evidence == 'directory':
                    path = self.runner.run_dir(job) / 'preserved.txt'
                else:
                    path = self.runner.metrics_path(job)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original evidence\n')
                before = self.tree()
                with mock.patch.object(launcher.subprocess, 'Popen') as popen:
                    with self.assertRaisesRegex(ValueError, 'Refusing'):
                        self.runner.run()
                popen.assert_not_called()
                self.assertEqual(self.tree(), before)
                path.unlink()

    def test_parallel_training_serial_evaluation_and_verified_done(self):
        barrier = threading.Barrier(3, timeout=5)
        active = {'evaluations': 0, 'maximum': 0}
        counter_lock = threading.Lock()
        calls = []

        def fake_execute(command, log, cwd, environment):
            with counter_lock:
                calls.append(command)
            if command[2] == str(self.root / 'train.py'):
                barrier.wait()
                return
            self.assertEqual(cwd.name, 'source_snapshot')
            self.assertEqual(environment['PYTHONPATH'], str(cwd))
            with counter_lock:
                active['evaluations'] += 1
                active['maximum'] = max(active['maximum'], active['evaluations'])
            time.sleep(0.02)
            with counter_lock:
                active['evaluations'] -= 1

        with mock.patch.object(self.runner, 'execute', side_effect=fake_execute), \
             mock.patch.object(self.runner, 'complete_snapshot',
                               side_effect=lambda job: self.runner.run_dir(job) / 'source_snapshot'), \
             mock.patch.object(self.runner, 'validate_artifacts') as validate:
            self.assertEqual(self.runner.run(), 0)
            self.assertEqual(validate.call_count, 3)
            self.assertEqual(active['maximum'], 1)
            self.assertEqual(len(calls), 6)
            self.assertTrue((self.runner.queue / 'pipeline.done').is_file())
            self.assertFalse((self.runner.queue / 'pipeline.running').exists())
            for job in self.jobs:
                marker = self.runner.status / (job['run_id'] + '.done')
                values = dict(line.split('=', 1) for line in marker.read_text().splitlines())
                self.assertEqual(values['exit_code'], '0')
                self.assertIn('evaluation_wait_seconds', values)
                self.assertIn('evaluation_seconds', values)
            before = self.tree()
            self.assertEqual(self.runner.run(), 0)
            self.assertEqual(self.tree(), before)

    def test_zero_exit_with_invalid_artifacts_marks_failed(self):
        with mock.patch.object(self.runner, 'execute'), \
             mock.patch.object(self.runner, 'complete_snapshot',
                               side_effect=lambda job: self.runner.run_dir(job) / 'source_snapshot'), \
             mock.patch.object(self.runner, 'validate_artifacts',
                               side_effect=ValueError('epoch32 checkpoint missing')):
            self.assertEqual(self.runner.run(), 1)
        self.assertTrue((self.runner.queue / 'pipeline.failed').is_file())
        for job in self.jobs:
            self.assertTrue((self.runner.status / (job['run_id'] + '.failed')).is_file())
            self.assertFalse((self.runner.status / (job['run_id'] + '.done')).exists())
        with self.assertRaisesRegex(ValueError, 'existing pipeline state'):
            self.runner.run()

    def test_snapshot_closure_imports_outside_repository(self):
        job = self.jobs[0]
        snapshot = self.runner.run_dir(job) / 'source_snapshot'
        for relative in launcher.SNAPSHOT_FILES + launcher.EXTRA_SNAPSHOT_FILES:
            source = REPO_ROOT / relative
            local_source = self.root / relative
            local_source.parent.mkdir(parents=True, exist_ok=True)
            local_source.write_bytes(source.read_bytes())
            if relative in launcher.SNAPSHOT_FILES:
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        self.runner.complete_snapshot(job)
        script = (
            'import importlib, pathlib; import test, train; '
            'root = pathlib.Path.cwd(); '
            'names = ["test", "train", "runtime_utils", "sequence_utils", '
            '"networks.losses", "data_utils.TestDataLoader", '
            '"tools_forSatVideoIRSTD.seg2centroid_txt", '
            '"networks.models.DeepPro-Plus_BCTPro", "networks.layers.bc_tpro_adapter", '
            '"networks.losses.segmentation_losses"]; '
            'modules = [importlib.import_module(name) for name in names]; '
            'assert all(root in pathlib.Path(module.__file__).parents for module in modules)'
        )
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=snapshot,
                                env=dict(os.environ, PYTHONPATH=str(snapshot)),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (snapshot / 'test.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed training snapshot'):
            self.runner.complete_snapshot(job)

    def test_cancellation_terminates_only_owned_process_group(self):
        command = [sys.executable, '-c', 'import time; time.sleep(30)']
        timer = threading.Timer(0.2, self.runner.cancel.set)
        started = time.monotonic()
        with tempfile.TemporaryFile(mode='w+') as log:
            timer.start()
            try:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    self.runner.execute(command, log, self.root, os.environ.copy())
            finally:
                timer.cancel()
            log.seek(0)
            text = log.read()
        self.assertLess(time.monotonic() - started, 7)
        pid = int(text.split('CHILD pid=')[1].split()[0])
        with self.assertRaises(ProcessLookupError):
            os.killpg(pid, 0)


if __name__ == '__main__':
    unittest.main()
