#!/usr/bin/env python3
"""Run the registered single-seed, three-GPU no-gate ablation exactly once."""

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = 'bc_tpro_nongate_noise8_seed47_2026-09-10'
OLD_EXPERIMENT_NAME = 'bc_tpro_stage1_noise8_upstream_2026-09-09'
DATASET = 'NUDT-MIRSDT-Noise8.0_FJY'
MODEL = 'DeepPro-Plus_BCTPro'
COLUMNS = ['run_id', 'wave', 'model', 'structure_variant', 'seed', 'gpu', 'log_dir']
SNAPSHOT_FILES = (
    'train.py', 'test.py', 'ShootingRules.py', 'write_results.py',
    'data_utils/TrainDataLoader.py', 'data_utils/TestDataLoader.py',
    'data_utils/loader_utils.py', 'networks/layers/basic.py',
    'networks/layers/TPro.py', 'networks/layers/bc_tpro_adapter.py',
    'networks/losses/segmentation_losses.py',
    'networks/models/DeepPro-Plus_BCTPro.py',
)
EXTRA_SNAPSHOT_FILES = (
    'runtime_utils.py', 'sequence_utils.py', 'networks/losses/__init__.py',
    'tools_forSatVideoIRSTD/seg2centroid_txt.py',
)
PACKAGES = ('data_utils', 'networks', 'networks/models', 'networks/layers',
            'networks/losses', 'tools_forSatVideoIRSTD')


def expected_manifest_rows():
    rows = []
    for index, (prefix, variant) in enumerate((
        ('ng1_ring_difference', 'center_ring_difference'),
        ('ng2_bandpass', 'temporal_bandpass'),
        ('ng3_spatial_smooth', 'center_spatial_smooth'),
    ), 1):
        rows.append(dict(
            run_id=prefix + '_seed47', wave='1', model=MODEL,
            structure_variant=variant, seed='47', gpu=str(index - 1),
            log_dir=('2026-09-10/%s__Upstream8fa1a68-FP32-SoftIoU-NG%d_seed47_E32'
                     % (DATASET, index)),
        ))
    return rows


def read_manifest(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != COLUMNS:
            raise ValueError('Unexpected no-gate manifest columns.')
        rows = list(reader)
    if rows != expected_manifest_rows():
        raise ValueError('No-gate manifest differs from the registered three-run design.')
    return rows


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def write_marker(path, values):
    # Exclusive creation preserves any previous evidence, including empty files.
    with path.open('x', encoding='utf-8') as handle:
        for key, value in values.items():
            handle.write('%s=%s\n' % (key, str(value).replace('\n', ' ')))


class Launcher:
    def __init__(self, repo_root=REPO_ROOT, python_bin=None):
        self.repo = Path(repo_root).resolve()
        self.python = python_bin or os.environ.get(
            'PYTHON_BIN', '/home/user/anaconda3/envs/sjyPID/bin/python')
        self.experiment = self.repo / 'experiments' / EXPERIMENT_NAME
        self.old_experiment = self.repo / 'experiments' / OLD_EXPERIMENT_NAME
        self.data = self.repo.parent / 'datasets' / DATASET
        self.save = self.repo / 'log'
        self.queue = self.save / 'sem_seg' / '_queues' / EXPERIMENT_NAME
        self.status = self.queue / 'status'
        self.cancel = threading.Event()
        self.signal_number = None
        self.eval_lock = threading.Lock()
        self.environment = os.environ.copy()
        self.environment['PYTHONDONTWRITEBYTECODE'] = '1'
        self.environment.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.environment['CSIG_ALLOWED_GPU_IDS'] = '0,1,2'

    def run_dir(self, job):
        return self.save / 'sem_seg' / job['log_dir']

    def metrics_path(self, job):
        return self.experiment / 'metrics' / (job['run_id'] + '__noise8_internal_val.json')

    def split(self, name):
        return self.old_experiment / 'splits' / name

    def validate_setup(self):
        command = [self.python, str(self.repo / 'tools/validate_bc_tpro_noise8_setup.py'),
                   '--data-root', str(self.data), '--expected-data-root', str(self.data),
                   '--dataset', DATASET, '--manifest', str(self.old_experiment / 'manifest.tsv'),
                   '--train-list', str(self.split('train_sequences.txt')),
                   '--val-list', str(self.split('val_sequences.txt')),
                   '--split-manifest', str(self.split('split_manifest.json')),
                   '--profile', 'upstream8fa1a68_fp32', '--protocol-config',
                   str(self.old_experiment / 'UPSTREAM_PROTOCOL.json')]
        subprocess.run(command, cwd=self.repo, env=self.environment, check=True)
        return read_manifest(self.experiment / 'manifest.tsv')

    def train_command(self, job):
        values = [
            '--model', MODEL, '--structure_variant', job['structure_variant'],
            '--batch_size', '4', '--gradient_accumulation_steps', '1', '--epoch', '32',
            '--learning_rate', '0.001', '--optimizer', 'Adam', '--decay_rate', '0.0001',
            '--step_size', '10', '--lr_decay', '0.7', '--gpu', job['gpu'], '--gpu_num', '1',
            '--datapath', str(self.data), '--dataset', DATASET,
            '--train_sequence_list', str(self.split('train_sequences.txt')),
            '--val_sequence_list', str(self.split('val_sequences.txt')),
            '--log_dir', job['log_dir'], '--savepath', str(self.save), '--seqlen', '40',
            '--patch_size', '128', '--sample_rate', '0.1', '--sequence_augmentation', '0',
            '--loss', 'soft_iou', '--threshold_eval', '0.5', '--train_amp', '0',
            '--eval_amp', '0', '--eval_chunk_rows', '32', '--eval_interval', '8',
            '--skip_inprocess_validation', '1', '--early_stopping_patience', '0',
            '--early_stopping_metric', 'eval_iou', '--train_workers', '4',
            '--val_workers', '1', '--prefetch_factor', '2', '--seed', job['seed'],
            '--deterministic', '1', '--resume', 'never', '--run_test_after_train', '0',
            '--use_swanlab', '1', '--swanlab_project', 'DeepPro-BC-TPro',
            '--swanlab_group', 'bc-tpro-stage1-noise8-upstream8fa1a68-fp32-scratch',
            '--swanlab_mode', 'cloud', '--swanlab_resume', 'never',
            '--base_ckpt', '', '--spatial_ckpt', '', '--st_ckpt', '',
            '--freeze_pretrained', '0', '--upstream_compat', '1',
        ]
        return [self.python, '-u', str(self.repo / 'train.py')] + values

    def eval_command(self, job):
        return [self.python, '-u', str(self.run_dir(job) / 'source_snapshot/test.py'),
                '--epoch', '32', '--gpu', job['gpu'], '--seqlen', '40',
                '--datapath', str(self.data), '--dataset', DATASET,
                '--sequence_list', str(self.split('val_sequences.txt')),
                '--logpath', str(self.save), '--log_dir', job['log_dir'],
                '--test_workers', '1', '--prefetch_factor', '1', '--eval_chunk_rows', '32',
                '--threshold_eval', '0.5', '--threshold_grid_step', '0.01',
                '--metrics_json', str(self.metrics_path(job))]

    def validate_artifacts(self, job):
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from tools import analyze_bc_tpro_noise8_stage1 as analysis
        payload = json.loads(self.metrics_path(job).read_text(encoding='utf-8'))
        names = self.split('val_sequences.txt').read_text(encoding='utf-8').splitlines()
        analysis.validate_one_payload(
            payload, job, self.metrics_path(job), self.experiment, self.save / 'sem_seg',
            self.data, self.split('train_sequences.txt'), self.split('val_sequences.txt'),
            names, {}, {}, analysis.UPSTREAM_PROFILE,
        )

    def check_job(self, job):
        for suffix in ('running', 'failed'):
            marker = self.status / (job['run_id'] + '.' + suffix)
            if marker.exists():
                raise ValueError('Refusing automatic retry; inspect existing marker: %s' % marker)
        done = self.status / (job['run_id'] + '.done')
        if done.exists():
            values = dict(line.split('=', 1) for line in done.read_text().splitlines())
            for field in ('run_id', 'wave', 'gpu', 'seed'):
                if values.get(field) != job[field]:
                    raise ValueError('Completed marker identity mismatch: %s' % done)
            if values.get('exit_code') != '0':
                raise ValueError('Completed marker lacks successful exit code: %s' % done)
            self.validate_artifacts(job)
            return 'done'
        directory = self.run_dir(job)
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError('Refusing non-empty experiment directory: %s' % directory)
        for path in (self.metrics_path(job), self.queue / 'launcher_logs' / (job['run_id'] + '.log')):
            if path.exists():
                raise ValueError('Refusing existing run evidence: %s' % path)
        return 'ready'

    def complete_snapshot(self, job):
        snapshot = self.run_dir(job) / 'source_snapshot'
        for relative in SNAPSHOT_FILES:
            copied = snapshot / relative
            source = self.repo / relative
            if not copied.is_file() or copied.read_bytes() != source.read_bytes():
                raise ValueError('Missing or changed training snapshot: %s' % copied)
        for relative in EXTRA_SNAPSHOT_FILES:
            target = snapshot / relative
            content = (self.repo / relative).read_bytes()
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Snapshot supplement differs: %s' % target)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    handle.write(content)
        for package in PACKAGES:
            target = snapshot / package / '__init__.py'
            target.parent.mkdir(parents=True, exist_ok=True)
            source = self.repo / package / '__init__.py'
            content = source.read_bytes() if source.is_file() else b''
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Snapshot package differs: %s' % target)
            if not target.exists():
                with target.open('xb') as handle:
                    handle.write(content)
        return snapshot

    def execute(self, command, log, cwd, environment):
        if self.cancel.is_set():
            raise RuntimeError('Launcher interrupted before child startup.')
        print('COMMAND ' + shlex.join(command), file=log, flush=True)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=cwd, env=environment, start_new_session=True)
        print('CHILD pid=%d pgid=%d' % (process.pid, process.pid), file=log, flush=True)
        try:
            while True:
                if self.cancel.is_set():
                    raise RuntimeError('Launcher interrupted; terminating owned process group.')
                try:
                    code = process.wait(timeout=0.5)
                    if code:
                        raise RuntimeError('Child exit code %d.' % code)
                    return
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The parent may already have exited while its workers remain alive.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise

    def run_job(self, job):
        lock_path = self.status / (job['run_id'] + '.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # These checks precede opening any writable run log.
            if self.check_job(job) == 'done':
                print('SKIP completed ' + job['run_id'], flush=True)
                return True
            started = time.monotonic()
            values = {field: job[field] for field in ('run_id', 'wave', 'gpu', 'seed')}
            values['started_at'] = timestamp()
            running = self.status / (job['run_id'] + '.running')
            write_marker(running, values)
            try:
                log_path = self.queue / 'launcher_logs' / (job['run_id'] + '.log')
                with log_path.open('x', encoding='utf-8', buffering=1) as log:
                    self.execute(self.train_command(job), log, self.repo, self.environment)
                    values['train_seconds'] = round(time.monotonic() - started, 3)
                    snapshot = self.complete_snapshot(job)
                    environment = dict(self.environment, PYTHONPATH=str(snapshot))
                    with self.eval_lock:
                        with (self.queue / '.evaluation.lock').open('a') as evaluation_lock:
                            while True:
                                if self.cancel.is_set():
                                    raise RuntimeError('Interrupted while waiting for evaluation.')
                                try:
                                    fcntl.flock(evaluation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    break
                                except BlockingIOError:
                                    self.cancel.wait(0.5)
                            values['evaluation_wait_seconds'] = round(
                                time.monotonic() - started - values['train_seconds'], 3)
                            evaluation_started = time.monotonic()
                            self.execute(self.eval_command(job), log, snapshot, environment)
                            values['evaluation_seconds'] = round(time.monotonic() - evaluation_started, 3)
                    self.validate_artifacts(job)
                values.update(finished_at=timestamp(), exit_code=0,
                              elapsed_seconds=round(time.monotonic() - started, 3))
                write_marker(self.status / (job['run_id'] + '.done'), values)
                running.unlink()
                print('DONE ' + job['run_id'], flush=True)
                return True
            except Exception as error:
                values.update(failed_at=timestamp(), exit_code=1, error=str(error),
                              elapsed_seconds=round(time.monotonic() - started, 3))
                write_marker(self.status / (job['run_id'] + '.failed'), values)
                running.unlink()
                print('FAILED %s: %s' % (job['run_id'], error), flush=True)
                return False

    def run(self, dry_run=False):
        jobs = self.validate_setup()
        if dry_run:
            for job in jobs:
                state = self.check_job(job)
                print('RUN id=%s gpu=%s seed=%s status=%s' % (
                    job['run_id'], job['gpu'], job['seed'], state))
                print('TRAIN ' + shlex.join(self.train_command(job)))
                print('EVAL ' + shlex.join(self.eval_command(job)))
            return 0
        for marker in ('pipeline.running', 'pipeline.failed'):
            if (self.queue / marker).exists():
                raise ValueError('Refusing existing pipeline state: %s' % (self.queue / marker))
        states = [self.check_job(job) for job in jobs]
        if (self.queue / 'pipeline.done').exists():
            if states != ['done'] * len(jobs):
                raise ValueError('Pipeline completion conflicts with run artifacts.')
            print('SKIP verified completed pipeline', flush=True)
            return 0
        self.status.mkdir(parents=True, exist_ok=True)
        (self.queue / 'launcher_logs').mkdir(exist_ok=True)
        (self.experiment / 'metrics').mkdir(exist_ok=True)
        with (self.queue / '.launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Recheck under the cross-process queue lock before mutation.
            for marker in ('pipeline.running', 'pipeline.failed', 'pipeline.done'):
                if (self.queue / marker).exists():
                    raise ValueError('Pipeline state changed during startup: ' + marker)
            for job in jobs:
                self.check_job(job)
            pipeline = dict(started_at=timestamp(), pid=os.getpid(), run_count=len(jobs))
            write_marker(self.queue / 'pipeline.running', pipeline)
            previous = {}

            def interrupted(number, _frame):
                self.signal_number = number
                self.cancel.set()

            try:
                for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    previous[number] = signal.signal(number, interrupted)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(self.run_job, job) for job in jobs]
                    results = []
                    for future in as_completed(futures):
                        try:
                            results.append(future.result())
                        except Exception as error:
                            # Cancel while the pool is alive, before __exit__ waits.
                            self.cancel.set()
                            results.append(False)
                            pipeline['error'] = str(error)
                success = all(results) and not self.cancel.is_set()
            except Exception as error:
                self.cancel.set()
                success = False
                pipeline['error'] = str(error)
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)
            pipeline.update(finished_at=timestamp(), exit_code=0 if success else 1)
            if self.signal_number is not None:
                pipeline['signal'] = self.signal_number
            write_marker(self.queue / ('pipeline.done' if success else 'pipeline.failed'), pipeline)
            (self.queue / 'pipeline.running').unlink()
            return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='Validate and print without creating files.')
    mode.add_argument('--run', action='store_true', help='Launch the three registered single-GPU jobs.')
    args = parser.parse_args(argv)
    try:
        return Launcher().run(dry_run=args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('REFUSED: %s' % error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
