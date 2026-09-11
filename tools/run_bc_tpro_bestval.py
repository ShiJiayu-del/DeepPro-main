#!/usr/bin/env python3
"""Run the registered seed-47 BC-TPro best-validation-checkpoint rerun."""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import sys


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_bc_tpro_nongate as legacy  # noqa: E402


EXPERIMENT_NAME = 'bc_tpro_stage1_noise8_bestval_seed47_2026-09-11'
SOURCE_EXPERIMENT_NAME = 'bc_tpro_stage1_noise8_upstream_2026-09-09'
DATASET = legacy.DATASET
MODEL = legacy.MODEL
COLUMNS = legacy.COLUMNS
EPOCHS = 32
VAL_SEQUENCE_COUNT = 16
VAL_EVALUATION_WINDOWS = 48
METRICS_SUFFIX = '__best_val_internal_val.json'
GPU_IDLE_MEMORY_MIB = 512

REGISTERED_VARIANTS = (
    ('bv_b1_none_seed47', '1', 'none', '0', 'B1'),
    ('bv_c0_temporal_control_seed47', '1', 'temporal_control', '1', 'C0'),
    ('bv_c1_center_multiscale_seed47', '1', 'center_multiscale', '2', 'C1'),
    ('bv_c2_center_ring_seed47', '2', 'center_ring', '0', 'C2'),
    ('bv_ng1_ring_difference_seed47', '2', 'center_ring_difference', '1', 'NG1'),
    ('bv_ng2_bandpass_seed47', '2', 'temporal_bandpass', '2', 'NG2'),
    ('bv_ng3_spatial_smooth_seed47', '3', 'center_spatial_smooth', '0', 'NG3'),
)


def expected_manifest_rows():
    rows = []
    for run_id, wave, variant, gpu, label in REGISTERED_VARIANTS:
        rows.append({
            'run_id': run_id,
            'wave': wave,
            'model': MODEL,
            'structure_variant': variant,
            'seed': '47',
            'gpu': gpu,
            'log_dir': (
                '2026-09-11/%s__Upstream8fa1a68-FP32-SoftIoU-'
                '%s%s_seed47_E32_ValEveryEpoch_BestIoU'
                % (DATASET, 'BCTPro-' if not label.startswith('NG') else '', label)
            ),
        })
    return rows


def read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != COLUMNS:
            raise ValueError('Unexpected best-validation manifest columns.')
        rows = list(reader)
    if rows != expected_manifest_rows():
        raise ValueError(
            'Best-validation manifest differs from the registered seven-run design.'
        )
    return rows


def validate_protocol(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = {
        'official_commit': '8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28',
        'training_dataset': DATASET,
        'seed': 47,
        'epochs': EPOCHS,
        'initialization': 'scratch_only',
        'precision': 'FP32',
        'learning_rate': 0.001,
        'loss': 'soft_iou',
        'swanlab': False,
        'allowed_physical_gpus': [0, 1, 2],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                'Best-validation protocol %s must be %r.' % (field, value)
            )
    validation = payload.get('validation')
    required_validation = {
        'interval_epochs': 1,
        'split': 'internal_val16',
        'safe_cudnn': True,
        'eval_chunk_rows': 32,
        'checkpoint_selection_metric': (
            'official_window_micro_pixel_iou_at_0.5'
        ),
        'overlap_policy': 'repeat_overlap_frames_per_official_train.py',
        'checkpoint_selection_direction': 'maximize',
        'exact_tie_policy': 'later_epoch',
        'selected_checkpoint': 'best_model.pth',
    }
    if not isinstance(validation, dict):
        raise ValueError('Best-validation protocol validation block is missing.')
    for field, value in required_validation.items():
        if validation.get(field) != value:
            raise ValueError(
                'Best-validation protocol validation.%s must be %r.'
                % (field, value)
            )
    post_training = payload.get('post_training_evaluation')
    if not isinstance(post_training, dict) or post_training.get(
        'checkpoint'
    ) != 'best_model.pth':
        raise ValueError(
            'Post-training evaluation must use best_model.pth.'
        )
    return payload


def _set_option(command, flag, value):
    """Replace one required option in an argv list."""
    try:
        index = command.index(flag)
    except ValueError as error:
        raise ValueError('Inherited launcher is missing required option %s.' % flag) from error
    command[index + 1] = str(value)


def _required_int(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('%s must be an integer.' % context)
    return value


def _required_finite(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('%s must be numeric.' % context)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('%s must be finite.' % context)
    return result


class Launcher(legacy.Launcher):
    """Reuse the hardened run-once pipeline with a corrected validation policy."""

    def __init__(self, repo_root=REPO_ROOT, python_bin=None):
        super().__init__(repo_root=repo_root, python_bin=python_bin)
        self.experiment = self.repo / 'experiments' / EXPERIMENT_NAME
        self.old_experiment = self.repo / 'experiments' / SOURCE_EXPERIMENT_NAME
        self.queue = self.save / 'sem_seg' / '_queues' / EXPERIMENT_NAME
        self.status = self.queue / 'status'
        self.max_workers = len(REGISTERED_VARIANTS)
        # This is a new protocol, not one of the frozen external-only runs.
        self.environment.pop('CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION', None)

    def metrics_path(self, job):
        return self.experiment / 'metrics' / (job['run_id'] + METRICS_SUFFIX)

    def validate_setup(self):
        command = [
            self.python,
            str(self.repo / 'tools/validate_bc_tpro_noise8_setup.py'),
            '--data-root', str(self.data),
            '--expected-data-root', str(self.data),
            '--dataset', DATASET,
            '--manifest', str(self.old_experiment / 'manifest.tsv'),
            '--train-list', str(self.split('train_sequences.txt')),
            '--val-list', str(self.split('val_sequences.txt')),
            '--split-manifest', str(self.split('split_manifest.json')),
            '--profile', 'upstream8fa1a68_fp32',
            '--protocol-config', str(self.old_experiment / 'UPSTREAM_PROTOCOL.json'),
        ]
        subprocess.run(command, cwd=self.repo, env=self.environment, check=True)
        validate_protocol(self.experiment / 'PROTOCOL.json')
        return read_manifest(self.experiment / 'manifest.tsv')

    def train_command(self, job):
        command = super().train_command(job)
        for flag, value in (
            ('--epoch', EPOCHS),
            ('--train_amp', 0),
            ('--eval_amp', 0),
            ('--eval_chunk_rows', 32),
            ('--eval_interval', 1),
            ('--skip_inprocess_validation', 0),
            ('--early_stopping_patience', 0),
            ('--early_stopping_metric', 'eval_iou'),
            ('--run_test_after_train', 0),
            ('--seed', 47),
            ('--resume', 'never'),
            ('--use_swanlab', 0),
            ('--swanlab_group', 'bc-tpro-stage1-noise8-bestval-seed47'),
        ):
            _set_option(command, flag, value)
        if '--validation_safe_cudnn' in command:
            _set_option(command, '--validation_safe_cudnn', 1)
        else:
            command.extend(['--validation_safe_cudnn', '1'])
        if '--validation_overlap_policy' in command:
            _set_option(
                command, '--validation_overlap_policy', 'official_window'
            )
        else:
            command.extend([
                '--validation_overlap_policy', 'official_window'
            ])
        return command

    def eval_command(self, job):
        command = super().eval_command(job)
        epoch_index = command.index('--epoch')
        del command[epoch_index:epoch_index + 2]
        _set_option(command, '--eval_chunk_rows', 32)
        return command

    def gpu_memory_used_mib(self, gpu):
        result = subprocess.run(
            [
                'nvidia-smi', '--id=' + str(gpu),
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(values) != 1 or not values[0].isdigit():
            raise ValueError(
                'Cannot read physical GPU %s memory usage: %r'
                % (gpu, result.stdout)
            )
        return int(values[0])

    def wait_for_gpu_idle(self, gpu):
        announced = False
        while True:
            used_mib = self.gpu_memory_used_mib(gpu)
            if used_mib <= GPU_IDLE_MEMORY_MIB:
                if announced:
                    print('GPU %s is now idle; starting queued job.' % gpu, flush=True)
                return
            if not announced:
                print(
                    'WAIT GPU %s is occupied (%d MiB used); keeping this job queued.'
                    % (gpu, used_mib),
                    flush=True,
                )
                announced = True
            if self.cancel.wait(10.0):
                raise RuntimeError('Launcher interrupted while waiting for idle GPU.')

    def run_job(self, job):
        """Hold a per-physical-GPU lock for training plus final evaluation."""
        gpu_lock_path = self.queue / ('.gpu-%s.lock' % job['gpu'])
        with gpu_lock_path.open('a') as gpu_lock:
            while True:
                if self.cancel.is_set():
                    raise RuntimeError('Launcher interrupted while waiting for GPU lock.')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.cancel.wait(0.5)
            self.wait_for_gpu_idle(job['gpu'])
            return super().run_job(job)

    def validate_artifacts(self, job):
        run_dir = self.run_dir(job)
        training_log = run_dir / 'logs' / (MODEL + '.txt')
        if not training_log.is_file():
            raise FileNotFoundError('Missing training log: %s' % training_log)
        training_text = training_log.read_text(encoding='utf-8')
        epochs = [
            int(value) for value in re.findall(
                r'---- EPOCH ([0-9]{3}) EVALUATION ----',
                training_text,
            )
        ]
        if epochs != list(range(1, EPOCHS + 1)):
            raise ValueError(
                '%s: expected exactly one complete validation for epochs 1..32; got %s.'
                % (job['run_id'], epochs)
            )
        completed_fields = (
            'Eval mean loss:',
            'Eval avg class IoU of prediction:',
            'Eval pixel precision:',
            'Eval pixel recall:',
            'Eval pixel F1:',
            'Best validation pixel IoU:',
        )
        incomplete = {
            field: training_text.count(field)
            for field in completed_fields
            if training_text.count(field) != EPOCHS
        }
        if incomplete:
            raise ValueError(
                '%s: 32 complete validation metric records are required; got %s.'
                % (job['run_id'], incomplete)
            )

        best_path = (run_dir / 'checkpoints' / 'best_model.pth').resolve()
        if not best_path.is_file():
            raise FileNotFoundError('Missing best-validation checkpoint: %s' % best_path)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from runtime_utils import load_checkpoint

        checkpoint = load_checkpoint(best_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise ValueError('%s: best checkpoint is not a mapping.' % job['run_id'])
        if checkpoint.get('model_name') != MODEL:
            raise ValueError('%s: checkpoint model identity mismatch.' % job['run_id'])
        model_config = checkpoint.get('model_config')
        if not isinstance(model_config, dict):
            raise ValueError('%s: checkpoint model_config is missing.' % job['run_id'])
        if model_config.get('structure_variant') != job['structure_variant']:
            raise ValueError('%s: checkpoint structure variant mismatch.' % job['run_id'])

        selection = checkpoint.get('checkpoint_selection')
        if not isinstance(selection, dict):
            raise ValueError('%s: checkpoint_selection is missing.' % job['run_id'])
        if (
            selection.get('metric') != 'eval_iou'
            or selection.get('mode') != 'max'
            or selection.get('overlap_policy') != 'official_window'
        ):
            raise ValueError(
                '%s: checkpoint selection must maximize official-window '
                'eval_iou.' % job['run_id']
            )
        best_epoch = _required_int(
            selection.get('best_epoch'), '%s checkpoint best_epoch' % job['run_id']
        )
        if best_epoch not in range(1, EPOCHS + 1):
            raise ValueError('%s: best_epoch is outside 1..32.' % job['run_id'])
        if _required_int(
            checkpoint.get('epoch'), '%s checkpoint epoch' % job['run_id']
        ) + 1 != best_epoch:
            raise ValueError('%s: best checkpoint epoch does not match best_epoch.' % job['run_id'])

        validation = checkpoint.get('validation_metrics')
        if not isinstance(validation, dict):
            raise ValueError('%s: validation_metrics is missing.' % job['run_id'])
        if _required_int(
            validation.get('epoch'), '%s validation_metrics.epoch' % job['run_id']
        ) != best_epoch:
            raise ValueError('%s: validation_metrics epoch does not match best_epoch.' % job['run_id'])
        if validation.get('overlap_policy') != 'official_window':
            raise ValueError(
                '%s: validation metrics do not use official window overlap.'
                % job['run_id']
            )
        for field in ('loss', 'iou', 'precision', 'recall', 'f1'):
            _required_finite(
                validation.get(field), '%s validation_metrics.%s' % (job['run_id'], field)
            )
        best_value = _required_finite(
            selection.get('best_value'), '%s checkpoint best_value' % job['run_id']
        )
        if not math.isclose(
            best_value, float(validation['iou']), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError('%s: best_value does not match validation IoU.' % job['run_id'])

        metrics_path = self.metrics_path(job).resolve()
        if not metrics_path.is_file():
            raise FileNotFoundError('Missing best-checkpoint metrics: %s' % metrics_path)
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('%s: metrics payload is not a mapping.' % job['run_id'])
        try:
            payload_checkpoint = Path(payload['checkpoint']).expanduser().resolve()
        except (KeyError, TypeError) as error:
            raise ValueError('%s: metrics checkpoint path is missing.' % job['run_id']) from error
        if payload_checkpoint != best_path:
            raise ValueError('%s: metrics were not produced from best_model.pth.' % job['run_id'])
        if _required_int(
            payload.get('checkpoint_epoch'), '%s metrics checkpoint_epoch' % job['run_id']
        ) != best_epoch:
            raise ValueError('%s: metrics checkpoint_epoch does not match best_epoch.' % job['run_id'])
        expected_payload = {
            'dataset': DATASET,
            'model': MODEL,
            'inference_amp': False,
            'sequence_count': VAL_SEQUENCE_COUNT,
            'evaluation_windows': VAL_EVALUATION_WINDOWS,
        }
        for field, expected in expected_payload.items():
            if payload.get(field) != expected:
                raise ValueError(
                    '%s: metrics %s must be %r.' % (job['run_id'], field, expected)
                )
        paper_metrics = payload.get('all')
        if not isinstance(paper_metrics, dict):
            raise ValueError('%s: Pd/Fa/AUC payload is missing.' % job['run_id'])
        for field in ('pd', 'fa', 'auc'):
            value = _required_finite(
                paper_metrics.get(field), '%s metrics all.%s' % (job['run_id'], field)
            )
            if not 0.0 <= value <= 1.0:
                raise ValueError('%s: metrics all.%s is outside [0, 1].' % (job['run_id'], field))

        eval_log = run_dir / 'eval.txt'
        if not eval_log.is_file():
            raise FileNotFoundError('Missing best-checkpoint evaluation log: %s' % eval_log)
        completion = 'Paper-aligned metrics saved to %s.' % metrics_path
        if eval_log.read_text(encoding='utf-8').count(completion) != 1:
            raise ValueError('%s: evaluation completion marker is missing or duplicated.' % job['run_id'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='Validate and print without creating files.')
    mode.add_argument('--run', action='store_true', help='Launch the seven registered single-GPU jobs.')
    args = parser.parse_args(argv)
    try:
        return Launcher().run(dry_run=args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('REFUSED: %s' % error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
