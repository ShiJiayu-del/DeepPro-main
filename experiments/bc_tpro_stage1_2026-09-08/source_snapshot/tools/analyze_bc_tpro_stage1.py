#!/usr/bin/env python3
"""Aggregate preregistered BC-TPro stage-1 paper metrics."""

import argparse
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


VARIANT_LABELS = {
    'none': 'B1 DeepPro-Plus',
    'temporal_control': 'C0 temporal control',
    'center_multiscale': 'C1 center multiscale',
    'center_ring': 'C2 center+ring',
}
EXPECTED_MODEL = 'DeepPro-Plus_BCTPro'
CONDITIONS = ('clean_val', 'noise8_val')
CONDITION_DATASETS = {
    'clean_val': 'NUDT-MIRSDT',
    'noise8_val': 'NUDT-MIRSDT-Noise8.0_FJY',
}
EXPECTED_SEEDS = (47, 49, 51)
EXPECTED_SEQUENCE_COUNT = 16
EXPECTED_CHECKPOINT_EPOCH = 32
EXPECTED_SEQUENCE_LENGTH = 40
EXPECTED_EVALUATION_WINDOWS = 48
EXPECTED_SOURCE_SHA256 = (
    '31e71368775adbc6c893bcb29ae1f39df6ffc83a693f42f84a4322bcf5235a8c'
)
EXPECTED_TRAIN_SHA256 = (
    '3e18da9a5c8dccea57b5155c244de1327ded05c6867f0059336f7a231ebbd967'
)
EXPECTED_VAL_SHA256 = (
    'bb92ecfdb0c379acc9971eaffed99154a2f985b063fd1f091aa8bceb09b07338'
)
SEED_GPU = {47: '0', 49: '1', 51: '2'}
EXPECTED_VALIDATION_NAMES = (
    'Sequence9', 'Sequence13', 'Sequence14', 'Sequence16',
    'Sequence17', 'Sequence20', 'Sequence29', 'Sequence31',
    'Sequence45', 'Sequence49', 'Sequence55', 'Sequence61',
    'Sequence68', 'Sequence74', 'Sequence77', 'Sequence84',
)
PAPER_THRESHOLDS = np.asarray([
    0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
    1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1,
], dtype=np.float64)
EXPECTED_THRESHOLD_GRID = np.unique(np.concatenate((
    PAPER_THRESHOLDS,
    np.round(np.arange(0.0, 1.005, 0.01, dtype=np.float64), 12),
)))
BOOTSTRAP_METRICS = (
    'delta_pd_at_fixed_fa',
    'delta_fa_at_fixed_pd',
    'fa_at_fixed_pd_relative_reduction',
    'delta_low_fa_pauc',
)
PRIMARY_FIELDS = (
    'pd_at_fixed_fa',
    'fa_at_fixed_pd',
    'low_fa_pauc',
    'pd_at_0_5',
    'fa_at_0_5',
    'auc',
    'auc_dense_grid',
    'pixel_iou_at_0_5',
    'pixel_f1_at_0_5',
    'elapsed_seconds',
    'cuda_peak_allocated_gib',
    'cuda_peak_reserved_gib',
    'parameters_m',
)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / 'experiments' / 'bc_tpro_stage1_2026-09-08'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, default=default_root)
    parser.add_argument(
        '--log-root', type=Path,
        default=repo_root / 'log' / 'sem_seg',
        help='Root containing the manifest log_dir experiment directories.',
    )
    parser.add_argument('--low-fa-cap', type=float, default=5e-5)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260908)
    return parser.parse_args()


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def load_manifest(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if not rows:
        raise ValueError('Empty manifest: %s' % path)
    required_fields = {
        'run_id', 'wave', 'model', 'structure_variant',
        'seed', 'gpu', 'log_dir',
    }
    missing_fields = required_fields - set(rows[0])
    if missing_fields:
        raise ValueError(
            'Manifest lacks required fields: %s'
            % ', '.join(sorted(missing_fields))
        )
    run_ids = [row['run_id'] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError('Manifest run_id values must be unique.')
    variant_protocol = (
        ('b1', 'none', 1, 'B1'),
        ('c0', 'temporal_control', 2, 'C0'),
        ('c1', 'center_multiscale', 3, 'C1'),
        ('c2', 'center_ring', 4, 'C2'),
    )
    expected_rows = []
    for run_prefix, variant, wave, log_label in variant_protocol:
        for seed in EXPECTED_SEEDS:
            expected_rows.append({
                'run_id': '%s_%s_seed%d' % (run_prefix, variant, seed),
                'wave': str(wave),
                'model': EXPECTED_MODEL,
                'structure_variant': variant,
                'seed': str(seed),
                'gpu': SEED_GPU[seed],
                'log_dir': (
                    '2026-09-08/NUDT-MIRSDT__SoftIoU-BCTPro-'
                    '%s_seed%d_E32' % (log_label, seed)
                ),
            })
    expected_by_run = {row['run_id']: row for row in expected_rows}
    observed_by_run = {row['run_id']: row for row in rows}
    if observed_by_run != expected_by_run:
        raise ValueError(
            'Manifest does not exactly match the 12-run preregistered '
            'wave/run_id/model/variant/seed/GPU/log_dir mapping.'
        )
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_split_artifacts(experiment_root):
    repo_root = Path(__file__).resolve().parents[1]
    paths = {
        'source train.txt': (
            repo_root.parent / 'datasets' / 'NUDT-MIRSDT' / 'train.txt',
            EXPECTED_SOURCE_SHA256,
        ),
        'Noise8 source train.txt': (
            repo_root.parent / 'datasets' / 'NUDT-MIRSDT-Noise8.0_FJY'
            / 'train.txt',
            EXPECTED_SOURCE_SHA256,
        ),
        'fixed train split': (
            experiment_root / 'splits' / 'train_sequences.txt',
            EXPECTED_TRAIN_SHA256,
        ),
        'fixed validation split': (
            experiment_root / 'splits' / 'val_sequences.txt',
            EXPECTED_VAL_SHA256,
        ),
    }
    for label, (path, expected_hash) in paths.items():
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Missing %s: %s' % (label, path))
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                '%s SHA256 mismatch: expected %s, found %s (%s)'
                % (label, expected_hash, observed_hash, path)
            )


def read_expected_validation_names(experiment_root):
    path = experiment_root / 'splits' / 'val_sequences.txt'
    if not path.is_file():
        raise FileNotFoundError('Missing validation sequence list: %s' % path)
    names = [
        line.strip() for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(names) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(
            'Expected %d validation sequences, found %d in %s.'
            % (EXPECTED_SEQUENCE_COUNT, len(names), path)
        )
    if len(names) != len(set(names)):
        raise ValueError('Validation sequence names must be unique: %s' % path)
    if names != list(EXPECTED_VALIDATION_NAMES):
        raise ValueError(
            'Validation sequence list does not match the preregistered order: %s'
            % path
        )
    return names


def _context_error(context, message):
    raise ValueError('%s: %s' % (context, message))


def _require_equal(context, field, observed, expected):
    if observed != expected:
        _context_error(
            context,
            '%s mismatch: expected %r, found %r'
            % (field, expected, observed),
        )


def _require_typed_equal(context, field, observed, expected):
    if type(observed) is not type(expected) or observed != expected:
        _context_error(
            context,
            '%s mismatch: expected %r (%s), found %r (%s)'
            % (
                field, expected, type(expected).__name__,
                observed, type(observed).__name__,
            ),
        )


def _require_int_equal(context, field, observed, expected):
    if isinstance(observed, bool) or not isinstance(observed, int):
        _context_error(
            context, '%s must be the integer %d' % (field, expected)
        )
    _require_equal(context, field, observed, expected)


def _require_path_equal(context, field, observed, expected):
    if not isinstance(observed, str) or not observed:
        _context_error(context, '%s must be a non-empty path' % field)
    observed_path = Path(observed).expanduser().resolve()
    expected_path = Path(expected).expanduser().resolve()
    _require_equal(context, field, observed_path, expected_path)


def _finite_number(context, field, observed):
    if isinstance(observed, bool) or not isinstance(
        observed, (int, float, np.integer, np.floating)
    ):
        _context_error(context, '%s must be numeric' % field)
    value = float(observed)
    if not np.isfinite(value):
        _context_error(context, '%s must be finite' % field)
    return value


def _require_close(context, field, observed, expected, atol=1e-12):
    value = _finite_number(context, field, observed)
    if not np.isclose(
        value, expected, rtol=0.0, atol=atol
    ):
        _context_error(
            context,
            '%s mismatch: expected %.17g, found %r'
            % (field, expected, observed),
        )
    return value


def parse_namespace_records(log_path):
    """Parse literal argparse Namespace records without executing them."""
    if not log_path.is_file():
        raise FileNotFoundError('Missing Namespace log: %s' % log_path)
    namespace_nodes = []
    for line_number, line in enumerate(
        log_path.read_text(encoding='utf-8').splitlines(), start=1
    ):
        marker = line.find('Namespace(')
        if marker < 0:
            continue
        fragment = line[marker:]
        try:
            node = ast.parse(fragment, mode='eval').body
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                'Cannot parse Namespace at %s line %d: %s'
                % (log_path, line_number, error)
            ) from error
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != 'Namespace'
            or node.args
            or any(keyword.arg is None for keyword in node.keywords)
        ):
            raise ValueError(
                'Unsupported Namespace representation at %s line %d.'
                % (log_path, line_number)
            )
        try:
            values = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
            }
        except (ValueError, TypeError) as error:
            raise ValueError(
                'Namespace contains a non-literal value at %s line %d.'
                % (log_path, line_number)
            ) from error
        namespace_nodes.append({
            'line_number': line_number,
            'values': values,
        })
    return namespace_nodes


def parse_training_namespace(log_path):
    """Parse one stable training Namespace record."""
    namespace_nodes = parse_namespace_records(log_path)
    if len(namespace_nodes) != 1:
        raise ValueError(
            'Expected exactly one stable Namespace line in %s, found %d.'
            % (log_path, len(namespace_nodes))
        )
    return namespace_nodes[0]['values']


def validate_training_namespace(
    metadata,
    job,
    context,
    experiment_root,
    log_root,
):
    repo_root = Path(__file__).resolve().parents[1]
    clean_data = (repo_root.parent / 'datasets' / 'NUDT-MIRSDT').resolve()
    train_list = (
        experiment_root / 'splits' / 'train_sequences.txt'
    ).resolve()
    validation_list = (
        experiment_root / 'splits' / 'val_sequences.txt'
    ).resolve()
    expected = {
        'seed': int(job['seed']),
        'gpu': job['gpu'],
        'gpu_num': 1,
        'model': job['model'],
        'structure_variant': job['structure_variant'],
        'dataset': CONDITION_DATASETS['clean_val'],
        'batch_size': 4,
        'gradient_accumulation_steps': 1,
        'epoch': EXPECTED_CHECKPOINT_EPOCH,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'decay_rate': 0.0001,
        'step_size': 10,
        'lr_decay': 0.7,
        'seqlen': EXPECTED_SEQUENCE_LENGTH,
        'patch_size': 128,
        'sample_rate': 0.1,
        'sequence_augmentation': 0,
        'loss': 'soft_iou',
        'threshold_eval': 0.5,
        'train_amp': 1,
        'eval_amp': 1,
        'eval_chunk_rows': 32,
        'eval_interval': 8,
        'skip_inprocess_validation': 1,
        'early_stopping_patience': 0,
        'train_workers': 4,
        'val_workers': 1,
        'prefetch_factor': 2,
        'deterministic': 1,
        'log_dir': job['log_dir'],
        'resume': 'never',
        'resume_checkpoint': None,
        'run_test_after_train': 0,
        'use_swanlab': 1,
        'swanlab_project': 'DeepPro-BC-TPro',
        'swanlab_group': 'bc-tpro-stage1-scratch',
        'swanlab_mode': 'cloud',
        'swanlab_resume': 'never',
        'base_ckpt': '',
        'spatial_ckpt': '',
        'st_ckpt': '',
        'freeze_pretrained': 0,
    }
    missing = sorted(set(expected) - set(metadata))
    if missing:
        _context_error(
            context,
            'training Namespace lacks fields: %s' % ', '.join(missing),
        )
    for field, expected_value in expected.items():
        _require_typed_equal(
            context,
            'training Namespace %s' % field,
            metadata[field],
            expected_value,
        )
    _require_path_equal(
        context, 'training Namespace datapath', metadata.get('datapath'),
        clean_data,
    )
    _require_path_equal(
        context, 'training Namespace train_sequence_list',
        metadata.get('train_sequence_list'), train_list,
    )
    _require_path_equal(
        context, 'training Namespace val_sequence_list',
        metadata.get('val_sequence_list'), validation_list,
    )
    _require_path_equal(
        context, 'training Namespace savepath', metadata.get('savepath'),
        Path(log_root).resolve().parent,
    )


def validate_evaluation_namespace(
    eval_log,
    metrics_path,
    job,
    condition,
    experiment_root,
    log_root,
    cache,
    context,
):
    log_key = str(eval_log)
    if log_key not in cache:
        cache[log_key] = parse_namespace_records(eval_log)
    expected_metrics_path = Path(metrics_path).expanduser().resolve()
    matches = []
    for record in cache[log_key]:
        observed = record['values'].get('metrics_json')
        if not isinstance(observed, str) or not observed:
            continue
        if Path(observed).expanduser().resolve() == expected_metrics_path:
            matches.append(record)
    if len(matches) != 1:
        _context_error(
            context,
            'expected exactly one eval Namespace matching metrics_json %s, '
            'found %d' % (expected_metrics_path, len(matches)),
        )
    metadata = matches[0]['values']
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (
        repo_root.parent / 'datasets' / CONDITION_DATASETS[condition]
    ).resolve()
    expected = {
        'amp': True,
        'batch_size': 1,
        'dataset': CONDITION_DATASETS[condition],
        'epoch': EXPECTED_CHECKPOINT_EPOCH,
        'eval_chunk_rows': 32,
        'gpu': job['gpu'],
        'log_dir': job['log_dir'],
        'output_only': False,
        'prefetch_factor': 1,
        'seqlen': EXPECTED_SEQUENCE_LENGTH,
        'sequence_start': 0,
        'sequence_stop': None,
        'test_workers': 1,
        'threshold_eval': 0.5,
        'threshold_grid_step': 0.01,
    }
    missing = sorted(set(expected) - set(metadata))
    if missing:
        _context_error(
            context,
            'evaluation Namespace lacks fields: %s' % ', '.join(missing),
        )
    for field, expected_value in expected.items():
        _require_typed_equal(
            context,
            'evaluation Namespace %s' % field,
            metadata[field],
            expected_value,
        )
    _require_path_equal(
        context, 'evaluation Namespace datapath', metadata.get('datapath'),
        data_root,
    )
    _require_path_equal(
        context, 'evaluation Namespace sequence_list',
        metadata.get('sequence_list'),
        experiment_root / 'splits' / 'val_sequences.txt',
    )
    _require_path_equal(
        context, 'evaluation Namespace logpath', metadata.get('logpath'),
        Path(log_root).resolve().parent,
    )
    _require_path_equal(
        context, 'evaluation Namespace metrics_json',
        metadata.get('metrics_json'), expected_metrics_path,
    )
    saved_marker = 'Paper-aligned metrics saved to %s.' % expected_metrics_path
    marker_count = eval_log.read_text(encoding='utf-8').count(saved_marker)
    if marker_count != 1:
        _context_error(
            context,
            'expected exactly one completed-evaluation marker for metrics_json, '
            'found %d' % marker_count,
        )


def validate_curve_counts(payload, expected_names, context):
    """Validate and return schema-v2 per-sequence curve counts."""
    schema_version = payload.get('schema_version')
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 2
    ):
        _context_error(context, 'schema_version must be the integer 2')
    counts = payload.get('curve_counts')
    if not isinstance(counts, dict):
        _context_error(context, 'schema v2 curve_counts are missing')

    sequence_names = counts.get('sequence_names')
    if sequence_names != list(expected_names):
        _context_error(
            context,
            'curve sequence_names do not exactly match the fixed validation order',
        )
    if len(sequence_names) != len(set(sequence_names)):
        _context_error(context, 'curve sequence_names contain duplicates')

    try:
        thresholds = np.asarray(counts['thresholds'], dtype=np.float64)
        false_counts = np.asarray(
            counts['false_pixels_by_sequence'], dtype=np.float64
        )
        true_counts = np.asarray(
            counts['true_targets_by_sequence'], dtype=np.float64
        )
        target_counts = np.asarray(
            counts['total_targets_by_sequence'], dtype=np.float64
        )
        pixels = np.asarray(
            counts['pixel_count_by_sequence'], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        _context_error(context, 'invalid curve_counts payload: %s' % error)

    if thresholds.ndim != 1 or thresholds.size < 2:
        _context_error(context, 'thresholds must be a one-dimensional grid')
    if not np.isfinite(thresholds).all():
        _context_error(context, 'thresholds contain non-finite values')
    if np.any(thresholds < 0) or np.any(thresholds > 1):
        _context_error(context, 'thresholds must lie in [0, 1]')
    if not np.all(np.diff(thresholds) > 0):
        _context_error(context, 'thresholds must be strictly increasing')
    if not np.array_equal(thresholds, EXPECTED_THRESHOLD_GRID):
        _context_error(
            context,
            'thresholds do not match the preregistered paper+0.01 grid',
        )
    half_matches = np.flatnonzero(np.isclose(
        thresholds, 0.5, rtol=0.0, atol=1e-15
    ))
    if half_matches.size != 1:
        _context_error(context, 'threshold grid must contain exactly one 0.5')

    expected_matrix_shape = (len(expected_names), thresholds.size)
    matrices = {
        'false_pixels_by_sequence': false_counts,
        'true_targets_by_sequence': true_counts,
        'total_targets_by_sequence': target_counts,
    }
    for name, values in matrices.items():
        if values.shape != expected_matrix_shape:
            _context_error(
                context,
                '%s shape must be %s, found %s'
                % (name, expected_matrix_shape, values.shape),
            )
        if not np.isfinite(values).all():
            _context_error(context, '%s contains non-finite values' % name)
        if np.any(values < 0):
            _context_error(context, '%s contains negative values' % name)
        if not np.equal(values, np.floor(values)).all():
            _context_error(context, '%s must contain integer counts' % name)
    if pixels.shape != (len(expected_names),):
        _context_error(
            context,
            'pixel_count_by_sequence shape must be (%d,), found %s'
            % (len(expected_names), pixels.shape),
        )
    if not np.isfinite(pixels).all() or np.any(pixels <= 0):
        _context_error(
            context, 'pixel_count_by_sequence must be finite and positive'
        )
    if not np.equal(pixels, np.floor(pixels)).all():
        _context_error(context, 'pixel_count_by_sequence must contain integers')
    if np.any(true_counts > target_counts):
        _context_error(context, 'true target counts exceed total targets')
    if not np.array_equal(
        target_counts,
        np.broadcast_to(target_counts[:, :1], target_counts.shape),
    ):
        _context_error(context, 'total targets vary across thresholds')
    if np.any(np.diff(true_counts, axis=1) > 0):
        _context_error(
            context, 'true target counts increase as threshold increases'
        )
    if np.any(np.diff(false_counts, axis=1) > 0):
        _context_error(
            context, 'false-pixel counts increase as threshold increases'
        )
    if np.any(false_counts > pixels[:, None]):
        _context_error(context, 'false-pixel counts exceed sequence pixels')
    if target_counts[:, 0].sum() <= 0:
        _context_error(context, 'validation curves contain no targets')

    return {
        'thresholds': thresholds,
        'sequence_names': list(sequence_names),
        'false_counts': false_counts,
        'true_counts': true_counts,
        'target_counts': target_counts,
        'pixels': pixels,
        'threshold_half_index': int(half_matches[0]),
    }


def validate_payload_metrics(payload, curve, context):
    """Recompute schema-v2 aggregate metrics from the validated counts."""
    paper_thresholds = np.asarray(
        payload.get('paper_thresholds'), dtype=np.float64
    )
    if not np.array_equal(paper_thresholds, PAPER_THRESHOLDS):
        _context_error(
            context, 'paper_thresholds do not match the paper protocol'
        )
    _require_close(
        context, 'operating_threshold', payload.get('operating_threshold'), 0.5
    )
    if payload.get('inference_amp') is not True:
        _context_error(context, 'inference_amp must be true')
    _require_int_equal(
        context, 'evaluation_windows', payload.get('evaluation_windows'),
        EXPECTED_EVALUATION_WINDOWS,
    )

    thresholds = curve['thresholds']
    true_totals = curve['true_counts'].sum(axis=0)
    target_totals = curve['target_counts'].sum(axis=0)
    false_totals = curve['false_counts'].sum(axis=0)
    pixel_total = float(curve['pixels'].sum())
    pd_curve = np.divide(
        true_totals,
        target_totals,
        out=np.full(true_totals.shape, np.nan, dtype=np.float64),
        where=target_totals != 0,
    )
    fa_curve = false_totals / pixel_total
    if not np.isfinite(pd_curve).all():
        _context_error(context, 'aggregate Pd curve is non-finite')

    paper_indices = np.asarray([
        int(np.flatnonzero(thresholds == threshold)[0])
        for threshold in PAPER_THRESHOLDS
    ], dtype=np.int64)
    expected_paper_auc = float(abs(np.trapz(
        pd_curve[paper_indices], fa_curve[paper_indices]
    )))
    expected_dense_auc = float(abs(np.trapz(pd_curve, fa_curve)))
    half_index = curve['threshold_half_index']
    expected = {
        'true_targets': float(true_totals[half_index]),
        'total_targets': float(target_totals[half_index]),
        'false_pixels': float(false_totals[half_index]),
        'pixel_count': pixel_total,
        'pd': float(pd_curve[half_index]),
        'fa': float(fa_curve[half_index]),
        'pd_percent': float(pd_curve[half_index] * 100.0),
        'fa_x1e5': float(fa_curve[half_index] * 1e5),
        'auc': expected_paper_auc,
        'auc_dense_grid': expected_dense_auc,
    }
    all_metrics = payload.get('all')
    if not isinstance(all_metrics, dict):
        _context_error(context, 'all metrics mapping is missing')
    for field, expected_value in expected.items():
        _require_close(
            context, 'all.%s' % field, all_metrics.get(field), expected_value
        )

    pixel_fields = (
        'pixel_iou_at_0_5',
        'pixel_precision_at_0_5',
        'pixel_recall_at_0_5',
        'pixel_f1_at_0_5',
    )
    pixel_values = {}
    for field in pixel_fields:
        try:
            observed = payload[field]
        except KeyError:
            _context_error(context, '%s is missing' % field)
        value = _finite_number(context, field, observed)
        if value < 0.0 or value > 1.0:
            _context_error(context, '%s must be finite and in [0, 1]' % field)
        pixel_values[field] = value
    precision = pixel_values['pixel_precision_at_0_5']
    recall = pixel_values['pixel_recall_at_0_5']
    expected_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    _require_close(
        context, 'pixel_f1_at_0_5',
        pixel_values['pixel_f1_at_0_5'], expected_f1,
    )
    iou = pixel_values['pixel_iou_at_0_5']
    expected_f1_from_iou = 2.0 * iou / (1.0 + iou)
    _require_close(
        context, 'pixel F1 implied by IoU',
        pixel_values['pixel_f1_at_0_5'], expected_f1_from_iou,
    )

    positive_fields = ('elapsed_seconds', 'parameters_m')
    for field in positive_fields:
        try:
            observed = payload[field]
        except KeyError:
            _context_error(context, '%s is missing' % field)
        value = _finite_number(context, field, observed)
        if value <= 0:
            _context_error(context, '%s must be finite and positive' % field)
    try:
        allocated_observed = payload['cuda_peak_allocated_gib']
        reserved_observed = payload['cuda_peak_reserved_gib']
    except KeyError as error:
        _context_error(context, 'CUDA peak metric is missing: %s' % error)
    peak_allocated = _finite_number(
        context, 'cuda_peak_allocated_gib', allocated_observed
    )
    peak_reserved = _finite_number(
        context, 'cuda_peak_reserved_gib', reserved_observed
    )
    if (
        peak_allocated <= 0
        or peak_reserved < peak_allocated
    ):
        _context_error(
            context,
            'CUDA peaks must be finite, allocated>0, and reserved>=allocated',
        )

    return {
        'pd_curve': pd_curve,
        'fa_curve': fa_curve,
        'paper_auc': expected_paper_auc,
        'dense_auc': expected_dense_auc,
    }


def _load_checkpoint(path, cache):
    cache_key = str(path)
    if cache_key not in cache:
        try:
            cache[cache_key] = torch.load(
                str(path), map_location='cpu', weights_only=True
            )
        except Exception as error:
            raise ValueError(
                'Cannot safely load checkpoint identity from %s: %s'
                % (path, error)
            ) from error
    return cache[cache_key]


def validate_payload_identity(
    payload,
    job,
    condition,
    expected_names,
    experiment_root,
    log_root,
    metrics_path,
    checkpoint_cache=None,
    log_cache=None,
    eval_cache=None,
    context='metrics payload',
):
    """Validate one metrics file against the manifest and run artifacts."""
    if checkpoint_cache is None:
        checkpoint_cache = {}
    if log_cache is None:
        log_cache = {}
    if eval_cache is None:
        eval_cache = {}
    if condition not in CONDITION_DATASETS:
        _context_error(context, 'unknown condition %r' % condition)

    _require_equal(
        context, 'dataset', payload.get('dataset'),
        CONDITION_DATASETS[condition],
    )
    _require_equal(context, 'model', payload.get('model'), job['model'])
    _require_int_equal(
        context, 'checkpoint_epoch', payload.get('checkpoint_epoch'),
        EXPECTED_CHECKPOINT_EPOCH,
    )
    _require_int_equal(
        context, 'sequence_length', payload.get('sequence_length'),
        EXPECTED_SEQUENCE_LENGTH,
    )
    _require_int_equal(
        context, 'sequence_count', payload.get('sequence_count'),
        EXPECTED_SEQUENCE_COUNT,
    )

    curve = validate_curve_counts(payload, expected_names, context)
    validate_payload_metrics(payload, curve, context)
    checkpoint_value = payload.get('checkpoint')
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        _context_error(context, 'checkpoint must be a non-empty path string')
    expected_checkpoint = (
        Path(log_root).expanduser().resolve()
        / job['log_dir'] / 'checkpoints'
        / ('epoch_%d_model.pth' % EXPECTED_CHECKPOINT_EPOCH)
    ).resolve()
    observed_checkpoint = Path(checkpoint_value).expanduser().resolve()
    _require_equal(
        context, 'checkpoint path', observed_checkpoint, expected_checkpoint
    )
    if not expected_checkpoint.is_file():
        _context_error(
            context, 'checkpoint does not exist: %s' % expected_checkpoint
        )
    checkpoint = _load_checkpoint(expected_checkpoint, checkpoint_cache)
    if not isinstance(checkpoint, dict):
        _context_error(context, 'checkpoint is not a mapping')
    _require_equal(
        context, 'checkpoint model_name', checkpoint.get('model_name'),
        job['model'],
    )
    checkpoint_epoch = checkpoint.get('epoch')
    _require_int_equal(
        context, 'checkpoint zero-based epoch', checkpoint_epoch,
        EXPECTED_CHECKPOINT_EPOCH - 1,
    )
    model_config = checkpoint.get('model_config')
    if not isinstance(model_config, dict):
        _context_error(context, 'checkpoint model_config is missing')
    _require_equal(
        context, 'checkpoint structure_variant',
        model_config.get('structure_variant'), job['structure_variant'],
    )

    run_dir = expected_checkpoint.parents[1]
    training_log = run_dir / 'logs' / ('%s.txt' % job['model'])
    log_key = str(training_log)
    if log_key not in log_cache:
        log_cache[log_key] = parse_training_namespace(training_log)
    validate_training_namespace(
        log_cache[log_key], job, context, experiment_root, log_root
    )
    validate_evaluation_namespace(
        run_dir / ('eval_epoch-%d.txt' % EXPECTED_CHECKPOINT_EPOCH),
        metrics_path,
        job,
        condition,
        experiment_root,
        log_root,
        eval_cache,
        context,
    )
    return curve


def curve_from_metrics(payload):
    counts = payload.get('curve_counts')
    if not counts:
        raise ValueError('schema v2 curve_counts are missing')
    thresholds = np.asarray(counts['thresholds'], dtype=np.float64)
    false_counts = np.asarray(
        counts['false_pixels_by_sequence'], dtype=np.float64
    )
    true_counts = np.asarray(
        counts['true_targets_by_sequence'], dtype=np.float64
    )
    target_counts = np.asarray(
        counts['total_targets_by_sequence'], dtype=np.float64
    )
    pixels = np.asarray(
        counts['pixel_count_by_sequence'], dtype=np.float64
    )
    if false_counts.shape != true_counts.shape or false_counts.shape != target_counts.shape:
        raise ValueError('Inconsistent curve-count shapes')
    if false_counts.shape[1] != thresholds.size:
        raise ValueError('Threshold/count length mismatch')
    pd_curve = safe_divide(true_counts.sum(axis=0), target_counts.sum(axis=0))
    fa_curve = safe_divide(false_counts.sum(axis=0), pixels.sum())
    return thresholds, pd_curve, fa_curve


def low_fa_pauc(pd_curve, fa_curve, cap):
    """Area under the monotone empirical Pd-Fa envelope, normalized by cap."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('low-Fa cap must be finite and positive')
    order = np.argsort(fa_curve)
    fa_sorted = np.asarray(fa_curve[order], dtype=np.float64)
    pd_sorted = np.asarray(pd_curve[order], dtype=np.float64)

    unique_fa, inverse = np.unique(fa_sorted, return_inverse=True)
    envelope = np.full(unique_fa.shape, -np.inf, dtype=np.float64)
    np.maximum.at(envelope, inverse, pd_sorted)
    envelope = np.maximum.accumulate(envelope)
    if unique_fa[0] > 0:
        unique_fa = np.insert(unique_fa, 0, 0.0)
        envelope = np.insert(envelope, 0, 0.0)

    if unique_fa[-1] < cap:
        unique_fa = np.append(unique_fa, cap)
        envelope = np.append(envelope, envelope[-1])
    elif not np.any(np.isclose(unique_fa, cap, rtol=0.0, atol=1e-15)):
        pd_at_cap = np.interp(cap, unique_fa, envelope)
        insertion = np.searchsorted(unique_fa, cap)
        unique_fa = np.insert(unique_fa, insertion, cap)
        envelope = np.insert(envelope, insertion, pd_at_cap)

    keep = unique_fa <= cap + 1e-15
    return float(np.trapz(envelope[keep], unique_fa[keep]) / cap)


def max_pd_at_fa(pd_curve, fa_curve, target_fa):
    eligible = fa_curve <= target_fa + 1e-15
    if not np.any(eligible):
        return float('nan')
    return float(np.max(pd_curve[eligible]))


def min_fa_at_pd(pd_curve, fa_curve, target_pd):
    eligible = pd_curve >= target_pd - 1e-15
    if not np.any(eligible):
        return float('nan')
    return float(np.min(fa_curve[eligible]))


def working_point_diagnostics(
    thresholds,
    pd_curve,
    fa_curve,
    target,
    mode,
):
    if mode == 'max_pd_at_fa':
        eligible = fa_curve <= target + 1e-15
        values = pd_curve
        best = max_pd_at_fa(pd_curve, fa_curve, target)
        choose_last = False
    elif mode == 'min_fa_at_pd':
        eligible = pd_curve >= target - 1e-15
        values = fa_curve
        best = min_fa_at_pd(pd_curve, fa_curve, target)
        choose_last = True
    else:
        raise ValueError('Unknown working-point mode: %s' % mode)
    if not np.isfinite(best):
        return {
            'selected_threshold': float('nan'),
            'selected_is_endpoint': False,
            'plateau_threshold_min': float('nan'),
            'plateau_threshold_max': float('nan'),
            'plateau_points': 0,
        }
    plateau = eligible & np.isclose(
        values, best, rtol=0.0, atol=1e-15
    )
    indices = np.flatnonzero(plateau)
    selected_index = int(indices[-1] if choose_last else indices[0])
    return {
        'selected_threshold': float(thresholds[selected_index]),
        'selected_is_endpoint': bool(
            selected_index == 0 or selected_index == thresholds.size - 1
        ),
        'plateau_threshold_min': float(thresholds[indices[0]]),
        'plateau_threshold_max': float(thresholds[indices[-1]]),
        'plateau_points': int(indices.size),
    }


def low_fa_support_diagnostics(fa_curve, cap):
    unique_fa = np.unique(np.asarray(fa_curve, dtype=np.float64))
    internal = unique_fa[(unique_fa > 0.0) & (unique_fa < cap)]
    empirical_at_or_below = unique_fa[unique_fa <= cap]
    origin_only = internal.size == 0
    return {
        'low_fa_internal_unique_points': int(internal.size),
        'low_fa_empirical_points_at_or_below_cap': int(
            empirical_at_or_below.size
        ),
        'low_fa_min_empirical_fa': float(unique_fa[0]),
        'low_fa_max_empirical_fa_below_cap': (
            float(empirical_at_or_below[-1])
            if empirical_at_or_below.size else float('nan')
        ),
        'low_fa_cap_empirically_bracketed': bool(
            np.any(unique_fa <= cap) and np.any(unique_fa >= cap)
        ),
        'low_fa_origin_interpolation_only': bool(origin_only),
        'low_fa_support': (
            'origin-interpolation-only'
            if origin_only else 'empirical-internal-support'
        ),
    }


def sample_sd(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        return float('nan')
    return float(np.std(values, ddof=1))


def mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float('nan')
    return float(np.mean(values))


def fmt(value, digits=6):
    return 'NA' if not np.isfinite(value) else ('%.*f' % (digits, value))


def load_rows(experiment_root, manifest, low_fa_cap, log_root):
    metrics_root = experiment_root / 'metrics'
    validate_split_artifacts(experiment_root)
    expected_names = read_expected_validation_names(experiment_root)
    loaded = {}
    missing = []
    checkpoint_cache = {}
    log_cache = {}
    eval_cache = {}
    reference_thresholds = None
    for job in manifest:
        for condition in CONDITIONS:
            path = metrics_root / ('%s__%s.json' % (job['run_id'], condition))
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding='utf-8'))
            curve = validate_payload_identity(
                payload,
                job,
                condition,
                expected_names,
                experiment_root,
                log_root,
                path,
                checkpoint_cache=checkpoint_cache,
                log_cache=log_cache,
                eval_cache=eval_cache,
                context=str(path),
            )
            thresholds = curve['thresholds']
            if reference_thresholds is None:
                reference_thresholds = thresholds
            elif not np.array_equal(reference_thresholds, thresholds):
                _context_error(
                    str(path),
                    'threshold grid differs from earlier metrics payloads',
                )
            pd_curve = safe_divide(
                curve['true_counts'].sum(axis=0),
                curve['target_counts'].sum(axis=0),
            )
            fa_curve = safe_divide(
                curve['false_counts'].sum(axis=0),
                curve['pixels'].sum(),
            )
            loaded[(job['run_id'], condition)] = {
                'job': job,
                'condition': condition,
                'payload': payload,
                'curve': curve,
                'thresholds': thresholds,
                'pd_curve': pd_curve,
                'fa_curve': fa_curve,
                'low_fa_pauc': low_fa_pauc(pd_curve, fa_curve, low_fa_cap),
                'low_fa_support': low_fa_support_diagnostics(
                    fa_curve, low_fa_cap
                ),
            }

    reference_targets = None
    reference_pixels = None
    for entry in loaded.values():
        curve = entry['curve']
        if reference_targets is None:
            reference_targets = curve['target_counts']
            reference_pixels = curve['pixels']
        elif (
            not np.array_equal(reference_targets, curve['target_counts'])
            or not np.array_equal(reference_pixels, curve['pixels'])
        ):
            _context_error(
                '%s/%s' % (entry['job']['run_id'], entry['condition']),
                'target counts or sequence pixel counts differ across '
                'runs/conditions',
            )

    for job in manifest:
        clean = loaded.get((job['run_id'], 'clean_val'))
        noise = loaded.get((job['run_id'], 'noise8_val'))
        if clean is None or noise is None:
            continue
        _require_close(
            job['run_id'], 'clean/noise parameters_m',
            clean['payload'].get('parameters_m'),
            float(noise['payload']['parameters_m']),
        )

    baseline = {}
    for entry in loaded.values():
        if entry['job']['structure_variant'] == 'none':
            key = (entry['job']['seed'], entry['condition'])
            baseline[key] = entry

    rows = []
    for entry in loaded.values():
        job = entry['job']
        condition = entry['condition']
        base = baseline.get((job['seed'], condition))
        if base is None:
            continue
        base_all = base['payload']['all']
        target_fa = float(base_all['fa'])
        target_pd = float(base_all['pd'])
        payload = entry['payload']
        all_metrics = payload['all']
        pd_working_point = working_point_diagnostics(
            entry['thresholds'], entry['pd_curve'], entry['fa_curve'],
            target_fa, 'max_pd_at_fa',
        )
        fa_working_point = working_point_diagnostics(
            entry['thresholds'], entry['pd_curve'], entry['fa_curve'],
            target_pd, 'min_fa_at_pd',
        )
        row = {
            'run_id': job['run_id'],
            'variant': job['structure_variant'],
            'variant_label': VARIANT_LABELS[job['structure_variant']],
            'seed': int(job['seed']),
            'gpu': int(job['gpu']),
            'condition': condition,
            'baseline_target_fa': target_fa,
            'baseline_target_pd': target_pd,
            'pd_at_fixed_fa': max_pd_at_fa(
                entry['pd_curve'], entry['fa_curve'], target_fa
            ),
            'fa_at_fixed_pd': min_fa_at_pd(
                entry['pd_curve'], entry['fa_curve'], target_pd
            ),
            'low_fa_pauc': entry['low_fa_pauc'],
            'pd_at_0_5': float(all_metrics['pd']),
            'fa_at_0_5': float(all_metrics['fa']),
            'auc': float(all_metrics['auc']),
            'auc_dense_grid': float(all_metrics['auc_dense_grid']),
            'pixel_iou_at_0_5': float(payload['pixel_iou_at_0_5']),
            'pixel_f1_at_0_5': float(payload['pixel_f1_at_0_5']),
            'elapsed_seconds': float(payload['elapsed_seconds']),
            'cuda_peak_allocated_gib': float(
                payload.get('cuda_peak_allocated_gib', float('nan'))
            ),
            'cuda_peak_reserved_gib': float(
                payload.get('cuda_peak_reserved_gib', float('nan'))
            ),
            'parameters_m': float(payload['parameters_m']),
            'checkpoint_epoch': int(payload['checkpoint_epoch']),
        }
        row.update({
            'pd_fixed_' + key: value
            for key, value in pd_working_point.items()
        })
        row.update({
            'fa_fixed_' + key: value
            for key, value in fa_working_point.items()
        })
        row.update(entry['low_fa_support'])
        rows.append(row)

    row_index = {
        (row['seed'], row['condition'], row['variant']): row for row in rows
    }
    for row in rows:
        base_row = row_index[(row['seed'], row['condition'], 'none')]
        for field in PRIMARY_FIELDS:
            row['delta_' + field] = row[field] - base_row[field]
        base_fa = base_row['fa_at_fixed_pd']
        row['fa_at_fixed_pd_relative_reduction'] = (
            (base_fa - row['fa_at_fixed_pd']) / base_fa
            if base_fa > 0 and np.isfinite(row['fa_at_fixed_pd'])
            else float('nan')
        )
        row['latency_ratio'] = (
            row['elapsed_seconds'] / base_row['elapsed_seconds']
            if base_row['elapsed_seconds'] > 0 else float('nan')
        )
    return rows, missing, loaded


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['variant'], row['condition'])].append(row)
    summaries = []
    numeric_fields = list(PRIMARY_FIELDS) + [
        'delta_' + field for field in PRIMARY_FIELDS
    ] + ['fa_at_fixed_pd_relative_reduction', 'latency_ratio']
    for variant in VARIANT_LABELS:
        for condition in CONDITIONS:
            group = groups.get((variant, condition))
            if not group:
                continue
            summary = {
                'variant': variant,
                'variant_label': VARIANT_LABELS[variant],
                'condition': condition,
                'n': len(group),
            }
            for field in numeric_fields:
                values = [row[field] for row in group]
                summary[field + '_mean'] = mean(values)
                summary[field + '_sd'] = sample_sd(values)
            summaries.append(summary)
    return summaries


def _bootstrap_weights(replicates, sequence_count, seed):
    if replicates <= 0:
        raise ValueError('bootstrap_replicates must be positive')
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(
        0, sequence_count, size=(replicates, sequence_count)
    )
    weights = np.zeros((replicates, sequence_count), dtype=np.int16)
    replicate_indices = np.repeat(np.arange(replicates), sequence_count)
    np.add.at(weights, (replicate_indices, draws.reshape(-1)), 1)
    return weights


def _resampled_curves(curve, weights):
    true_counts = weights @ curve['true_counts']
    target_counts = weights @ curve['target_counts']
    false_counts = weights @ curve['false_counts']
    pixel_counts = weights @ curve['pixels']
    return {
        'pd': np.divide(
            true_counts,
            target_counts,
            out=np.full(true_counts.shape, np.nan, dtype=np.float64),
            where=target_counts != 0,
        ),
        'fa': np.divide(
            false_counts,
            pixel_counts[:, None],
            out=np.full(false_counts.shape, np.nan, dtype=np.float64),
            where=pixel_counts[:, None] != 0,
        ),
    }


def _rowwise_max(values, eligible):
    masked = np.where(eligible, values, -np.inf)
    result = masked.max(axis=1)
    result[~eligible.any(axis=1)] = np.nan
    return result


def _rowwise_min(values, eligible):
    masked = np.where(eligible, values, np.inf)
    result = masked.min(axis=1)
    result[~eligible.any(axis=1)] = np.nan
    return result


def _bootstrap_pauc(pd_curves, fa_curves, cap):
    values = []
    for pd_curve, fa_curve in zip(pd_curves, fa_curves):
        if not np.isfinite(pd_curve).all() or not np.isfinite(fa_curve).all():
            values.append(float('nan'))
        else:
            values.append(low_fa_pauc(pd_curve, fa_curve, cap))
    return np.asarray(values, dtype=np.float64)


def _mean_if_all_finite(per_seed_values):
    values = np.stack(per_seed_values, axis=0)
    valid = np.isfinite(values).all(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    if np.any(valid):
        result[valid] = values[:, valid].mean(axis=0)
    return result


def _bootstrap_summary_row(
    variant,
    condition,
    metric,
    point_estimate,
    values,
    replicates,
):
    finite_values = values[np.isfinite(values)]
    valid_replicates = int(finite_values.size)
    valid_rate = valid_replicates / float(replicates)
    median = (
        float(np.percentile(finite_values, 50.0))
        if valid_replicates else float('nan')
    )
    ci_available = valid_replicates > 0 and valid_rate >= 0.95
    if ci_available:
        ci_low, ci_high = np.percentile(
            finite_values, [2.5, 97.5]
        ).tolist()
    else:
        ci_low = ci_high = float('nan')
    return {
        'variant': variant,
        'variant_label': VARIANT_LABELS[variant],
        'condition': condition,
        'metric': metric,
        'point_estimate': point_estimate,
        'bootstrap_median': median,
        'ci95_low': float(ci_low),
        'ci95_high': float(ci_high),
        'valid_replicates': valid_replicates,
        'replicates': int(replicates),
        'valid_rate': valid_rate,
        'ci_available': bool(ci_available),
    }


def paired_video_bootstrap(
    loaded,
    rows,
    replicates=10000,
    seed=20260908,
    low_fa_cap=5e-5,
):
    """Paired sequence bootstrap, conditional on the three training seeds."""
    if not np.isfinite(low_fa_cap) or low_fa_cap <= 0:
        raise ValueError('low_fa_cap must be finite and positive')
    if not loaded:
        return []
    first_entry = next(iter(loaded.values()))
    sequence_count = len(first_entry['curve']['sequence_names'])
    weights = _bootstrap_weights(replicates, sequence_count, seed)
    entry_index = {
        (
            entry['job']['structure_variant'],
            int(entry['job']['seed']),
            entry['condition'],
        ): entry
        for entry in loaded.values()
    }
    row_index = {
        (row['variant'], row['seed'], row['condition']): row
        for row in rows
    }
    pauc_cache = {}
    summaries = []

    for variant in VARIANT_LABELS:
        if variant == 'none':
            continue
        for condition in CONDITIONS:
            required_keys = [
                (variant_name, training_seed, condition)
                for training_seed in EXPECTED_SEEDS
                for variant_name in ('none', variant)
            ]
            if not all(key in entry_index for key in required_keys):
                continue

            metric_values = {
                metric: [] for metric in BOOTSTRAP_METRICS
            }
            for training_seed in EXPECTED_SEEDS:
                base_entry = entry_index[
                    ('none', training_seed, condition)
                ]
                candidate_entry = entry_index[
                    (variant, training_seed, condition)
                ]
                base = _resampled_curves(base_entry['curve'], weights)
                candidate = _resampled_curves(
                    candidate_entry['curve'], weights
                )
                half_index = base_entry['curve']['threshold_half_index']
                target_fa = base['fa'][:, half_index]
                target_pd = base['pd'][:, half_index]

                base_pd_fixed = _rowwise_max(
                    base['pd'], base['fa'] <= target_fa[:, None] + 1e-15
                )
                candidate_pd_fixed = _rowwise_max(
                    candidate['pd'],
                    candidate['fa'] <= target_fa[:, None] + 1e-15,
                )
                base_fa_fixed = _rowwise_min(
                    base['fa'], base['pd'] >= target_pd[:, None] - 1e-15
                )
                candidate_fa_fixed = _rowwise_min(
                    candidate['fa'],
                    candidate['pd'] >= target_pd[:, None] - 1e-15,
                )
                delta_fa = candidate_fa_fixed - base_fa_fixed
                relative_fa_reduction = np.full(
                    replicates, np.nan, dtype=np.float64
                )
                relative_valid = (
                    np.isfinite(delta_fa)
                    & np.isfinite(base_fa_fixed)
                    & (base_fa_fixed > 0)
                )
                relative_fa_reduction[relative_valid] = (
                    -delta_fa[relative_valid]
                    / base_fa_fixed[relative_valid]
                )

                for entry, curves in (
                    (base_entry, base),
                    (candidate_entry, candidate),
                ):
                    cache_key = (
                        entry['job']['run_id'], entry['condition']
                    )
                    if cache_key not in pauc_cache:
                        pauc_cache[cache_key] = _bootstrap_pauc(
                            curves['pd'], curves['fa'], low_fa_cap
                        )
                base_pauc = pauc_cache[
                    (base_entry['job']['run_id'], condition)
                ]
                candidate_pauc = pauc_cache[
                    (candidate_entry['job']['run_id'], condition)
                ]
                metric_values['delta_pd_at_fixed_fa'].append(
                    candidate_pd_fixed - base_pd_fixed
                )
                metric_values['delta_fa_at_fixed_pd'].append(delta_fa)
                metric_values[
                    'fa_at_fixed_pd_relative_reduction'
                ].append(relative_fa_reduction)
                metric_values['delta_low_fa_pauc'].append(
                    candidate_pauc - base_pauc
                )

            point_rows = [
                row_index[(variant, training_seed, condition)]
                for training_seed in EXPECTED_SEEDS
            ]
            point_fields = {
                'delta_pd_at_fixed_fa': 'delta_pd_at_fixed_fa',
                'delta_fa_at_fixed_pd': 'delta_fa_at_fixed_pd',
                'fa_at_fixed_pd_relative_reduction': (
                    'fa_at_fixed_pd_relative_reduction'
                ),
                'delta_low_fa_pauc': 'delta_low_fa_pauc',
            }
            for metric in BOOTSTRAP_METRICS:
                paired_means = _mean_if_all_finite(metric_values[metric])
                point_estimate = mean([
                    row[point_fields[metric]] for row in point_rows
                ])
                summaries.append(_bootstrap_summary_row(
                    variant,
                    condition,
                    metric,
                    point_estimate,
                    paired_means,
                    replicates,
                ))
    return summaries


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gate_status(rows):
    expected_keys = {
        (variant, seed, condition)
        for variant in VARIANT_LABELS
        for seed in EXPECTED_SEEDS
        for condition in CONDITIONS
    }
    observed_keys = [
        (row['variant'], row['seed'], row['condition']) for row in rows
    ]
    if len(observed_keys) != len(set(observed_keys)):
        return 'NOT_EVALUABLE', [
            'Duplicate variant/seed/condition result keys were found.'
        ]
    observed_key_set = set(observed_keys)
    if observed_key_set != expected_keys:
        missing = sorted(expected_keys - observed_key_set)
        unexpected = sorted(observed_key_set - expected_keys)
        details = [
            'The gate requires all 24 B1/C0/C1/C2 × 3-seed × '
            'Clean/Noise8 keys; controls cannot be omitted.'
        ]
        if missing:
            details.append('Missing keys: %s' % missing)
        if unexpected:
            details.append('Unexpected keys: %s' % unexpected)
        return 'NOT_EVALUABLE', details
    for variant in VARIANT_LABELS:
        for condition in CONDITIONS:
            seeds = sorted(
                row['seed'] for row in rows
                if row['variant'] == variant
                and row['condition'] == condition
            )
            if seeds != list(EXPECTED_SEEDS):
                return 'NOT_EVALUABLE', [
                    '%s/%s seed set is %s, expected %s.'
                    % (variant, condition, seeds, EXPECTED_SEEDS)
                ]
    invalid_latency = [
        (row['variant'], row['seed'], row['condition'])
        for row in rows
        if (
            not np.isfinite(row['elapsed_seconds'])
            or row['elapsed_seconds'] <= 0
            or not np.isfinite(row['latency_ratio'])
            or row['latency_ratio'] <= 0
        )
    ]
    if invalid_latency:
        return 'NOT_EVALUABLE', [
            'Latency must be finite and positive for every result: %s'
            % invalid_latency
        ]

    c2 = [row for row in rows if row['variant'] == 'center_ring']
    by_condition = {
        condition: [row for row in c2 if row['condition'] == condition]
        for condition in CONDITIONS
    }
    clean = by_condition['clean_val']
    noise = by_condition['noise8_val']
    required_gate_fields = (
        'fa_at_fixed_pd_relative_reduction',
        'delta_pd_at_fixed_fa',
        'latency_ratio',
    )
    invalid = [
        '%s/%s/%s' % (row['condition'], row['seed'], field)
        for row in c2
        for field in required_gate_fields
        if not np.isfinite(row[field])
    ]
    if invalid:
        return 'FAIL', [
            'A required working point is unreachable or non-finite: %s'
            % ', '.join(invalid)
        ]
    clean_fa_reduction = mean([
        row['fa_at_fixed_pd_relative_reduction'] for row in clean
    ])
    clean_pd_delta = mean([row['delta_pd_at_fixed_fa'] for row in clean])
    noise_fa_reduction = mean([
        row['fa_at_fixed_pd_relative_reduction'] for row in noise
    ])
    noise_pd_delta = mean([row['delta_pd_at_fixed_fa'] for row in noise])
    worst_fa_reduction = min(
        row['fa_at_fixed_pd_relative_reduction'] for row in c2
    )
    worst_pd_delta = min(row['delta_pd_at_fixed_fa'] for row in c2)
    max_latency = max(row['latency_ratio'] for row in c2)
    checks = [
        ('Clean mean Fa reduction >=20%', clean_fa_reduction >= 0.20),
        ('Clean mean Pd loss <=1pp', clean_pd_delta >= -0.01),
        ('Noise8 direction does not reverse',
         noise_fa_reduction >= 0.0 and noise_pd_delta >= -0.01),
        ('No severe reverse seed (analysis amendment: Fa >=-20%, Pd >=-3pp)',
         worst_fa_reduction >= -0.20 and worst_pd_delta >= -0.03),
        ('Every latency ratio <=1.3x', max_latency <= 1.30),
    ]
    details = ['%s: %s' % (name, 'PASS' if passed else 'FAIL')
               for name, passed in checks]
    return ('PASS' if all(passed for _, passed in checks) else 'FAIL'), details


def _format_bootstrap_metric(metric, value):
    if not np.isfinite(value):
        return 'NA'
    if metric == 'delta_pd_at_fixed_fa':
        return '%.3f pp' % (value * 100.0)
    if metric == 'delta_fa_at_fixed_pd':
        return '%.3f ×1e-5' % (value * 1e5)
    if metric == 'fa_at_fixed_pd_relative_reduction':
        return '%.2f%%' % (value * 100.0)
    return '%.4f' % value


def write_markdown(
    path,
    rows,
    summaries,
    bootstrap_rows,
    missing,
    low_fa_cap,
    bootstrap_replicates,
    bootstrap_seed,
):
    status, gate_details = gate_status(rows)
    lines = [
        '# BC-TPro 第一阶段结果分析',
        '',
        '状态：已获得 %d/24 份 Clean/Noise8 验证指标；C2 继续门槛：**%s**。'
        % (len(rows), status),
        '',
        '主表使用 `epoch_32_model.pth`。`Pd@fixedFa` 和 `Fa@fixedPd` 的工作点由同 seed、同条件 B1 在阈值 0.5 处预先定义；低 Fa pAUC 在 `Fa<=%.1e` 内归一化。'
        % low_fa_cap,
        '',
        '## 身份与完整性检查',
        '',
        '- manifest 必须精确匹配 12 个计划 run 的 wave/model/variant/seed/GPU/log_dir 映射。',
        '- Clean/Noise8 `train.txt`、固定 train split、固定 validation split 的 SHA256 均已钉住。',
        '- 每份已加载指标均通过训练 Namespace、checkpoint metadata、匹配 `metrics_json` 的独立评测 Namespace，以及 counts 重算检查。',
        '',
        '## 三随机种子汇总',
        '',
        '| Condition | Variant | n | Pd@fixedFa (%) | Fa@fixedPd (×1e-5) | low-Fa pAUC | Paper/Dense AUC | Pd@0.5 (%) | Fa@0.5 (×1e-5) | Pixel IoU/F1 (%) | Time (s) | Peak alloc/reserved (GiB) | Params (M) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for summary in summaries:
        lines.append(
            '| {condition} | {label} | {n} | {pd} ± {pd_sd} | {fa} ± {fa_sd} | {pauc} ± {pauc_sd} | {auc} ± {auc_sd} / {dense_auc} ± {dense_auc_sd} | {pd05} ± {pd05_sd} | {fa05} ± {fa05_sd} | {iou} ± {iou_sd} / {f1} ± {f1_sd} | {time} ± {time_sd} | {alloc} ± {alloc_sd} / {reserved} ± {reserved_sd} | {params} |'.format(
                condition=summary['condition'],
                label=summary['variant_label'],
                n=summary['n'],
                pd=fmt(summary['pd_at_fixed_fa_mean'] * 100, 3),
                pd_sd=fmt(summary['pd_at_fixed_fa_sd'] * 100, 3),
                fa=fmt(summary['fa_at_fixed_pd_mean'] * 1e5, 3),
                fa_sd=fmt(summary['fa_at_fixed_pd_sd'] * 1e5, 3),
                pauc=fmt(summary['low_fa_pauc_mean'], 4),
                pauc_sd=fmt(summary['low_fa_pauc_sd'], 4),
                auc=fmt(summary['auc_mean'], 4),
                auc_sd=fmt(summary['auc_sd'], 4),
                dense_auc=fmt(summary['auc_dense_grid_mean'], 4),
                dense_auc_sd=fmt(summary['auc_dense_grid_sd'], 4),
                pd05=fmt(summary['pd_at_0_5_mean'] * 100, 3),
                pd05_sd=fmt(summary['pd_at_0_5_sd'] * 100, 3),
                fa05=fmt(summary['fa_at_0_5_mean'] * 1e5, 3),
                fa05_sd=fmt(summary['fa_at_0_5_sd'] * 1e5, 3),
                iou=fmt(summary['pixel_iou_at_0_5_mean'] * 100, 3),
                iou_sd=fmt(summary['pixel_iou_at_0_5_sd'] * 100, 3),
                f1=fmt(summary['pixel_f1_at_0_5_mean'] * 100, 3),
                f1_sd=fmt(summary['pixel_f1_at_0_5_sd'] * 100, 3),
                time=fmt(summary['elapsed_seconds_mean'], 1),
                time_sd=fmt(summary['elapsed_seconds_sd'], 1),
                alloc=fmt(summary['cuda_peak_allocated_gib_mean'], 3),
                alloc_sd=fmt(summary['cuda_peak_allocated_gib_sd'], 3),
                reserved=fmt(summary['cuda_peak_reserved_gib_mean'], 3),
                reserved_sd=fmt(summary['cuda_peak_reserved_gib_sd'], 3),
                params=fmt(summary['parameters_m_mean'], 6),
            )
        )

    lines.extend([
        '',
        '## 配对差值（相对同 seed B1）',
        '',
        '| Condition | Variant | ΔPd@fixedFa (pp) | Fa@fixedPd 相对下降 | ΔpAUC | Latency ratio |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for summary in summaries:
        if summary['variant'] == 'none':
            continue
        lines.append(
            '| {condition} | {label} | {pd} ± {pd_sd} | {fa} ± {fa_sd} | {pauc} ± {pauc_sd} | {latency} ± {latency_sd} |'.format(
                condition=summary['condition'],
                label=summary['variant_label'],
                pd=fmt(summary['delta_pd_at_fixed_fa_mean'] * 100, 3),
                pd_sd=fmt(summary['delta_pd_at_fixed_fa_sd'] * 100, 3),
                fa=fmt(summary['fa_at_fixed_pd_relative_reduction_mean'] * 100, 2) + '%',
                fa_sd=fmt(summary['fa_at_fixed_pd_relative_reduction_sd'] * 100, 2) + '%',
                pauc=fmt(summary['delta_low_fa_pauc_mean'], 4),
                pauc_sd=fmt(summary['delta_low_fa_pauc_sd'], 4),
                latency=fmt(summary['latency_ratio_mean'], 3),
                latency_sd=fmt(summary['latency_ratio_sd'], 3),
            )
        )

    lines.extend([
        '',
        '## 按视频配对 Bootstrap 95% CI',
        '',
        '固定 seed `%d`、%d 次有放回序列重采样；每个 replicate 在候选/B1、三个训练 seed 及 Clean/Noise8 间使用同一组 16 个序列索引。区间是给定这三个训练 seed 条件下的视频抽样不确定性，不代表训练随机性的总体置信区间。'
        % (bootstrap_seed, bootstrap_replicates),
        '',
        '| Condition | Variant | Metric | Paired estimate | Bootstrap median | 95% percentile CI | Valid |',
        '|---|---|---|---:|---:|---:|---:|',
    ])
    for item in bootstrap_rows:
        interval = (
            '[%s, %s]'
            % (
                _format_bootstrap_metric(
                    item['metric'], item['ci95_low']
                ),
                _format_bootstrap_metric(
                    item['metric'], item['ci95_high']
                ),
            )
            if item['ci_available'] else 'NA'
        )
        lines.append(
            '| {condition} | {label} | `{metric}` | {estimate} | {median} | {interval} | {valid}/{total} ({rate:.1%}) |'.format(
                condition=item['condition'],
                label=item['variant_label'],
                metric=item['metric'],
                estimate=_format_bootstrap_metric(
                    item['metric'], item['point_estimate']
                ),
                median=_format_bootstrap_metric(
                    item['metric'], item['bootstrap_median']
                ),
                interval=interval,
                valid=item['valid_replicates'],
                total=item['replicates'],
                rate=item['valid_rate'],
            )
        )
    lines.extend([
        '',
        '所有指标只有在至少 95% bootstrap replicate 有效时才报告区间；否则区间显示 `NA`。相对 Fa 下降还要求基线 Fa 非零，此时应优先解释绝对 `ΔFa@fixedPd`。',
    ])

    lines.extend([
        '',
        '## 工作点与低 Fa 支持诊断',
        '',
        '| Condition | Variant | Seed | Pd@fixedFa selected threshold | Pd plateau [min,max] / n | Endpoint | Fa@fixedPd selected threshold | Fa plateau [min,max] / n | Endpoint | Internal unique Fa points | Min empirical Fa | Support |',
        '|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|',
    ])
    for row in sorted(
        rows, key=lambda item: (
            CONDITIONS.index(item['condition']),
            tuple(VARIANT_LABELS).index(item['variant']),
            item['seed'],
        )
    ):
        lines.append(
            '| {condition} | {label} | {seed} | {pd_threshold} | [{pd_min}, {pd_max}] / {pd_n} | {pd_endpoint} | {fa_threshold} | [{fa_min}, {fa_max}] / {fa_n} | {fa_endpoint} | {internal} | {minimum_fa} | `{support}` |'.format(
                condition=row['condition'],
                label=row['variant_label'],
                seed=row['seed'],
                pd_threshold=fmt(row['pd_fixed_selected_threshold'], 6),
                pd_min=fmt(row['pd_fixed_plateau_threshold_min'], 6),
                pd_max=fmt(row['pd_fixed_plateau_threshold_max'], 6),
                pd_n=row['pd_fixed_plateau_points'],
                pd_endpoint=('yes' if row['pd_fixed_selected_is_endpoint'] else 'no'),
                fa_threshold=fmt(row['fa_fixed_selected_threshold'], 6),
                fa_min=fmt(row['fa_fixed_plateau_threshold_min'], 6),
                fa_max=fmt(row['fa_fixed_plateau_threshold_max'], 6),
                fa_n=row['fa_fixed_plateau_points'],
                fa_endpoint=('yes' if row['fa_fixed_selected_is_endpoint'] else 'no'),
                internal=row['low_fa_internal_unique_points'],
                minimum_fa=fmt(row['low_fa_min_empirical_fa'], 8),
                support=row['low_fa_support'],
            )
        )
    lines.extend([
        '',
        '`origin-interpolation-only` 表示 `0 < Fa < cap` 没有任何经验曲线点；该 pAUC 仅由人为补入的原点与第一个经验点线性插值得到。尤其在 Noise8 出现此标记时，不能把数值解释为实测低虚警性能。',
        '工作点并列时，`Pd@fixedFa` 选择满足最优 Pd 的最低阈值，`Fa@fixedPd` 选择满足最优 Fa 的最高阈值；表中同时给出完整并列 plateau 范围及点数。',
    ])

    lines.extend(['', '## C2 继续门槛', ''])
    lines.append(
        '原计划写明“无单 seed 严重反向异常”但未给数值；本分析修正（非原始预注册）将其操作化为 Fa 相对下降不低于 -20%、Pd 差不低于 -3 pp。'
    )
    lines.append('')
    lines.extend(['- ' + detail for detail in gate_details])
    if status == 'PASS':
        lines.append('- 结论：C2 通过预注册工程门槛，可以设计 C3 动态门控。')
    elif status == 'FAIL':
        lines.append('- 结论：C2 未通过预注册门槛，不应在同一证据基础上直接堆叠 C3/C4。')
    else:
        lines.append('- 结论：结果未齐，不能判断是否进入 C3。')

    lines.extend([
        '',
        '## 解释边界',
        '',
        '- 三个训练 seed 只支持稳定性筛查，不支持“证明等效”或强显著性结论。',
        '- Noise8 是 Clean 权重的外部分布测试，不是第二套训练重复。',
        '- pAUC 使用本次加密阈值网格；完整 AUC 同时保留论文阈值协议。',
        '- Pixel F1/IoU 只用于定位训练退化，不替代目标级 Pd/Fa。',
        '- 本阶段没有使用官方 test，最终方法确定后才允许进行一次锁定测试。',
    ])
    if missing:
        lines.extend(['', '## 尚缺文件', ''])
        lines.extend(['- `%s`' % item for item in missing])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError('--bootstrap-replicates must be positive.')
    experiment_root = args.experiment_root.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    manifest = load_manifest(experiment_root / 'manifest.tsv')
    rows, missing, loaded = load_rows(
        experiment_root, manifest, args.low_fa_cap, log_root
    )
    summaries = aggregate_rows(rows)
    bootstrap_rows = paired_video_bootstrap(
        loaded,
        rows,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        low_fa_cap=args.low_fa_cap,
    )
    write_csv(experiment_root / 'results.csv', rows)
    write_csv(experiment_root / 'summary.csv', summaries)
    write_csv(
        experiment_root / 'bootstrap_summary.csv', bootstrap_rows
    )
    write_markdown(
        experiment_root / 'ANALYSIS.md',
        rows,
        summaries,
        bootstrap_rows,
        missing,
        args.low_fa_cap,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    print('Loaded %d/24 metric files; missing %d.' % (len(rows), len(missing)))
    print('Strict identity and curve validation passed for loaded metrics.')
    print(
        'Wrote %d paired-video bootstrap summary rows.'
        % len(bootstrap_rows)
    )
    print('Wrote %s' % (experiment_root / 'ANALYSIS.md'))


if __name__ == '__main__':
    main()
