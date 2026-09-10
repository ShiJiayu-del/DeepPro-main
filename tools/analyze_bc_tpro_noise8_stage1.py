#!/usr/bin/env python3
"""Validate and summarize the dedicated Noise8 BC-TPro stage-1 experiment.

This analyzer is intentionally separate from ``analyze_bc_tpro_stage1.py``:
the older experiment trains on Clean and evaluates Clean plus Noise8, whereas
this protocol trains and validates only on NUDT-MIRSDT-Noise8.0_FJY.

No checksum is required or generated.  Provenance is established from exact
run identities, normalized paths, semantic split contents, argparse logs,
checkpoint metadata, and recomputed integer evaluation counts.
"""

import argparse
import csv
import importlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_bc_tpro_stage1 as common  # noqa: E402


DATASET = 'NUDT-MIRSDT-Noise8.0_FJY'
MODEL = 'DeepPro-Plus_BCTPro'
SEEDS = (47, 49, 51)
SEED_GPU = {47: '0', 49: '1', 51: '2'}
VARIANTS = {
    'none': ('b1', 1, 'B1', 'B1 DeepPro-Plus'),
    'temporal_control': ('c0', 2, 'C0', 'C0 temporal control'),
    'center_multiscale': ('c1', 3, 'C1', 'C1 center multiscale'),
    'center_ring': ('c2', 4, 'C2', 'C2 center+ring'),
}
VAL_NAMES = (
    'Sequence9', 'Sequence13', 'Sequence14', 'Sequence16',
    'Sequence17', 'Sequence20', 'Sequence29', 'Sequence31',
    'Sequence45', 'Sequence49', 'Sequence55', 'Sequence61',
    'Sequence68', 'Sequence74', 'Sequence77', 'Sequence84',
)
METRICS_SUFFIX = '__noise8_internal_val.json'
CHECKPOINT_EPOCH = 32
SEQUENCE_LENGTH = 40
EVALUATION_WINDOWS = 48
MODERNIZED_PROFILE = 'modernized'
UPSTREAM_PROFILE = 'upstream8fa1a68_fp32'
PROFILES = {
    MODERNIZED_PROFILE: {
        'log_prefix': '',
        'train_amp': 1,
        'eval_amp': 1,
        'inference_amp': True,
        'swanlab_group': 'bc-tpro-stage1-noise8-scratch',
        'upstream_compat': None,
    },
    UPSTREAM_PROFILE: {
        'log_prefix': 'Upstream8fa1a68-FP32-',
        'train_amp': 0,
        'eval_amp': 0,
        'inference_amp': False,
        'swanlab_group': (
            'bc-tpro-stage1-noise8-upstream8fa1a68-fp32-scratch'
        ),
        'upstream_compat': 1,
    },
}
BOOTSTRAP_METRICS = (
    'delta_pd_at_0_5',
    'delta_fa_at_0_5',
    'fa_at_0_5_relative_reduction',
    'delta_auc',
)
OFFICIAL_METRIC_FIELDS = ('pd_at_0_5', 'fa_at_0_5', 'auc')
STATE_KEY_PATTERN = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*$'
)


def official_metric_contract():
    """Return the machine-readable TinaLRJ/DeepPro detection protocol."""
    return {
        'schema_version': 1,
        'source_repository': 'git@github.com:TinaLRJ/DeepPro.git',
        'source_commit': '8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28',
        'operating_threshold': 0.5,
        'reported_detection_metrics': [
            'Pd_at_threshold_0.5',
            'Fa_at_threshold_0.5',
            'Pd-Fa_AUC_27_thresholds',
        ],
        'auc_thresholds': common.PAPER_THRESHOLDS.tolist(),
    }


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--experiment-root', type=Path,
        default=repo_root / 'experiments' / 'bc_tpro_stage1_noise8_2026-09-09',
    )
    parser.add_argument(
        '--log-root', type=Path, default=repo_root / 'log' / 'sem_seg',
    )
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260909)
    parser.add_argument('--profile', choices=tuple(PROFILES), default=MODERNIZED_PROFILE)
    return parser.parse_args(argv)


def expected_manifest_rows(profile=MODERNIZED_PROFILE):
    settings = PROFILES[profile]
    rows = []
    for variant, (prefix, wave, label, _description) in VARIANTS.items():
        for seed in SEEDS:
            rows.append({
                'run_id': '%s_%s_seed%d' % (prefix, variant, seed),
                'wave': str(wave),
                'model': MODEL,
                'structure_variant': variant,
                'seed': str(seed),
                'gpu': SEED_GPU[seed],
                'log_dir': (
                    '2026-09-09/%s__%sSoftIoU-BCTPro-%s_seed%d_E32'
                    % (DATASET, settings['log_prefix'], label, seed)
                ),
            })
    return rows


def load_manifest(path, profile=MODERNIZED_PROFILE):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    expected = expected_manifest_rows(profile)
    if not rows:
        raise ValueError('Empty manifest: %s' % path)
    if list(rows[0]) != list(expected[0]):
        raise ValueError(
            'Manifest columns must be exactly: %s'
            % ', '.join(expected[0])
        )
    observed_by_id = {row.get('run_id'): row for row in rows}
    expected_by_id = {row['run_id']: row for row in expected}
    if len(observed_by_id) != len(rows):
        raise ValueError('Manifest run_id values must be unique.')
    if observed_by_id != expected_by_id:
        raise ValueError(
            'Manifest must exactly match the frozen 12-run Noise8 protocol.'
        )
    return rows


def _read_nonblank_lines(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing split file: %s' % path)
    raw = path.read_text(encoding='utf-8').splitlines()
    if any(not line.strip() for line in raw):
        raise ValueError('Split contains blank lines: %s' % path)
    return [line.strip() for line in raw]


def _dataset_split_records(path, expected_sequence_count):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing dataset split: %s' % path)
    ordered_names = []
    counts = Counter()
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped:
            raise ValueError('%s contains a blank record.' % path)
        name = stripped.split('/', 1)[0]
        if name not in counts:
            ordered_names.append(name)
        counts[name] += 1
    if len(ordered_names) != expected_sequence_count:
        raise ValueError(
            '%s must contain exactly %d ordered sequences, found %d.'
            % (path, expected_sequence_count, len(ordered_names))
        )
    if any(count != 100 for count in counts.values()):
        raise ValueError('%s must contain exactly 100 records per sequence.' % path)
    return ordered_names


def expected_internal_train_names(dataset_root):
    official_train = _dataset_split_records(dataset_root / 'train.txt', 80)
    official_test = _dataset_split_records(dataset_root / 'test.txt', 20)
    if set(official_train) & set(official_test):
        raise ValueError('Official Noise8 train/test sequence sets overlap.')
    if not set(VAL_NAMES).issubset(official_train):
        raise ValueError('Frozen internal validation set is not inside official train.')
    return [name for name in official_train if name not in VAL_NAMES], official_test


def validate_splits(experiment_root, dataset_root):
    train_path = experiment_root / 'splits' / 'train_sequences.txt'
    val_path = experiment_root / 'splits' / 'val_sequences.txt'
    train_names = _read_nonblank_lines(train_path)
    val_names = _read_nonblank_lines(val_path)
    expected_train, official_test = expected_internal_train_names(dataset_root)
    if train_names != expected_train:
        raise ValueError('Training split is not the frozen ordered 64-sequence set.')
    if val_names != list(VAL_NAMES):
        raise ValueError('Validation split is not the frozen ordered 16-sequence set.')
    if set(train_names) & set(val_names):
        raise ValueError('Training and validation splits overlap.')
    if len(train_names) != 64 or len(val_names) != 16:
        raise ValueError('The internal split must contain exactly 64 train/16 val sequences.')
    if set(train_names + val_names) & set(official_test):
        raise ValueError('Internal train/validation includes an official-test sequence.')
    if set(train_names + val_names) != set(expected_train + list(VAL_NAMES)):
        raise ValueError('The internal 64/16 split does not partition official train.')
    return train_path.resolve(), val_path.resolve(), val_names


def _require_namespace_value(context, namespace, field, expected):
    if field not in namespace:
        raise ValueError('%s: Namespace lacks %s.' % (context, field))
    common._require_typed_equal(
        context, 'Namespace %s' % field, namespace[field], expected,
    )


def validate_training_provenance(
    namespace, training_log, job, dataset_root, train_list, val_list, log_root,
    profile=MODERNIZED_PROFILE,
):
    context = '%s training' % job['run_id']
    settings = PROFILES[profile]
    expected = {
        'seed': int(job['seed']),
        'gpu': job['gpu'],
        'gpu_num': 1,
        'model': MODEL,
        'structure_variant': job['structure_variant'],
        'structure_bottleneck_channels': 8,
        'dataset': DATASET,
        'batch_size': 4,
        'gradient_accumulation_steps': 1,
        'epoch': CHECKPOINT_EPOCH,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'decay_rate': 0.0001,
        'step_size': 10,
        'lr_decay': 0.7,
        'seqlen': SEQUENCE_LENGTH,
        'patch_size': 128,
        'sample_rate': 0.1,
        'sequence_augmentation': 0,
        'loss': 'soft_iou',
        'threshold_eval': 0.5,
        'train_amp': settings['train_amp'],
        'eval_amp': settings['eval_amp'],
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
        'base_ckpt': '',
        'spatial_ckpt': '',
        'st_ckpt': '',
        'freeze_pretrained': 0,
        'use_swanlab': 1,
        'swanlab_project': 'DeepPro-BC-TPro',
        'swanlab_group': settings['swanlab_group'],
        'swanlab_mode': 'cloud',
        'swanlab_resume': 'never',
    }
    if settings['upstream_compat'] is not None:
        expected['upstream_compat'] = settings['upstream_compat']
    for field, expected_value in expected.items():
        _require_namespace_value(context, namespace, field, expected_value)
    common._require_path_equal(
        context, 'Namespace datapath', namespace.get('datapath'), dataset_root,
    )
    common._require_path_equal(
        context, 'Namespace train_sequence_list',
        namespace.get('train_sequence_list'), train_list,
    )
    common._require_path_equal(
        context, 'Namespace val_sequence_list',
        namespace.get('val_sequence_list'), val_list,
    )
    common._require_path_equal(
        context, 'Namespace savepath', namespace.get('savepath'), log_root.parent,
    )
    text = training_log.read_text(encoding='utf-8')
    random_marker = 'from random weights; no base checkpoint loaded.'
    if text.count(random_marker) != 1:
        raise ValueError(
            '%s: expected exactly one scratch-initialization marker.' % context
        )


def validate_model_state_dict(state_dict, variant, context):
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError('%s: checkpoint model_state_dict must be non-empty.' % context)
    if len(state_dict) < 100:
        raise ValueError(
            '%s: checkpoint model_state_dict is implausibly incomplete (%d keys).'
            % (context, len(state_dict))
        )
    invalid_keys = [
        key for key in state_dict
        if not isinstance(key, str) or not STATE_KEY_PATTERN.fullmatch(key)
    ]
    if invalid_keys:
        raise ValueError(
            '%s: checkpoint has malformed state key(s): %r'
            % (context, invalid_keys[:3])
        )
    if any(key.startswith('module.') for key in state_dict):
        raise ValueError('%s: checkpoint unexpectedly retains a DDP module prefix.' % context)
    for key, tensor in state_dict.items():
        if not common.torch.is_tensor(tensor) or tensor.numel() <= 0:
            raise ValueError('%s: state value %s is not a non-empty tensor.' % (context, key))
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            common.torch.isfinite(tensor).all().item()
        ):
            raise ValueError('%s: state tensor %s contains non-finite values.' % (context, key))

    required_shapes = {
        'conv_in.0.weight': (8, 1, 5, 7, 7),
        'TPro.conv.0.weight': (32, 32, 1, 1, 1),
        'conv_out2.weight': (1, 8, 1, 1, 1),
        'conv_out2.bias': (1,),
    }
    for key, expected_shape in required_shapes.items():
        if key not in state_dict:
            raise ValueError('%s: state_dict lacks required core key %s.' % (context, key))
        if tuple(state_dict[key].shape) != expected_shape:
            raise ValueError(
                '%s: state_dict %s shape must be %r, found %r.'
                % (context, key, expected_shape, tuple(state_dict[key].shape))
            )

    adapter_shapes = {
        'bc_tpro.evidence_fusion.0.weight': (
            8, 9 if variant == 'center_ring' else 3, 1, 1, 1
        ),
        'bc_tpro.evidence_fusion.0.bias': (8,),
        'bc_tpro.residual_projection.weight': (32, 8, 1, 1, 1),
        'bc_tpro.residual_projection.bias': (32,),
    }
    adapter_keys = {key for key in state_dict if key.startswith('bc_tpro.')}
    if variant == 'none':
        if adapter_keys:
            raise ValueError('%s: B1 state_dict unexpectedly contains adapter weights.' % context)
    else:
        if adapter_keys != set(adapter_shapes):
            raise ValueError(
                '%s: adapter state keys do not match variant %s.'
                % (context, variant)
            )
        for key, expected_shape in adapter_shapes.items():
            if tuple(state_dict[key].shape) != expected_shape:
                raise ValueError(
                    '%s: adapter state %s shape must be %r, found %r.'
                    % (context, key, expected_shape, tuple(state_dict[key].shape))
                )

    model_module = importlib.import_module(
        'networks.models.DeepPro-Plus_BCTPro'
    )
    reference_model = model_module.detector(
        1, 40, 40,
        structure_variant=variant,
        structure_bottleneck_channels=8,
        eval_chunk_rows=32,
    )
    reference_state = reference_model.state_dict()
    observed_keys = set(state_dict)
    expected_keys = set(reference_state)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            '%s: complete state key set mismatch; missing=%r unexpected=%r.'
            % (context, missing[:5], unexpected[:5])
        )
    for key, reference_tensor in reference_state.items():
        observed_shape = tuple(state_dict[key].shape)
        expected_shape = tuple(reference_tensor.shape)
        if observed_shape != expected_shape:
            raise ValueError(
                '%s: complete state shape mismatch for %s; expected %r, found %r.'
                % (context, key, expected_shape, observed_shape)
            )
    try:
        reference_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            '%s: strict CPU state_dict loading failed: %s' % (context, error)
        ) from error


def validate_evaluation_provenance(
    eval_log, metrics_path, job, dataset_root, val_list, log_root,
    profile=MODERNIZED_PROFILE,
):
    context = '%s evaluation' % job['run_id']
    settings = PROFILES[profile]
    records = common.parse_namespace_records(eval_log)
    resolved_metrics = metrics_path.resolve()
    matches = []
    for record in records:
        value = record['values'].get('metrics_json')
        if isinstance(value, str) and Path(value).expanduser().resolve() == resolved_metrics:
            matches.append(record['values'])
    if len(matches) != 1:
        raise ValueError(
            '%s: expected exactly one Namespace for metrics_json, found %d.'
            % (context, len(matches))
        )
    namespace = matches[0]
    expected = {
        'amp': settings['inference_amp'],
        'batch_size': 1,
        'dataset': DATASET,
        'epoch': CHECKPOINT_EPOCH,
        'eval_chunk_rows': 32,
        'gpu': job['gpu'],
        'log_dir': job['log_dir'],
        'output_only': False,
        'prefetch_factor': 1,
        'seqlen': SEQUENCE_LENGTH,
        'sequence_start': 0,
        'sequence_stop': None,
        'test_workers': 1,
        'threshold_eval': 0.5,
        'threshold_grid_step': 0.01,
    }
    for field, expected_value in expected.items():
        _require_namespace_value(context, namespace, field, expected_value)
    common._require_path_equal(
        context, 'Namespace datapath', namespace.get('datapath'), dataset_root,
    )
    common._require_path_equal(
        context, 'Namespace sequence_list', namespace.get('sequence_list'), val_list,
    )
    common._require_path_equal(
        context, 'Namespace logpath', namespace.get('logpath'), log_root.parent,
    )
    marker = 'Paper-aligned metrics saved to %s.' % resolved_metrics
    if eval_log.read_text(encoding='utf-8').count(marker) != 1:
        raise ValueError('%s: completed-evaluation marker is missing or duplicated.' % context)


def validate_one_payload(
    payload, job, metrics_path, experiment_root, log_root, dataset_root,
    train_list, val_list, val_names, checkpoint_cache, training_cache,
    profile=MODERNIZED_PROFILE,
):
    context = '%s metric' % job['run_id']
    settings = PROFILES[profile]
    common._require_equal(context, 'dataset', payload.get('dataset'), DATASET)
    common._require_equal(context, 'model', payload.get('model'), MODEL)
    common._require_int_equal(
        context, 'checkpoint_epoch', payload.get('checkpoint_epoch'), CHECKPOINT_EPOCH,
    )
    common._require_int_equal(
        context, 'sequence_length', payload.get('sequence_length'), SEQUENCE_LENGTH,
    )
    common._require_int_equal(
        context, 'sequence_count', payload.get('sequence_count'), len(VAL_NAMES),
    )
    common._require_int_equal(
        context, 'evaluation_windows', payload.get('evaluation_windows'),
        EVALUATION_WINDOWS,
    )
    common._require_equal(
        context, 'inference_amp', payload.get('inference_amp'),
        settings['inference_amp'],
    )
    curve = common.validate_curve_counts(payload, val_names, context)
    common.validate_payload_metrics(
        payload, curve, context,
        expected_inference_amp=settings['inference_amp'],
        validate_pixel_diagnostics=False,
    )

    run_dir = (log_root / job['log_dir']).resolve()
    expected_checkpoint = (
        run_dir / 'checkpoints' / 'epoch_32_model.pth'
    ).resolve()
    common._require_path_equal(
        context, 'checkpoint', payload.get('checkpoint'), expected_checkpoint,
    )
    if not expected_checkpoint.is_file():
        raise FileNotFoundError('Missing checkpoint: %s' % expected_checkpoint)
    key = str(expected_checkpoint)
    if key not in checkpoint_cache:
        checkpoint_cache[key] = common._load_checkpoint(
            expected_checkpoint, checkpoint_cache,
        )
    checkpoint = checkpoint_cache[key]
    if not isinstance(checkpoint, dict):
        raise ValueError('%s: checkpoint is not a mapping.' % context)
    common._require_equal(
        context, 'checkpoint model_name', checkpoint.get('model_name'), MODEL,
    )
    common._require_int_equal(
        context, 'checkpoint zero-based epoch', checkpoint.get('epoch'), 31,
    )
    model_config = checkpoint.get('model_config')
    if not isinstance(model_config, dict):
        raise ValueError('%s: checkpoint model_config is missing.' % context)
    common._require_equal(
        context, 'checkpoint structure_variant',
        model_config.get('structure_variant'), job['structure_variant'],
    )
    common._require_int_equal(
        context, 'checkpoint structure_bottleneck_channels',
        model_config.get('structure_bottleneck_channels'), 8,
    )
    common._require_int_equal(
        context, 'checkpoint eval_chunk_rows',
        model_config.get('eval_chunk_rows'), 32,
    )
    for field in ('spatial_ckpt', 'st_ckpt'):
        if field in model_config and model_config[field] not in (None, ''):
            raise ValueError('%s: checkpoint %s is not scratch-only.' % (context, field))
    if model_config.get('freeze_pretrained', False) not in (False, 0):
        raise ValueError('%s: checkpoint freeze_pretrained is enabled.' % context)
    validate_model_state_dict(
        checkpoint.get('model_state_dict'), job['structure_variant'], context,
    )

    training_log = run_dir / 'logs' / (MODEL + '.txt')
    training_key = str(training_log)
    if training_key not in training_cache:
        training_cache[training_key] = common.parse_training_namespace(training_log)
    validate_training_provenance(
        training_cache[training_key], training_log, job, dataset_root,
        train_list, val_list, log_root, profile,
    )
    validate_evaluation_provenance(
        run_dir / 'eval_epoch-32.txt', metrics_path, job, dataset_root,
        val_list, log_root, profile,
    )
    return curve


def _curve_totals(curve):
    true_counts = curve['true_counts'].sum(axis=0)
    target_counts = curve['target_counts'].sum(axis=0)
    false_counts = curve['false_counts'].sum(axis=0)
    pixels = float(curve['pixels'].sum())
    return (
        common.safe_divide(true_counts, target_counts),
        false_counts / pixels,
    )


def load_rows(experiment_root, log_root, profile=MODERNIZED_PROFILE):
    manifest = load_manifest(experiment_root / 'manifest.tsv', profile)
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root.parent / 'datasets' / DATASET).resolve()
    train_list, val_list, val_names = validate_splits(
        experiment_root, dataset_root,
    )
    metrics_root = experiment_root / 'metrics'
    expected_paths = {
        (metrics_root / (job['run_id'] + METRICS_SUFFIX)).resolve()
        for job in manifest
    }
    missing = sorted(str(path) for path in expected_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            'All 12 Noise8 metrics are required; missing:\n- ' + '\n- '.join(missing)
        )
    observed = {
        path.resolve() for path in metrics_root.glob('*' + METRICS_SUFFIX)
    }
    unexpected = sorted(str(path) for path in observed - expected_paths)
    if unexpected:
        raise ValueError('Unexpected Noise8 metric files:\n- ' + '\n- '.join(unexpected))

    loaded = {}
    checkpoint_cache = {}
    training_cache = {}
    reference_targets = None
    reference_pixels = None
    for job in manifest:
        path = (metrics_root / (job['run_id'] + METRICS_SUFFIX)).resolve()
        payload = json.loads(path.read_text(encoding='utf-8'))
        curve = validate_one_payload(
            payload, job, path, experiment_root, log_root, dataset_root,
            train_list, val_list, val_names, checkpoint_cache, training_cache,
            profile,
        )
        if reference_targets is None:
            reference_targets = curve['target_counts']
            reference_pixels = curve['pixels']
        elif (
            not np.array_equal(reference_targets, curve['target_counts'])
            or not np.array_equal(reference_pixels, curve['pixels'])
        ):
            raise ValueError(
                '%s: target/pixel denominators differ from another run.'
                % job['run_id']
            )
        pd_curve, fa_curve = _curve_totals(curve)
        loaded[(job['structure_variant'], int(job['seed']))] = {
            'job': job,
            'payload': payload,
            'curve': curve,
            'pd_curve': pd_curve,
            'fa_curve': fa_curve,
        }

    rows = []
    half = int(np.flatnonzero(common.EXPECTED_THRESHOLD_GRID == 0.5)[0])
    for variant in VARIANTS:
        for seed in SEEDS:
            entry = loaded[(variant, seed)]
            baseline = loaded[('none', seed)]
            baseline_fa = float(baseline['fa_curve'][half])
            baseline_pd = float(baseline['pd_curve'][half])
            payload = entry['payload']
            values = payload['all']
            row = {
                'run_id': entry['job']['run_id'],
                'variant': variant,
                'variant_label': VARIANTS[variant][3],
                'seed': seed,
                'pd_at_fixed_fa': common.max_pd_at_fa(
                    entry['pd_curve'], entry['fa_curve'], baseline_fa,
                ),
                'fa_at_fixed_pd': common.min_fa_at_pd(
                    entry['pd_curve'], entry['fa_curve'], baseline_pd,
                ),
                'auc': float(values['auc']),
                'pd_at_0_5': float(values['pd']),
                'fa_at_0_5': float(values['fa']),
                'elapsed_seconds': float(payload['elapsed_seconds']),
            }
            rows.append(row)
    index = {(row['variant'], row['seed']): row for row in rows}
    for row in rows:
        baseline = index[('none', row['seed'])]
        for field in (
            'pd_at_fixed_fa', 'fa_at_fixed_pd', 'auc', 'pd_at_0_5',
            'fa_at_0_5', 'elapsed_seconds',
        ):
            row['delta_' + field] = row[field] - baseline[field]
        base_fa = baseline['fa_at_fixed_pd']
        row['fa_at_fixed_pd_relative_reduction'] = (
            (base_fa - row['fa_at_fixed_pd']) / base_fa
            if base_fa > 0 and math.isfinite(row['fa_at_fixed_pd'])
            else float('nan')
        )
        base_fa_at_0_5 = baseline['fa_at_0_5']
        row['fa_at_0_5_relative_reduction'] = (
            (base_fa_at_0_5 - row['fa_at_0_5']) / base_fa_at_0_5
            if base_fa_at_0_5 > 0 else float('nan')
        )
        row['latency_ratio'] = (
            row['elapsed_seconds'] / baseline['elapsed_seconds']
            if baseline['elapsed_seconds'] > 0 else float('nan')
        )
    return rows, loaded


def aggregate_rows(rows):
    fields = (
        'pd_at_fixed_fa', 'fa_at_fixed_pd', 'auc', 'pd_at_0_5',
        'fa_at_0_5', 'elapsed_seconds',
        'delta_pd_at_fixed_fa', 'delta_fa_at_fixed_pd', 'delta_auc',
        'delta_pd_at_0_5', 'delta_fa_at_0_5',
        'fa_at_fixed_pd_relative_reduction',
        'fa_at_0_5_relative_reduction', 'latency_ratio',
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['variant']].append(row)
    result = []
    for variant in VARIANTS:
        group = grouped[variant]
        if sorted(row['seed'] for row in group) != list(SEEDS):
            raise ValueError('%s does not have exactly seeds 47/49/51.' % variant)
        item = {
            'variant': variant,
            'variant_label': VARIANTS[variant][3],
            'n': len(group),
        }
        for field in fields:
            values = [row[field] for row in group]
            item[field + '_mean'] = common.mean(values)
            item[field + '_sd'] = common.sample_sd(values)
        result.append(item)
    return result


def _bootstrap_auc(pd, fa):
    indices = np.asarray([
        int(np.flatnonzero(common.EXPECTED_THRESHOLD_GRID == value)[0])
        for value in common.PAPER_THRESHOLDS
    ], dtype=np.int64)
    return np.abs(np.trapz(pd[:, indices], fa[:, indices], axis=1))


def paired_video_bootstrap(loaded, rows, replicates, seed):
    weights = common._bootstrap_weights(replicates, len(VAL_NAMES), seed)
    row_index = {(row['variant'], row['seed']): row for row in rows}
    curves = {
        key: common._resampled_curves(entry['curve'], weights)
        for key, entry in loaded.items()
    }
    half = int(np.flatnonzero(common.EXPECTED_THRESHOLD_GRID == 0.5)[0])
    summaries = []
    for variant in VARIANTS:
        if variant == 'none':
            continue
        values = {metric: [] for metric in BOOTSTRAP_METRICS}
        for training_seed in SEEDS:
            base = curves[('none', training_seed)]
            candidate = curves[(variant, training_seed)]
            base_fa = base['fa'][:, half]
            candidate_fa = candidate['fa'][:, half]
            delta_fa = candidate_fa - base_fa
            relative = np.full(replicates, np.nan, dtype=np.float64)
            valid = np.isfinite(delta_fa) & np.isfinite(base_fa) & (base_fa > 0)
            relative[valid] = -delta_fa[valid] / base_fa[valid]
            values['delta_auc'].append(
                _bootstrap_auc(candidate['pd'], candidate['fa'])
                - _bootstrap_auc(base['pd'], base['fa'])
            )
            values['delta_pd_at_0_5'].append(
                candidate['pd'][:, half] - base['pd'][:, half]
            )
            values['delta_fa_at_0_5'].append(
                delta_fa
            )
            values['fa_at_0_5_relative_reduction'].append(relative)
        point_rows = [row_index[(variant, training_seed)] for training_seed in SEEDS]
        for metric in BOOTSTRAP_METRICS:
            replicate_means = common._mean_if_all_finite(values[metric])
            point = common.mean([row[metric] for row in point_rows])
            summaries.append(common._bootstrap_summary_row(
                variant, 'noise8_internal_val', metric, point,
                replicate_means, replicates,
            ))
    return summaries


def provisional_gate(rows):
    """Apply the amended official Pd/Fa/AUC-only C2 continuation gate."""
    if len(rows) != 12:
        return 'NOT_EVALUABLE', ['All 12 validated rows are required.']
    index = {(row['variant'], row['seed']): row for row in rows}
    c2 = [index[('center_ring', seed)] for seed in SEEDS]
    c1 = [index[('center_multiscale', seed)] for seed in SEEDS]
    required = [
        row[field] for row in c2
        for field in (
            'fa_at_0_5_relative_reduction',
            'delta_pd_at_0_5', 'delta_auc', 'latency_ratio',
        )
    ]
    if not all(math.isfinite(value) for value in required):
        return 'FAIL', ['A C2 gate metric is non-finite or unreachable.']
    mean_fa_reduction = common.mean([
        row['fa_at_0_5_relative_reduction'] for row in c2
    ])
    mean_pd_delta = common.mean([row['delta_pd_at_0_5'] for row in c2])
    mean_auc_delta = common.mean([row['delta_auc'] for row in c2])
    c2_c1_supported = all(
        c2_row['fa_at_0_5'] <= c1_row['fa_at_0_5']
        and c2_row['pd_at_0_5'] >= c1_row['pd_at_0_5']
        and c2_row['auc'] >= c1_row['auc']
        and (
            c2_row['fa_at_0_5'] < c1_row['fa_at_0_5']
            or c2_row['pd_at_0_5'] > c1_row['pd_at_0_5']
            or c2_row['auc'] > c1_row['auc']
        )
        for c2_row, c1_row in zip(c2, c1)
    )
    checks = [
        ('C2 mean Fa@0.5 relative reduction >=20%', mean_fa_reduction >= 0.20),
        ('C2 mean Pd@0.5 delta >=-1pp', mean_pd_delta >= -0.01),
        ('C2 mean 27-threshold AUC delta >=0', mean_auc_delta >= 0.0),
        ('Every seed Fa@0.5 reduction >=-20%', min(
            row['fa_at_0_5_relative_reduction'] for row in c2
        ) >= -0.20),
        ('Every seed Pd@0.5 delta >=-3pp', min(
            row['delta_pd_at_0_5'] for row in c2
        ) >= -0.03),
        ('Every seed latency ratio <=1.3', all(
            0 < row['latency_ratio'] <= 1.30 for row in c2
        )),
        ('C2 official-metric Pareto-improves C1 in every seed', c2_c1_supported),
    ]
    details = ['%s: %s' % (label, 'PASS' if passed else 'FAIL')
               for label, passed in checks]
    return ('PROVISIONAL_PASS' if all(passed for _, passed in checks)
            else 'PROVISIONAL_FAIL'), details


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summaries, bootstrap_rows, status, gate_details,
                   replicates, bootstrap_seed, profile=MODERNIZED_PROFILE):
    profile_note = (
        'upstream commit 8fa1a68 training-data semantics with FP32'
        if profile == UPSTREAM_PROFILE else
        'modernized deterministic AMP protocol'
    )
    lines = [
        '# BC-TPro Noise8 专训 Stage1 分析', '',
        '协议身份：`%s`（%s）。' % (profile, profile_note), '',
        '验证状态：12/12 run 已通过语义身份、scratch、固定划分、checkpoint、评测日志与原始整数计数重算。',
        '',
        '官方指标继续门槛：**%s**。只使用阈值 0.5 的 Pd/Fa 与论文 27 阈值 Pd-Fa AUC；raw-logit 仅为补充敏感性分析。'
        % status, '',
        '## 三随机种子 mean ± sample-SD', '',
        '| Variant | Pd@0.5 (%) | Fa@0.5 (×1e-5) | 27-threshold Pd-Fa AUC | Latency (s) |',
        '|---|---:|---:|---:|---:|',
    ]
    for item in summaries:
        lines.append(
            '| {label} | {pd05:.3f} ± {pd05_sd:.3f} | '
            '{fa05:.3f} ± {fa05_sd:.3f} | '
            '{auc:.6f} ± {auc_sd:.6f} | {lat:.3f} ± {lat_sd:.3f} |'.format(
                label=item['variant_label'],
                auc=item['auc_mean'], auc_sd=item['auc_sd'],
                pd05=item['pd_at_0_5_mean'] * 100,
                pd05_sd=item['pd_at_0_5_sd'] * 100,
                fa05=item['fa_at_0_5_mean'] * 1e5,
                fa05_sd=item['fa_at_0_5_sd'] * 1e5,
                lat=item['elapsed_seconds_mean'],
                lat_sd=item['elapsed_seconds_sd'],
            )
        )
    lines.extend([
        '', '## Noise8 专训继续门槛', '',
        '该门槛不声称来自原 `EXPERIMENT_PLAN.md`；其权威来源必须是候选结果产生前另行冻结的协议修订。',
        '',
    ])
    lines.extend('- ' + detail for detail in gate_details)
    lines.extend([
        '', '## 配对视频 Bootstrap 95% CI', '',
        '- 重采样单位：固定 16 个验证视频；候选与同 seed B1、三个训练 seed 共用同一组抽样权重。',
        '- 重复：%d；随机种子：%d；区间仅反映给定三个训练 seed 条件下的视频抽样不确定性。'
        % (replicates, bootstrap_seed),
        '- latency 不是视频级可加统计量，因此只报告三 seed mean ± SD，不伪造视频 bootstrap CI。',
        '',
        '| Variant | Metric | Paired estimate | 95% percentile CI | Valid |',
        '|---|---|---:|---:|---:|',
    ])
    for item in bootstrap_rows:
        interval = (
            '[%.8g, %.8g]' % (item['ci95_low'], item['ci95_high'])
            if item['ci_available'] else 'NA'
        )
        lines.append(
            '| %s | `%s` | %.8g | %s | %d/%d |'
            % (
                item['variant_label'], item['metric'], item['point_estimate'],
                interval, item['valid_replicates'], item['replicates'],
            )
        )
    lines.extend([
        '', '## 解释边界', '',
        '- 这是 Noise8 专门训练的 in-domain internal-validation 实验，不是 Clean 权重的 zero-shot Noise8 测试。',
        '- 主结果严格使用论文的 27 个 sigmoid 概率阈值 AUC，以及 sigmoid 阈值 0.5 的 Pd/Fa。',
        '- 109 点网格与 raw-logit 工作点只可作为敏感性补充，不参与继续门槛、候选资格、排序或锁定。',
        '- 三个训练 seed 支持稳定性筛查，不支持等效性证明；bootstrap CI 也不覆盖训练 seed 总体不确定性。',
        '- 本分析器不要求或生成 SHA256；身份来自规范化路径、精确 split 内容、run manifest、日志、checkpoint metadata 和逐序列整数计数。',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError('--bootstrap-replicates must be positive.')
    experiment_root = args.experiment_root.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    rows, loaded = load_rows(experiment_root, log_root, args.profile)
    summaries = aggregate_rows(rows)
    bootstrap_rows = paired_video_bootstrap(
        loaded, rows, args.bootstrap_replicates, args.bootstrap_seed,
    )
    status, details = provisional_gate(rows)
    _write_csv(experiment_root / 'noise8_results.csv', rows)
    _write_csv(experiment_root / 'noise8_summary.csv', summaries)
    _write_csv(experiment_root / 'noise8_bootstrap_summary.csv', bootstrap_rows)
    write_markdown(
        experiment_root / 'NOISE8_ANALYSIS.md', summaries, bootstrap_rows,
        status, details, args.bootstrap_replicates, args.bootstrap_seed,
        args.profile,
    )
    print('Validated 12/12 dedicated Noise8 runs.')
    print('Official Pd/Fa/AUC gate: %s.' % status)
    print('Wrote %s' % (experiment_root / 'NOISE8_ANALYSIS.md'))


if __name__ == '__main__':
    main()
