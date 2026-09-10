#!/usr/bin/env python3
"""Fail-closed validation for locked Noise8 80/20 paper experiments.

The ``plan`` command reads only split text and Stage-1 evidence.  It never
opens an official-test image.  The ``checkpoint`` command validates a finished
epoch-32 training artifact before the separate test phase is allowed to start.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_bc_tpro_noise8_paper as selector  # noqa: E402
from tools import analyze_bc_tpro_noise8_stage1 as stage1  # noqa: E402
from tools import analyze_bc_tpro_stage1 as common  # noqa: E402


DATASET = 'NUDT-MIRSDT-Noise8.0_FJY'
MODEL = 'DeepPro-Plus_BCTPro'
PROFILE = stage1.UPSTREAM_PROFILE
EXPECTED_DATA_ROOT = (
    REPO_ROOT.parent / 'datasets' / DATASET
).resolve()
EXPECTED_STAGE1_ROOT = (
    REPO_ROOT / 'experiments'
    / 'bc_tpro_stage1_noise8_upstream_2026-09-09'
).resolve()
EXPECTED_PROTOCOL = (
    EXPECTED_STAGE1_ROOT / 'OFFICIAL_METRIC_AMENDMENT_2026-09-10.md'
).resolve()
EXPECTED_FINAL_ROOT = (
    REPO_ROOT / 'experiments' / 'bc_tpro_final_noise8_2026-09-09'
).resolve()
EXPECTED_LOG_ROOT = (REPO_ROOT / 'log' / 'sem_seg').resolve()
FINAL_PROTOCOL_LOCK_NAME = 'FINAL_PROTOCOL_LOCK.json'
FINAL_MANIFEST_NAME = 'final_manifest.tsv'
EXPECTED_TEST_NAMES = (
    'Sequence85', 'Sequence86', 'Sequence87', 'Sequence88',
    'Sequence89', 'Sequence90', 'Sequence91', 'Sequence92',
    'Sequence93', 'Sequence94', 'Sequence95', 'Sequence96',
    'Sequence97', 'Sequence47', 'Sequence56', 'Sequence59',
    'Sequence76', 'Sequence101', 'Sequence105', 'Sequence119',
)
EXPECTED_TRAIN_NAMES = tuple(
    'Sequence%d' % index
    for index in range(1, 85)
    if index not in {47, 56, 59, 76}
)
SEED_GPU = {47: '0', 49: '1', 51: '2'}
VARIANT_CODES = {
    'none': 'B1',
    'temporal_control': 'C0',
    'center_multiscale': 'C1',
    'center_ring': 'C2',
}
LOCK_KEYS = {
    'schema_version', 'status', 'selection_complete', 'protocol_path',
    'locked_model', 'model', 'selected_variant', 'baseline', 'seeds',
    'eligible_candidates', 'candidate_qualification', 'selection_rule',
    'selector_inputs', 'selection_uses_single_seed',
    'detection_metric_contract', 'official_test_accessed',
}
FINAL_GROUP = (
    'bc-tpro-final80-noise8-upstream8fa1a68-fp32-scratch-locked'
)
FINAL_PROTOCOL_KEYS = {
    'schema_version', 'status', 'profile', 'protocol_path',
    'detection_metric_contract', 'dataset', 'data_root', 'stage1_root',
    'log_root', 'final_root', 'candidate_lock', 'jobs',
    'official_train_sequences', 'official_test_sequences',
    'official_test_accessed_at_lock',
}


def fail(message):
    raise ValueError(message)


def _relative_parts(raw, source):
    normalized = raw.strip().replace('\\', '/')
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or '..' in path.parts:
        fail('Unsafe or empty list entry in %s: %r' % (source, raw))
    parts = tuple(part for part in path.parts if part not in {'', '.'})
    if len(parts) < 2:
        fail('%s entries must identify a sequence and frame: %r'
             % (source, raw))
    return parts


def read_official_sequence_list(path, entry_count, sequence_count):
    """Read split metadata without resolving or opening referenced images."""
    path = Path(path).expanduser().resolve(strict=True)
    raw_lines = path.read_text(encoding='utf-8').splitlines()
    if any(not line.strip() for line in raw_lines):
        fail('%s must not contain blank split entries.' % path)
    lines = [line.strip() for line in raw_lines]
    if len(lines) != entry_count:
        fail('%s must contain %d entries, found %d.'
             % (path, entry_count, len(lines)))
    parsed = [_relative_parts(line, path) for line in lines]
    names = [parts[0] for parts in parsed]
    counts = Counter(names)
    ordered_names = list(dict.fromkeys(names))
    if len(counts) != sequence_count:
        fail('%s must identify %d sequences, found %d.'
             % (path, sequence_count, len(counts)))
    if set(counts.values()) != {100}:
        fail('%s must contain exactly 100 frames per sequence.' % path)
    expected_entries = [
        (name, 'Mix', '%05d.mat' % frame)
        for name in ordered_names
        for frame in range(1, 101)
    ]
    if parsed != expected_entries:
        fail(
            '%s must contain each unique Sequence/Mix/00001..00100.mat '
            'entry exactly once and in sequence/frame order.' % path
        )
    return ordered_names


def _directory_filenames(path, context):
    path = Path(path)
    if not path.is_dir():
        fail('%s directory is missing: %s.' % (context, path))
    return sorted(entry.name for entry in path.iterdir() if entry.is_file())


def validate_sequence_frame_layout(data_root, sequence_names, centroids=False):
    """Validate frame names and pairings without decoding image content."""
    data_root = Path(data_root).expanduser().resolve(strict=True)
    expected = ['%05d.png' % frame for frame in range(1, 101)]
    for name in sequence_names:
        image_names = _directory_filenames(
            data_root / name / 'images', name + ' images',
        )
        mask_names = _directory_filenames(
            data_root / name / 'masks', name + ' masks',
        )
        if image_names != expected or mask_names != expected:
            fail(
                '%s images and masks must both be the exact unique '
                '00001.png..00100.png frame set.' % name
            )
        if centroids:
            centroid_root = data_root / name / 'masks_centroid'
            if not centroid_root.is_dir():
                centroid_root = (
                    data_root.parent / 'NUDT-MIRSDT' / name
                    / 'masks_centroid'
                )
            centroid_names = _directory_filenames(
                centroid_root, name + ' centroid masks',
            )
            if centroid_names != expected:
                fail(
                    '%s centroid masks must be the exact unique '
                    '00001.png..00100.png frame set.' % name
                )


def validate_official_split_metadata(data_root):
    data_root = Path(data_root).expanduser().resolve(strict=True)
    if data_root != EXPECTED_DATA_ROOT:
        fail('Noise8 data root mismatch: %s != %s.'
             % (data_root, EXPECTED_DATA_ROOT))
    train_names = read_official_sequence_list(
        data_root / 'train.txt', 8000, 80,
    )
    test_names = read_official_sequence_list(
        data_root / 'test.txt', 2000, 20,
    )
    if set(train_names) & set(test_names):
        fail('Official train and test sequence sets overlap.')
    if tuple(train_names) != EXPECTED_TRAIN_NAMES:
        fail('Official train sequence order differs from the registered split.')
    if tuple(test_names) != EXPECTED_TEST_NAMES:
        fail('Official test sequence order differs from the registered paper split.')
    validate_sequence_frame_layout(data_root, train_names)
    validate_sequence_frame_layout(data_root, test_names, centroids=True)
    return train_names, test_names


def _strict_object(value, keys, context):
    if not isinstance(value, dict):
        fail('%s must be an object.' % context)
    observed = set(value)
    if observed != set(keys):
        fail('%s fields mismatch; missing=%r unexpected=%r.' % (
            context, sorted(set(keys) - observed),
            sorted(observed - set(keys)),
        ))


def require_profile(profile):
    if profile != PROFILE:
        fail(
            'Final protocol requires the explicit upstream profile %s; got %r.'
            % (PROFILE, profile)
        )
    return profile


def validate_lock_schema(payload, stage1_root, log_root, profile):
    """Validate the selector artifact before expensive evidence replay."""
    require_profile(profile)
    _strict_object(payload, LOCK_KEYS, 'candidate lock')
    stage1_root = Path(stage1_root).resolve()
    log_root = Path(log_root).resolve()
    if payload['schema_version'] != 2:
        fail('candidate lock schema_version must be 2.')
    if payload['status'] not in {'CANDIDATE_LOCKED', 'B1_FALLBACK'}:
        fail('candidate lock has no executable selection status.')
    if payload['selection_complete'] is not True:
        fail('candidate selection is incomplete.')
    expected_protocol = (
        stage1_root / 'OFFICIAL_METRIC_AMENDMENT_2026-09-10.md'
    ).resolve()
    if Path(payload['protocol_path']).expanduser().resolve() != expected_protocol:
        fail('candidate lock protocol_path mismatch.')
    if payload['model'] != MODEL:
        fail('candidate lock model mismatch.')
    selected = payload['selected_variant']
    if selected not in VARIANT_CODES:
        fail('selected variant is not supported by the frozen final launcher.')
    if payload['seeds'] != [47, 49, 51]:
        fail('candidate lock seeds must be exactly [47, 49, 51].')
    if payload['selection_uses_single_seed'] is not False:
        fail('single-seed selection is forbidden.')
    if payload['detection_metric_contract'] != stage1.official_metric_contract():
        fail('candidate lock detection metric contract mismatch.')
    if payload['official_test_accessed'] is not False:
        fail('candidate lock must predate official-test access.')

    _strict_object(
        payload['baseline'], {'role', 'model', 'selected_variant'},
        'candidate lock baseline',
    )
    if payload['baseline'] != {
        'role': 'B1', 'model': MODEL, 'selected_variant': 'none',
    }:
        fail('candidate lock baseline identity mismatch.')
    _strict_object(
        payload['locked_model'], {'role', 'model', 'selected_variant'},
        'candidate lock locked_model',
    )
    expected_role = 'B1' if selected == 'none' else 'candidate'
    if payload['locked_model'] != {
        'role': expected_role, 'model': MODEL,
        'selected_variant': selected,
    }:
        fail('candidate lock selected identity mismatch.')
    if payload['status'] == 'B1_FALLBACK' and selected != 'none':
        fail('B1_FALLBACK must select the none variant.')
    if payload['status'] == 'CANDIDATE_LOCKED' and selected == 'none':
        fail('CANDIDATE_LOCKED must select a non-B1 variant.')

    expected_inputs = [
        {'role': role, 'path': str(path.resolve())}
        for role, path in selector.required_artifacts(
            stage1_root, log_root, profile,
        )
    ]
    if payload['selector_inputs'] != expected_inputs:
        fail('candidate lock selector_inputs differ from the complete evidence set.')
    return selected


def replay_and_validate_lock(payload, stage1_root, log_root, profile):
    """Recompute the selector output and require an exact machine-level match."""
    require_profile(profile)
    selected = validate_lock_schema(payload, stage1_root, log_root, profile)
    probability_rows, probability_loaded = (
        selector.load_complete_evidence(stage1_root, log_root, profile)
    )
    selector.require_c3_gate_resolved(probability_rows)
    parameters = {
        entry['job']['run_id']: float(entry['payload']['parameters_m'])
        for entry in probability_loaded.values()
    }
    decisions = selector.evaluate_candidate_eligibility(
        probability_rows, parameters, identity_verified=True,
        profile=profile,
    )
    selection = selector.rank_qualified_candidates(decisions)
    if selection['status'] == 'UNRESOLVED_TRADEOFF':
        fail('Candidate metrics remain a Pareto trade-off; final training is forbidden.')
    expected = selector.build_locked_candidate_payload(
        selection, decisions, stage1_root, log_root, profile,
    )
    if payload != expected:
        fail('candidate lock differs from a fresh replay of the frozen selector.')
    return selected


def build_final_jobs(selected_variant, profile):
    require_profile(profile)
    if selected_variant not in VARIANT_CODES:
        fail('Unsupported selected variant: %s.' % selected_variant)
    variants = ['none']
    if selected_variant != 'none':
        variants.append(selected_variant)
    jobs = []
    for wave, variant in enumerate(variants, start=1):
        code = VARIANT_CODES[variant]
        for seed, gpu in SEED_GPU.items():
            run_id = 'final80_upstream_%s_%s_seed%d' % (
                code.lower(), variant, seed,
            )
            jobs.append({
                'run_id': run_id,
                'wave': wave,
                'role': 'B1' if variant == 'none' else 'candidate',
                'code': code,
                'model': MODEL,
                'structure_variant': variant,
                'seed': seed,
                'gpu': gpu,
                'log_dir': (
                    '2026-09-10/%s__Upstream8fa1a68-FP32-Final80-%s_seed%d_E32'
                    % (DATASET, code, seed)
                ),
            })
    return jobs


def jobs_tsv(jobs):
    fields = (
        'run_id', 'wave', 'role', 'code', 'model',
        'structure_variant', 'seed', 'gpu', 'log_dir',
    )
    lines = ['\t'.join(fields)]
    lines.extend(
        '\t'.join(str(job[field]) for field in fields)
        for job in jobs
    )
    return '\n'.join(lines) + '\n'


def _publish_text_once(path, text):
    """Publish immutable text atomically, or verify an identical prior copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding='utf-8') != text:
            fail('Immutable protocol artifact differs from expected: %s.' % path)
        return False

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=str(path.parent),
            prefix='.' + path.name + '.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            pass
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    if not path.is_file() or path.read_text(encoding='utf-8') != text:
        fail('Immutable protocol artifact raced or is incomplete: %s.' % path)
    return True


def build_final_protocol_lock(
    candidate_lock, jobs, data_root, stage1_root, log_root, final_root,
    profile,
):
    require_profile(profile)
    data_root = Path(data_root).expanduser().resolve()
    stage1_root = Path(stage1_root).expanduser().resolve()
    log_root = Path(log_root).expanduser().resolve()
    final_root = Path(final_root).expanduser().resolve()
    if data_root != EXPECTED_DATA_ROOT:
        fail('Final protocol lock data root mismatch.')
    if stage1_root != EXPECTED_STAGE1_ROOT:
        fail('Final protocol lock Stage-1 root mismatch.')
    if log_root != EXPECTED_LOG_ROOT:
        fail('Final protocol lock log root mismatch.')
    if final_root != EXPECTED_FINAL_ROOT:
        fail('Final protocol lock experiment root mismatch.')
    return {
        'schema_version': 2,
        'status': 'FINAL_PROTOCOL_LOCKED',
        'profile': profile,
        'protocol_path': str(
            (
                stage1_root / 'OFFICIAL_METRIC_AMENDMENT_2026-09-10.md'
            ).resolve()
        ),
        'detection_metric_contract': stage1.official_metric_contract(),
        'dataset': DATASET,
        'data_root': str(data_root),
        'stage1_root': str(stage1_root),
        'log_root': str(log_root),
        'final_root': str(final_root),
        'candidate_lock': candidate_lock,
        'jobs': jobs,
        'official_train_sequences': list(EXPECTED_TRAIN_NAMES),
        'official_test_sequences': list(EXPECTED_TEST_NAMES),
        'official_test_accessed_at_lock': False,
    }


def validate_final_protocol_lock(payload, expected):
    _strict_object(payload, FINAL_PROTOCOL_KEYS, 'final protocol lock')
    if payload != expected:
        fail(
            'Immutable final protocol lock differs from the freshly replayed '
            'candidate, split, or job plan.'
        )


def _final_protocol_paths(final_root):
    final_root = Path(final_root).expanduser().resolve()
    if final_root != EXPECTED_FINAL_ROOT:
        fail('Final experiment root mismatch: %s.' % final_root)
    return (
        final_root / FINAL_PROTOCOL_LOCK_NAME,
        final_root / FINAL_MANIFEST_NAME,
    )


def freeze_final_protocol(
    lock_path, stage1_root, log_root, data_root, final_root, profile,
):
    candidate_lock, jobs = load_and_validate_plan(
        lock_path, stage1_root, log_root, data_root, profile,
    )
    protocol_path, manifest_path = _final_protocol_paths(final_root)
    expected = build_final_protocol_lock(
        candidate_lock, jobs, data_root, stage1_root, log_root, final_root,
        profile,
    )
    status_root = (
        EXPECTED_LOG_ROOT / '_queues'
        / 'bc_tpro_final_noise8_2026-09-09' / 'status'
    )
    if not protocol_path.exists() and status_root.is_dir():
        receipts = sorted(path for path in status_root.iterdir() if path.is_file())
        if receipts:
            fail(
                'Final receipts exist without an immutable protocol lock; '
                'refusing to adopt a plan retroactively.'
            )
    serialized = json.dumps(
        expected, indent=2, sort_keys=True, allow_nan=False,
    ) + '\n'
    _publish_text_once(protocol_path, serialized)
    _publish_text_once(manifest_path, jobs_tsv(jobs))
    observed = json.loads(protocol_path.read_text(encoding='utf-8'))
    validate_final_protocol_lock(observed, expected)
    return candidate_lock, jobs


def load_and_validate_frozen_plan(
    lock_path, stage1_root, log_root, data_root, final_root, profile,
):
    candidate_lock, jobs = load_and_validate_plan(
        lock_path, stage1_root, log_root, data_root, profile,
    )
    protocol_path, manifest_path = _final_protocol_paths(final_root)
    observed = json.loads(protocol_path.read_text(encoding='utf-8'))
    expected = build_final_protocol_lock(
        candidate_lock, jobs, data_root, stage1_root, log_root, final_root,
        profile,
    )
    validate_final_protocol_lock(observed, expected)
    if manifest_path.read_text(encoding='utf-8') != jobs_tsv(jobs):
        fail('Immutable final manifest differs from the frozen job plan.')
    return candidate_lock, jobs


def validate_training_checkpoint(checkpoint_path, training_log, job, log_root):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    training_log = Path(training_log).expanduser().resolve(strict=True)
    log_root = Path(log_root).expanduser().resolve(strict=True)
    expected_checkpoint = (
        log_root / job['log_dir'] / 'checkpoints' / 'epoch_32_model.pth'
    ).resolve()
    if checkpoint_path != expected_checkpoint:
        fail('Final checkpoint path mismatch.')
    expected_log = (
        log_root / job['log_dir'] / 'logs' / (MODEL + '.txt')
    ).resolve()
    if training_log != expected_log:
        fail('Final training log path mismatch.')
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location='cpu', weights_only=True,
        )
    except Exception as error:
        fail('Cannot safely load final checkpoint: %s' % error)
    if not isinstance(checkpoint, dict):
        fail('Final checkpoint must be a mapping.')
    if checkpoint.get('epoch') != 31:
        fail('Final checkpoint must be fixed epoch 32 (stored epoch 31).')
    if checkpoint.get('model_name') != MODEL:
        fail('Final checkpoint model_name mismatch.')
    expected_config = {
        'eval_chunk_rows': 32,
        'structure_variant': job['structure_variant'],
        'structure_bottleneck_channels': 8,
    }
    if checkpoint.get('model_config') != expected_config:
        fail('Final checkpoint model_config mismatch.')
    stage1.validate_model_state_dict(
        checkpoint.get('model_state_dict'), job['structure_variant'],
        job['run_id'] + ' final checkpoint',
    )
    if not isinstance(checkpoint.get('optimizer_state_dict'), dict):
        fail('Final checkpoint lacks optimizer state.')

    namespace = common.parse_training_namespace(training_log)
    expected_values = {
        'seed': job['seed'],
        'gpu': job['gpu'],
        'gpu_num': 1,
        'model': MODEL,
        'structure_variant': job['structure_variant'],
        'structure_bottleneck_channels': 8,
        'dataset': DATASET,
        'batch_size': 4,
        'gradient_accumulation_steps': 1,
        'epoch': 32,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'decay_rate': 0.0001,
        'step_size': 10,
        'lr_decay': 0.7,
        'seqlen': 40,
        'patch_size': 128,
        'sample_rate': 0.1,
        'sequence_augmentation': 0,
        'loss': 'soft_iou',
        'threshold_eval': 0.5,
        'train_amp': 0,
        'eval_amp': 0,
        'upstream_compat': 1,
        'eval_chunk_rows': 32,
        'eval_interval': 8,
        'skip_inprocess_validation': 1,
        'early_stopping_patience': 0,
        'train_workers': 4,
        'val_workers': 0,
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
        'swanlab_group': FINAL_GROUP,
        'swanlab_mode': 'cloud',
        'swanlab_resume': 'never',
        'train_sequence_list': None,
        'val_sequence_list': None,
    }
    for field, expected in expected_values.items():
        if field not in namespace:
            fail('%s training Namespace lacks %s.' % (job['run_id'], field))
        common._require_typed_equal(
            job['run_id'], 'training Namespace ' + field,
            namespace[field], expected,
        )
    common._require_path_equal(
        job['run_id'], 'training Namespace datapath',
        namespace.get('datapath'), EXPECTED_DATA_ROOT,
    )
    common._require_path_equal(
        job['run_id'], 'training Namespace savepath',
        namespace.get('savepath'), log_root.parent,
    )
    text = training_log.read_text(encoding='utf-8')
    if text.count('from random weights; no base checkpoint loaded.') != 1:
        fail('%s lacks one unambiguous random-initialization marker.'
             % job['run_id'])
    if 'Resumed checkpoint ' in text:
        fail('%s unexpectedly resumed a checkpoint.' % job['run_id'])
    if '---- EPOCH ' in text and ' EVALUATION ----' in text:
        fail('%s performed in-process validation.' % job['run_id'])
    if text.count('Saved fixed-final-epoch checkpoint at ') != 1:
        fail('%s lacks exactly one fixed-epoch completion marker.'
             % job['run_id'])
    return checkpoint


def load_and_validate_plan(
    lock_path, stage1_root, log_root, data_root, profile,
):
    require_profile(profile)
    stage1_root = Path(stage1_root).expanduser().resolve(strict=True)
    log_root = Path(log_root).expanduser().resolve()
    if stage1_root != EXPECTED_STAGE1_ROOT:
        fail('Stage-1 root mismatch: %s.' % stage1_root)
    if not EXPECTED_PROTOCOL.is_file():
        fail('Official metric amendment is missing: %s.' % EXPECTED_PROTOCOL)
    lock_path = Path(lock_path).expanduser().resolve(strict=True)
    if lock_path != (stage1_root / 'LOCKED_CANDIDATE.json').resolve():
        fail('Only the canonical LOCKED_CANDIDATE.json may authorize final training.')
    validate_official_split_metadata(data_root)
    payload = json.loads(lock_path.read_text(encoding='utf-8'))
    selected = replay_and_validate_lock(
        payload, stage1_root, log_root, profile,
    )
    return payload, build_final_jobs(selected, profile)


def _job_from_args(args):
    job = {
        'run_id': args.run_id,
        'wave': args.wave,
        'role': args.role,
        'code': args.code,
        'model': args.model,
        'structure_variant': args.variant,
        'seed': args.seed,
        'gpu': args.gpu,
        'log_dir': args.log_dir,
    }
    expected = {
        candidate['run_id']: candidate
        for candidate in build_final_jobs(args.selected_variant, args.profile)
    }
    if job != expected.get(args.run_id):
        fail('Checkpoint job identity is not in the locked final plan.')
    return job


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'freeze', 'verify-frozen'):
        command = subparsers.add_parser(name)
        command.add_argument('--data-root', type=Path, required=True)
        command.add_argument('--stage1-root', type=Path, required=True)
        command.add_argument('--log-root', type=Path, required=True)
        command.add_argument('--lock', type=Path, required=True)
        command.add_argument('--profile', choices=(PROFILE,), required=True)
        command.add_argument('--emit-tsv', action='store_true')
        if name != 'plan':
            command.add_argument('--final-root', type=Path, required=True)

    checkpoint = subparsers.add_parser('checkpoint')
    checkpoint.add_argument('--checkpoint', type=Path, required=True)
    checkpoint.add_argument('--training-log', type=Path, required=True)
    checkpoint.add_argument('--log-root', type=Path, required=True)
    checkpoint.add_argument('--selected-variant', required=True)
    checkpoint.add_argument('--profile', choices=(PROFILE,), required=True)
    checkpoint.add_argument('--run-id', required=True)
    checkpoint.add_argument('--wave', type=int, required=True)
    checkpoint.add_argument('--role', required=True)
    checkpoint.add_argument('--code', required=True)
    checkpoint.add_argument('--model', required=True)
    checkpoint.add_argument('--variant', required=True)
    checkpoint.add_argument('--seed', type=int, required=True)
    checkpoint.add_argument('--gpu', required=True)
    checkpoint.add_argument('--log-dir', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command in {'plan', 'freeze', 'verify-frozen'}:
        if args.command == 'plan':
            payload, jobs = load_and_validate_plan(
                args.lock, args.stage1_root, args.log_root, args.data_root,
                args.profile,
            )
        elif args.command == 'freeze':
            payload, jobs = freeze_final_protocol(
                args.lock, args.stage1_root, args.log_root, args.data_root,
                args.final_root, args.profile,
            )
        else:
            payload, jobs = load_and_validate_frozen_plan(
                args.lock, args.stage1_root, args.log_root, args.data_root,
                args.final_root, args.profile,
            )
        if args.emit_tsv:
            print('\n'.join(jobs_tsv(jobs).splitlines()[1:]))
        else:
            print(
                'VALID %s final plan: status=%s selected=%s runs=%d '
                'official_train=80 official_test=20'
                % (
                    args.command, payload['status'],
                    payload['selected_variant'], len(jobs),
                )
            )
        return

    job = _job_from_args(args)
    validate_training_checkpoint(
        args.checkpoint, args.training_log, job, args.log_root,
    )
    print('VALID final checkpoint: %s epoch=32 scratch_only=true'
          % job['run_id'])


if __name__ == '__main__':
    try:
        main()
    except (
        FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError,
    ) as error:
        raise SystemExit('INVALID final Noise8 protocol: %s' % error)
