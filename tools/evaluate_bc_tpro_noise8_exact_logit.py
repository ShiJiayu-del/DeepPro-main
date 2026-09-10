#!/usr/bin/env python3
"""Exact raw-logit workpoints for the dedicated Noise8 BC-TPro experiment.

The threshold-discovery pass and counting pass use the same temporary float32
prediction arrays.  This avoids sigmoid saturation and avoids relying on a
second, potentially nondeterministic CUDA replay.  Temporary prediction arrays
are removed before the command exits; only integer count tables and summaries
are retained.

This tool is intentionally separate from ``evaluate_bc_tpro_exact_logit.py``.
The older tool belongs to the Clean-training/Noise8-transfer experiment and is
not modified by this protocol.
"""

import argparse
import importlib
import inspect
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.TestDataLoader import TestIRSeqDataLoader  # noqa: E402
from runtime_utils import load_checkpoint, parse_visible_devices  # noqa: E402
from tools import evaluate_bc_tpro_exact_logit as legacy  # noqa: E402


SCHEMA_VERSION = 1
PROTOCOL_NAME = 'bc_tpro_noise8_exact_logit_v1'
DATASET = 'NUDT-MIRSDT-Noise8.0_FJY'
MODEL = 'DeepPro-Plus_BCTPro'
EXPECTED_DATA_ROOT = Path(
    '/home/user/4T_Storage/SJY/CSIG2026/datasets/NUDT-MIRSDT-Noise8.0_FJY'
)
VAL_NAMES = (
    'Sequence9', 'Sequence13', 'Sequence14', 'Sequence16',
    'Sequence17', 'Sequence20', 'Sequence29', 'Sequence31',
    'Sequence45', 'Sequence49', 'Sequence55', 'Sequence61',
    'Sequence68', 'Sequence74', 'Sequence77', 'Sequence84',
)
SEED_GPU = {47: '0', 49: '1', 51: '2'}
VARIANT_PREFIX_LABEL = {
    'none': ('b1', 'B1'),
    'temporal_control': ('c0', 'C0'),
    'center_multiscale': ('c1', 'C1'),
    'center_ring': ('c2', 'C2'),
}
MODERNIZED_PROFILE = 'modernized'
UPSTREAM_PROFILE = 'upstream8fa1a68_fp32'
PROFILES = {
    MODERNIZED_PROFILE: {
        'log_prefix': '',
        'train_amp': 1,
        'eval_amp': 1,
        'inference_amp': True,
        'upstream_compat': None,
    },
    UPSTREAM_PROFILE: {
        'log_prefix': 'Upstream8fa1a68-FP32-',
        'train_amp': 0,
        'eval_amp': 0,
        'inference_amp': False,
        'upstream_compat': 1,
    },
}


# Re-export the mathematical primitives used by the CPU tests.  These helpers
# do not perform file identity checks or write metadata.
StreamingTopK = legacy.StreamingTopK
prepared_frame_events = legacy.prepared_frame_events
frame_counts_at_thresholds = legacy.frame_counts_at_thresholds
merge_raw_logit_window = legacy.merge_raw_logit_window
stitched_sequence_stream = legacy.stitched_sequence_stream
sequence_storage_length = legacy.sequence_storage_length
unique_sequence_frames = legacy.unique_sequence_frames
image_hw = legacy.image_hw
atomic_json_dump = legacy.atomic_json_dump
atomic_npz_dump = legacy.atomic_npz_dump
compute_workpoint_summary = legacy.compute_workpoint_summary
low_fa_pauc_from_counts = legacy.low_fa_pauc_from_counts
estimate_resources = legacy.estimate_resources


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='Exactly one physical CUDA GPU id.')
    parser.add_argument('--datapath', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sequence-list', type=Path, required=True)
    parser.add_argument('--logpath', type=Path, default=REPO_ROOT / 'log')
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--epoch', type=int, default=32)
    parser.add_argument('--seqlen', type=int, default=40)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--structure-variant', required=True)
    parser.add_argument('--repeat-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--eval-chunk-rows', type=int, default=32)
    parser.add_argument('--test-workers', type=int, default=1)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cudnn-deterministic', type=int, choices=(0, 1), default=0)
    parser.add_argument('--cudnn-benchmark', type=int, choices=(0, 1), default=0)
    parser.add_argument('--low-fa-cap', type=float, default=5e-5)
    parser.add_argument('--reference-json', type=Path)
    parser.add_argument('--max-cache-bytes', type=int, default=2 * 1024 ** 3)
    parser.add_argument('--max-topk-working-bytes', type=int, default=2 * 1024 ** 3)
    parser.add_argument('--max-count-matrix-bytes', type=int, default=1024 ** 3)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--profile', choices=tuple(PROFILES), default=MODERNIZED_PROFILE,
    )
    return parser.parse_args(argv)


def validate_frozen_evaluation_args(args):
    profile = getattr(args, 'profile', MODERNIZED_PROFILE)
    settings = PROFILES[profile]
    expected = {
        'epoch': 32,
        'seqlen': 40,
        'eval_chunk_rows': 32,
        'test_workers': 1,
        'prefetch_factor': 1,
        'cudnn_deterministic': 0,
        'cudnn_benchmark': 0,
    }
    for field, expected_value in expected.items():
        if getattr(args, field) != expected_value:
            raise ValueError(
                'The frozen exact-logit protocol requires %s=%r.'
                % (field, expected_value)
            )
    if args.amp is not settings['inference_amp']:
        if settings['inference_amp']:
            raise ValueError('The frozen exact-logit protocol requires AMP.')
        raise ValueError(
            'The frozen exact-logit profile %s requires amp=False.' % profile
        )
    if not math.isclose(args.low_fa_cap, 5e-5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError('The frozen exact-logit protocol requires low_fa_cap=5e-5.')


@dataclass
class PassOneResult:
    target_peaks: np.ndarray
    target_peak_sequence_indices: np.ndarray
    top_background_scores: np.ndarray
    targets_by_sequence: np.ndarray
    false_region_pixels_by_sequence: np.ndarray
    global_max_logit: float


@dataclass
class PassTwoResult:
    false_counts: np.ndarray
    true_counts: np.ndarray
    targets_by_sequence: np.ndarray
    pixels_by_sequence: np.ndarray


def collect_pass_one(sequence_stream, sequence_names, expected_pixels, retain_k):
    """Discover every target transition and the low-Fa background tail."""
    topk = StreamingTopK(retain_k)
    peak_parts = []
    owner_parts = []
    targets_by_sequence = np.zeros(len(sequence_names), dtype=np.int64)
    false_pixels_by_sequence = np.zeros(len(sequence_names), dtype=np.int64)
    global_max = -np.inf
    yielded = 0
    for sequence_index, (name, logits, targets) in enumerate(sequence_stream):
        if sequence_index >= len(sequence_names):
            raise ValueError('Pass 1 yielded more sequences than expected.')
        if name != sequence_names[sequence_index]:
            raise ValueError('Pass 1 sequence-order mismatch at index %d.' % sequence_index)
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets)
        if logits.shape != targets.shape or logits.ndim != 3:
            raise ValueError('Stitched sequence tensors must be same-shape T,H,W.')
        if int(np.prod(logits.shape, dtype=np.int64)) != int(expected_pixels[sequence_index]):
            raise ValueError('Pixel-count preflight mismatch for %s.' % name)
        if not np.isfinite(logits).all():
            raise FloatingPointError('Non-finite stitched logits in %s.' % name)
        yielded += 1
        global_max = max(global_max, float(logits.max()))
        sequence_peaks = []
        for frame_logits, frame_target in zip(logits, targets):
            peaks, false_values = prepared_frame_events(frame_logits, frame_target)
            if peaks.size:
                sequence_peaks.append(peaks)
            topk.update(false_values)
            false_pixels_by_sequence[sequence_index] += int(false_values.size)
        peaks = (
            np.concatenate(sequence_peaks)
            if sequence_peaks else np.empty(0, dtype=np.float32)
        )
        targets_by_sequence[sequence_index] = int(peaks.size)
        peak_parts.append(peaks)
        owner_parts.append(np.full(peaks.size, sequence_index, dtype=np.int32))
    if yielded != len(sequence_names):
        raise ValueError('Pass 1 did not yield every selected sequence.')
    if not np.isfinite(global_max):
        raise FloatingPointError('No finite logits were observed.')
    return PassOneResult(
        target_peaks=(
            np.concatenate(peak_parts) if peak_parts else np.empty(0, np.float32)
        ),
        target_peak_sequence_indices=(
            np.concatenate(owner_parts) if owner_parts else np.empty(0, np.int32)
        ),
        top_background_scores=topk.values(),
        targets_by_sequence=targets_by_sequence,
        false_region_pixels_by_sequence=false_pixels_by_sequence,
        global_max_logit=global_max,
    )


def build_exact_thresholds(pass_one):
    maximum = np.float64(pass_one.global_max_logit)
    sentinel = np.nextafter(maximum, np.float64(np.inf))
    if not np.isfinite(sentinel) or not sentinel > maximum:
        raise FloatingPointError('Cannot construct a finite sentinel above the maximum.')
    thresholds = np.unique(np.concatenate((
        pass_one.target_peaks.astype(np.float64),
        pass_one.top_background_scores.astype(np.float64),
        np.asarray([0.0, sentinel], dtype=np.float64),
    )))
    if not np.isfinite(thresholds).all():
        raise FloatingPointError('Candidate thresholds must all be finite.')
    return thresholds, float(sentinel)


def collect_pass_two(
    sequence_stream, sequence_names, expected_pixels, expected_targets, thresholds,
):
    """Count all false pixels and detected targets at every event threshold."""
    false_counts = np.zeros(
        (thresholds.size, len(sequence_names)), dtype=np.int64,
    )
    true_counts = np.zeros_like(false_counts)
    targets_by_sequence = np.zeros(len(sequence_names), dtype=np.int64)
    yielded = 0
    for sequence_index, (name, logits, targets) in enumerate(sequence_stream):
        if sequence_index >= len(sequence_names):
            raise ValueError('Pass 2 yielded more sequences than expected.')
        if name != sequence_names[sequence_index]:
            raise ValueError('Pass 2 sequence-order mismatch at index %d.' % sequence_index)
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets)
        if logits.shape != targets.shape or logits.ndim != 3:
            raise ValueError('Cached tensors must be same-shape T,H,W.')
        if int(np.prod(logits.shape, dtype=np.int64)) != int(expected_pixels[sequence_index]):
            raise ValueError('Pixel-count mismatch in pass 2 for %s.' % name)
        yielded += 1
        sequence_targets = 0
        for frame_logits, frame_target in zip(logits, targets):
            frame_false, frame_true, peaks = frame_counts_at_thresholds(
                frame_logits, frame_target, thresholds,
            )
            false_counts[:, sequence_index] += frame_false
            true_counts[:, sequence_index] += frame_true
            sequence_targets += int(peaks.size)
        targets_by_sequence[sequence_index] = sequence_targets
    if yielded != len(sequence_names):
        raise ValueError('Pass 2 did not yield every selected sequence.')
    if not np.array_equal(targets_by_sequence, expected_targets):
        raise RuntimeError('Target counts changed between threshold discovery and counting.')
    return PassTwoResult(
        false_counts=false_counts,
        true_counts=true_counts,
        targets_by_sequence=targets_by_sequence,
        pixels_by_sequence=np.asarray(expected_pixels, dtype=np.int64),
    )


def build_data_descriptor(dataset, sequence_names, sequence_datasets, sequence_list):
    """Validate the selected data semantically without content fingerprints."""
    per_sequence = []
    selected_files = 0
    selected_bytes = 0
    for name, seq_dataset in zip(sequence_names, sequence_datasets):
        frames = unique_sequence_frames(seq_dataset)
        if not frames:
            raise ValueError('Sequence %s contains no frames.' % name)
        height, width = image_hw(frames[0][2])
        for frame_index, paths in enumerate(frames):
            for role, raw_path in zip(('image', 'label', 'centroid'), paths):
                if raw_path == 'None':
                    raise ValueError('%s is missing for %s frame %d.' % (
                        role, name, frame_index,
                    ))
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError('Missing %s file: %s' % (role, path))
                selected_files += 1
                selected_bytes += int(path.stat().st_size)
            if image_hw(paths[2]) != (height, width):
                raise ValueError('Centroid dimensions change within %s.' % name)
        per_sequence.append({
            'name': name,
            'frames': len(frames),
            'height': height,
            'width': width,
            'pixels': len(frames) * height * width,
        })
    return {
        'identity_method': 'absolute paths, ordered names, dimensions, counts, and sizes',
        'dataset': dataset,
        'sequence_list_path': str(Path(sequence_list).expanduser().resolve()),
        'sequence_names': list(sequence_names),
        'selected_file_count': selected_files,
        'selected_file_bytes': selected_bytes,
        'sequences': per_sequence,
        'total_pixels': int(sum(item['pixels'] for item in per_sequence)),
    }


def resolve_experiment_dir(logpath, log_dir):
    root = (Path(logpath).expanduser().resolve() / 'sem_seg')
    experiment_dir = (root / log_dir).resolve()
    try:
        experiment_dir.relative_to(root)
    except ValueError as error:
        raise ValueError('--log-dir must remain under %s.' % root) from error
    if not experiment_dir.is_dir():
        raise FileNotFoundError('Missing experiment directory: %s' % experiment_dir)
    return experiment_dir


def validate_training_provenance(experiment_dir, args, model_name):
    settings = PROFILES[args.profile]
    log_path = experiment_dir / 'logs' / ('%s.txt' % model_name)
    namespace = legacy.parse_training_namespace(log_path)
    expected = {
        'model': MODEL,
        'structure_variant': args.structure_variant,
        'seed': args.seed,
        'gpu': SEED_GPU[args.seed],
        'epoch': args.epoch,
        'seqlen': args.seqlen,
        'log_dir': args.log_dir,
        'dataset': DATASET,
        'batch_size': 4,
        'gradient_accumulation_steps': 1,
        'patch_size': 128,
        'sample_rate': 0.1,
        'sequence_augmentation': 0,
        'loss': 'soft_iou',
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'decay_rate': 0.0001,
        'step_size': 10,
        'lr_decay': 0.7,
        'train_amp': settings['train_amp'],
        'eval_amp': settings['eval_amp'],
        'eval_chunk_rows': 32,
        'skip_inprocess_validation': 1,
        'train_workers': 4,
        'val_workers': 1,
        'prefetch_factor': 2,
        'deterministic': 1,
        'resume': 'never',
        'resume_checkpoint': None,
        'base_ckpt': '',
        'spatial_ckpt': '',
        'st_ckpt': '',
        'freeze_pretrained': 0,
    }
    if settings['upstream_compat'] is not None:
        expected['upstream_compat'] = settings['upstream_compat']
    for field, expected_value in expected.items():
        if field not in namespace or namespace[field] != expected_value:
            raise ValueError(
                'Training provenance %s mismatch: expected %r, found %r.'
                % (field, expected_value, namespace.get(field))
            )
    if Path(namespace.get('datapath', '')).expanduser().resolve() != args.datapath:
        raise ValueError('Training datapath does not match the Noise8 data root.')
    val_path = Path(namespace.get('val_sequence_list', '')).expanduser().resolve()
    if val_path != args.sequence_list:
        raise ValueError('Training validation split does not match --sequence-list.')
    expected_train = args.sequence_list.parent / 'train_sequences.txt'
    train_path = Path(namespace.get('train_sequence_list', '')).expanduser().resolve()
    if train_path != expected_train.resolve():
        raise ValueError('Training split does not match the frozen Noise8 split.')
    marker = 'from random weights; no base checkpoint loaded.'
    if log_path.read_text(encoding='utf-8').count(marker) != 1:
        raise ValueError('Scratch-initialization marker is missing or duplicated.')
    return {
        'log_path': str(log_path.resolve()),
        'profile': args.profile,
        'verified_fields': expected,
        'scratch_only_verified': True,
    }


def load_reference(path, args, expected_pixels, expected_targets):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Reference JSON has the wrong schema version.')
    if payload.get('protocol', {}).get('name') != PROTOCOL_NAME:
        raise ValueError('Reference JSON is not a Noise8 exact-logit output.')
    reference_profile = payload.get('protocol', {}).get(
        'profile', MODERNIZED_PROFILE,
    )
    if reference_profile != args.profile:
        raise ValueError('Reference JSON belongs to a different protocol profile.')
    identity = payload.get('run_identity', {})
    expected_identity = {
        'structure_variant': 'none',
        'seed': args.seed,
        'dataset': DATASET,
        'sequence_length': args.seqlen,
        'repeat_index': 0,
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError(
                'Reference %s mismatch: expected %r, found %r.'
                % (field, expected, identity.get(field))
            )
    workpoint = payload.get('workpoint_at_logit_zero')
    required = ('false_pixels', 'true_targets', 'total_targets', 'pixels')
    if not isinstance(workpoint, dict) or any(field not in workpoint for field in required):
        raise ValueError('Reference JSON lacks its logit-zero integer workpoint.')
    values = {}
    for field in required:
        value = workpoint[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Reference %s must be a non-negative integer.' % field)
        values[field] = value
    if values['pixels'] != int(np.sum(expected_pixels)):
        raise ValueError('Reference pixel denominator differs from this validation set.')
    if values['total_targets'] != int(np.sum(expected_targets)):
        raise ValueError('Reference target denominator differs from this validation set.')
    descriptor = payload.get('data_descriptor', {})
    if descriptor.get('sequence_names') != list(VAL_NAMES):
        raise ValueError('Reference sequence order differs from the fixed validation split.')
    return {'path': str(path), 'workpoint': values}


def verify_output_paths(args):
    json_path = args.output_json.expanduser().resolve()
    npz_path = args.output_npz.expanduser().resolve()
    if json_path == npz_path:
        raise ValueError('JSON and NPZ destinations must differ.')
    if not args.overwrite:
        existing = [str(path) for path in (json_path, npz_path) if path.exists()]
        if existing:
            raise FileExistsError('Refusing to overwrite: %s' % ', '.join(existing))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    return json_path, npz_path


def cache_inference_stream(sequence_stream, sequence_names, cache_dir):
    """Write temporary arrays before yielding them to threshold discovery."""
    for index, (name, logits, targets) in enumerate(sequence_stream):
        if index >= len(sequence_names) or name != sequence_names[index]:
            raise ValueError('Inference sequence order differs from the fixed split.')
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        targets = np.ascontiguousarray(targets)
        np.save(cache_dir / ('%02d_logits.npy' % index), logits, allow_pickle=False)
        np.save(cache_dir / ('%02d_targets.npy' % index), targets, allow_pickle=False)
        yield name, logits, targets


def cached_sequence_stream(cache_dir, sequence_names):
    for index, name in enumerate(sequence_names):
        logits_path = cache_dir / ('%02d_logits.npy' % index)
        targets_path = cache_dir / ('%02d_targets.npy' % index)
        if not logits_path.is_file() or not targets_path.is_file():
            raise FileNotFoundError('Temporary pass-1 arrays are incomplete.')
        logits = np.load(logits_path, mmap_mode='r', allow_pickle=False)
        targets = np.load(targets_path, mmap_mode='r', allow_pickle=False)
        yield name, logits, targets
        del logits, targets


def main(args):
    if args.profile not in PROFILES:
        raise ValueError('Unknown Noise8 protocol profile: %s.' % args.profile)
    if args.dataset != DATASET:
        raise ValueError('This evaluator only accepts dataset %s.' % DATASET)
    if args.epoch != 32 or args.seqlen != 40:
        raise ValueError('The frozen protocol requires epoch 32 and sequence length 40.')
    if args.test_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError('Invalid DataLoader worker/prefetch settings.')
    if not math.isfinite(args.low_fa_cap) or not 0 < args.low_fa_cap < 1:
        raise ValueError('--low-fa-cap must be finite and lie in (0, 1).')
    if min(
        args.max_cache_bytes, args.max_topk_working_bytes,
        args.max_count_matrix_bytes,
    ) <= 0:
        raise ValueError('Memory and cache limits must be positive.')
    validate_frozen_evaluation_args(args)
    if args.seed not in SEED_GPU:
        raise ValueError('The frozen protocol only includes seeds 47, 49, and 51.')
    if args.structure_variant not in VARIANT_PREFIX_LABEL:
        raise ValueError('Unknown frozen structure variant.')
    prefix, label = VARIANT_PREFIX_LABEL[args.structure_variant]
    expected_run_id = '%s_%s_seed%d' % (prefix, args.structure_variant, args.seed)
    expected_log_dir = (
        '2026-09-09/%s__%sSoftIoU-BCTPro-%s_seed%d_E32'
        % (
            DATASET, PROFILES[args.profile]['log_prefix'], label, args.seed,
        )
    )
    if args.run_id != expected_run_id or args.log_dir != expected_log_dir:
        raise ValueError('Run identity does not match the frozen Noise8 manifest.')
    output_json, output_npz = verify_output_paths(args)

    devices = parse_visible_devices(args.gpu)
    if len(devices) != 1:
        raise ValueError('Exact evaluation uses exactly one GPU.')
    if devices[0] != SEED_GPU[args.seed]:
        raise ValueError('Physical GPU does not match the frozen seed mapping.')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
    torch.backends.cudnn.deterministic = bool(args.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    args.datapath = args.datapath.expanduser().resolve()
    args.sequence_list = args.sequence_list.expanduser().resolve()
    args.logpath = args.logpath.expanduser().resolve()
    if args.datapath != EXPECTED_DATA_ROOT.resolve():
        raise ValueError('Noise8 data root must be exactly %s.' % EXPECTED_DATA_ROOT)
    if not args.datapath.is_dir() or not args.sequence_list.is_file():
        raise FileNotFoundError('Dataset root or sequence list does not exist.')
    listed_names = [
        line.strip() for line in args.sequence_list.read_text(encoding='utf-8').splitlines()
    ]
    if listed_names != list(VAL_NAMES):
        raise ValueError('Validation list is not the fixed ordered 16-sequence split.')

    experiment_dir = resolve_experiment_dir(args.logpath, args.log_dir)
    checkpoint_path = experiment_dir / 'checkpoints' / 'epoch_32_model.pth'
    checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
    if checkpoint.get('model_name') != MODEL:
        raise ValueError('Checkpoint model_name does not match the frozen model.')
    if int(checkpoint.get('epoch', -1)) + 1 != args.epoch:
        raise ValueError('Checkpoint epoch metadata mismatch.')
    model_config = dict(checkpoint.get('model_config', {}))
    expected_model_config = {
        'eval_chunk_rows': 32,
        'structure_variant': args.structure_variant,
        'structure_bottleneck_channels': 8,
    }
    if model_config != expected_model_config:
        raise ValueError('Checkpoint model_config differs from the frozen protocol.')
    training_provenance = validate_training_provenance(
        experiment_dir, args, MODEL,
    )

    dataset = TestIRSeqDataLoader(
        DATASET,
        data_root=str(args.datapath),
        seq_len=args.seqlen,
        cat_len=int(args.seqlen * 0.1),
        transform=None,
        load_annotations=True,
        split='val',
        sequence_list_file=str(args.sequence_list),
    )
    sequence_names = list(dataset.seq_names)
    if sequence_names != list(VAL_NAMES):
        raise ValueError('Loader sequence order differs from the fixed validation split.')
    sequence_datasets = [dataset[index] for index in range(len(dataset))]
    data_descriptor = build_data_descriptor(
        DATASET, sequence_names, sequence_datasets, args.sequence_list,
    )
    pixels_by_sequence = np.asarray(
        [item['pixels'] for item in data_descriptor['sequences']], dtype=np.int64,
    )
    total_pixels = int(pixels_by_sequence.sum())
    estimated_cache_bytes = int(total_pixels * 8)
    if estimated_cache_bytes > args.max_cache_bytes:
        raise MemoryError(
            'Conservative temporary-array estimate %d exceeds limit %d.'
            % (estimated_cache_bytes, args.max_cache_bytes)
        )
    if shutil.disk_usage(output_json.parent).free < estimated_cache_bytes * 2:
        raise OSError('Insufficient free space for bounded temporary prediction arrays.')

    cap_k = int(math.floor(args.low_fa_cap * total_pixels)) + 1
    resource_estimate = estimate_resources(cap_k, len(sequence_names))
    if resource_estimate['estimated_topk_working_bytes'] > args.max_topk_working_bytes:
        raise MemoryError('Estimated top-K working memory exceeds the configured limit.')
    if resource_estimate['estimated_dense_count_matrix_bytes'] > args.max_count_matrix_bytes:
        raise MemoryError('Estimated count matrices exceed the configured limit.')

    sys.path.insert(0, str(experiment_dir))
    model_module = importlib.import_module(MODEL)
    constructor_parameters = inspect.signature(model_module.detector).parameters
    constructor_config = {
        key: value for key, value in model_config.items()
        if key in constructor_parameters and key not in {'spatial_ckpt', 'st_ckpt'}
    }
    if 'freeze_pretrained' in constructor_config:
        constructor_config['freeze_pretrained'] = False
    if 'eval_chunk_rows' in constructor_parameters:
        constructor_config['eval_chunk_rows'] = args.eval_chunk_rows
    detector = model_module.detector(
        1, args.seqlen, args.seqlen, **constructor_config
    )
    detector.load_state_dict(
        legacy.clean_model_state_dict(checkpoint['model_state_dict']), strict=True,
    )
    del checkpoint
    detector = detector.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix='.noise8-exact-logit-', dir=str(output_json.parent),
    ) as cache_name:
        cache_dir = Path(cache_name)
        inference_start = time.time()
        inference_stream = stitched_sequence_stream(
            detector, sequence_names, sequence_datasets, args.seqlen,
            args.amp, args.test_workers, args.prefetch_factor,
        )
        pass_one = collect_pass_one(
            cache_inference_stream(inference_stream, sequence_names, cache_dir),
            sequence_names, pixels_by_sequence, cap_k,
        )
        inference_seconds = time.time() - inference_start
        thresholds, sentinel = build_exact_thresholds(pass_one)
        actual_matrix_bytes = int(thresholds.size * len(sequence_names) * 4 * 8)
        if actual_matrix_bytes > args.max_count_matrix_bytes:
            raise MemoryError('Actual count matrices exceed the configured limit.')
        counting_start = time.time()
        pass_two = collect_pass_two(
            cached_sequence_stream(cache_dir, sequence_names), sequence_names,
            pixels_by_sequence, pass_one.targets_by_sequence, thresholds,
        )
        counting_seconds = time.time() - counting_start

    reference = load_reference(
        args.reference_json, args, pass_two.pixels_by_sequence,
        pass_two.targets_by_sequence,
    )
    reference_workpoint = reference['workpoint'] if reference is not None else None
    at_zero, fixed_fa, fixed_pd = compute_workpoint_summary(
        pass_two, thresholds, reference_workpoint,
    )
    if reference_workpoint is None:
        reference_workpoint = {
            field: at_zero[field]
            for field in ('false_pixels', 'true_targets', 'total_targets', 'pixels')
        }
    low_fa = low_fa_pauc_from_counts(pass_two, args.low_fa_cap)
    target_matrix = np.broadcast_to(
        pass_two.targets_by_sequence[None, :], pass_two.false_counts.shape,
    )
    pixel_matrix = np.broadcast_to(
        pass_two.pixels_by_sequence[None, :], pass_two.false_counts.shape,
    )
    atomic_npz_dump(
        output_npz,
        thresholds=thresholds,
        sequence_names=np.asarray(sequence_names),
        false_pixels_by_threshold_sequence=pass_two.false_counts,
        true_targets_by_threshold_sequence=pass_two.true_counts,
        total_targets_by_threshold_sequence=target_matrix,
        pixel_count_by_threshold_sequence=pixel_matrix,
        target_peaks=pass_one.target_peaks,
        target_peak_sequence_indices=pass_one.target_peak_sequence_indices,
        top_background_scores=pass_one.top_background_scores,
    )
    elapsed_seconds = time.time() - start
    payload = {
        'schema_version': SCHEMA_VERSION,
        'protocol': {
            'name': PROTOCOL_NAME,
            'profile': args.profile,
            'status': 'post_training_sensitivity_analysis',
            'score_domain': 'raw logits',
            'threshold_comparison': 'score >= threshold',
            'window_stitch_rule': 'elementwise maximum before scoring',
            'passes': 2,
            'pass_replay': 'temporary pass-1 float32 arrays reused by pass 2',
            'temporary_prediction_arrays_retained': False,
            'official_test_accessed': False,
            'full_roc_auc': 'not covered',
            'declared_workpoints': {
                'pd_at_reference_fa': 'exact discrete maximum over all target events',
                'fa_at_reference_pd': 'exact discrete minimum over all target events',
            },
        },
        'run_identity': {
            'profile': args.profile,
            'run_id': args.run_id,
            'dataset': DATASET,
            'datapath': str(args.datapath),
            'model': MODEL,
            'structure_variant': args.structure_variant,
            'seed': args.seed,
            'checkpoint_epoch': args.epoch,
            'sequence_length': args.seqlen,
            'log_dir': args.log_dir,
            'repeat_index': args.repeat_index,
        },
        'checkpoint': {
            'path': str(checkpoint_path.resolve()),
            'model_config': expected_model_config,
        },
        'training_provenance': training_provenance,
        'data_descriptor': data_descriptor,
        'reference': reference,
        'reference_workpoint': reference_workpoint,
        'retention': {
            'low_fa_cap': args.low_fa_cap,
            'requested_background_tail': cap_k,
            'retained_background_scores': int(pass_one.top_background_scores.size),
            'false_region_scores_seen': int(
                pass_one.false_region_pixels_by_sequence.sum()
            ),
            'resource_estimate_before_inference': resource_estimate,
            'estimated_temporary_array_bytes': estimated_cache_bytes,
            'actual_dense_count_matrix_bytes': actual_matrix_bytes,
        },
        'thresholds': {
            'count': int(thresholds.size),
            'minimum': float(thresholds[0]),
            'maximum_finite_sentinel': sentinel,
            'global_max_raw_logit': pass_one.global_max_logit,
            'includes_logit_zero': True,
            'sources': [
                'all target peaks', 'low-Fa top background logits',
                'logit zero', 'finite sentinel above global maximum',
            ],
        },
        'counts': {
            'npz_path': str(output_npz),
            'axes': ['threshold', 'sequence'],
            'shape': [int(thresholds.size), len(sequence_names)],
            'dtype': 'int64',
            'targets_by_sequence': pass_two.targets_by_sequence.tolist(),
            'pixels_by_sequence': pass_two.pixels_by_sequence.tolist(),
            'false_region_pixels_by_sequence': (
                pass_one.false_region_pixels_by_sequence.tolist()
            ),
        },
        'workpoint_at_logit_zero': at_zero,
        'low_fa': low_fa,
        'pd_at_reference_fa': fixed_fa,
        'fa_at_reference_pd': fixed_pd,
        'inference': {
            'amp': bool(args.amp),
            'eval_chunk_rows': args.eval_chunk_rows,
            'test_workers': args.test_workers,
            'prefetch_factor': args.prefetch_factor,
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'inference_seconds': inference_seconds,
            'counting_seconds': counting_seconds,
            'elapsed_seconds': elapsed_seconds,
            'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / (1024 ** 3),
            'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved() / (1024 ** 3),
        },
        'environment': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'physical_gpu_argument': devices[0],
            'visible_gpu_name': torch.cuda.get_device_name(0),
        },
    }
    atomic_json_dump(payload, output_json)
    print('Noise8 exact-logit NPZ: %s' % output_npz)
    print('Noise8 exact-logit JSON: %s' % output_json)


if __name__ == '__main__':
    main(parse_args())
