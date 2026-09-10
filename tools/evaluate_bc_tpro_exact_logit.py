#!/usr/bin/env python3
"""Two-pass, exact-event logit evaluation for the BC-TPro protocol amendment.

This evaluator deliberately coexists with (and does not overwrite) the
preregistered 109-point probability-grid results produced by ``test.py``.
Only one stitched sequence is resident at a time and no prediction maps are
written to disk.
"""

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.TestDataLoader import TestIRSeqDataLoader  # noqa: E402
from runtime_utils import load_checkpoint, parse_visible_devices  # noqa: E402
from ShootingRules import ShootingRules  # noqa: E402


SCHEMA_VERSION = 1
PROTOCOL_NAME = 'bc_tpro_exact_logit_protocol_amendment_v1'
RETENTION_MODES = ('target-events', 'full-reference-grid')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='Exactly one physical CUDA GPU id.')
    parser.add_argument('--datapath', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--sequence-list', type=Path, required=True)
    parser.add_argument(
        '--condition', choices=('clean_val', 'noise8_val'), required=True,
    )
    parser.add_argument('--logpath', type=Path, default=REPO_ROOT / 'log')
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--epoch', type=int, default=32)
    parser.add_argument('--seqlen', type=int, default=40)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--structure-variant', required=True)
    parser.add_argument('--eval-chunk-rows', type=int, default=32)
    parser.add_argument('--test-workers', type=int, default=1)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--cudnn-deterministic', type=int, choices=(0, 1), default=0)
    parser.add_argument('--cudnn-benchmark', type=int, choices=(0, 1), default=0)
    parser.add_argument('--low-fa-cap', type=float, default=5e-5)
    parser.add_argument('--reference-json', type=Path)
    parser.add_argument(
        '--reference-retention-mode', choices=RETENTION_MODES,
        default='target-events',
        help=(
            'target-events retains only the low-Fa top K and is still exact '
            'for the discrete fixed-Fa/fixed-Pd workpoints; full-reference-grid '
            'also retains every background transition through F_ref.'
        ),
    )
    parser.add_argument(
        '--max-topk-working-bytes', type=int, default=2 * 1024 ** 3,
        help='Refuse before inference when estimated in-memory top-K work exceeds this.',
    )
    parser.add_argument(
        '--max-count-matrix-bytes', type=int, default=1024 ** 3,
        help='Refuse before inference when the conservative dense-count estimate exceeds this.',
    )
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-npz', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def sha256_file(path, chunk_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def update_file_content_identity(digest, logical_name, path):
    path = Path(path)
    file_digest = sha256_file(path)
    record = '%s\0%d\0%s\n' % (logical_name, path.stat().st_size, file_digest)
    digest.update(record.encode('utf-8'))
    return path.stat().st_size


def atomic_json_dump(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8',
            prefix='.%s.' % destination.name, suffix='.tmp',
            dir=str(destination.parent), delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(str(temporary), str(destination))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_npz_dump(destination, **arrays):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', prefix='.%s.' % destination.name, suffix='.tmp',
            dir=str(destination.parent), delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sequence_storage_length(seq_dataset):
    return max(
        frame_data[2]
        for sample in seq_dataset.samplelist
        for frame_data in sample
    ) + 1


def unique_sequence_frames(seq_dataset):
    by_index = {}
    for sample in seq_dataset.samplelist:
        for image_path, label_path, frame_index, centroid_path in sample:
            value = (str(image_path), str(label_path), str(centroid_path))
            if frame_index in by_index and by_index[frame_index] != value:
                raise ValueError('Overlapping windows disagree at frame %d.' % frame_index)
            by_index[frame_index] = value
    expected = list(range(sequence_storage_length(seq_dataset)))
    if sorted(by_index) != expected:
        raise ValueError('Sequence frame indices are not contiguous from zero.')
    return [by_index[index] for index in expected]


def image_hw(path):
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def build_data_identity(dataset, sequence_names, sequence_datasets, sequence_list):
    """Hash every selected image, label, and centroid by content."""
    digest = hashlib.sha256()
    digest.update(b'bc-tpro-data-identity-v1\n')
    digest.update(('dataset=%s\n' % dataset).encode('utf-8'))
    per_sequence = []
    total_bytes = 0
    total_files = 0
    for name, seq_dataset in zip(sequence_names, sequence_datasets):
        frames = unique_sequence_frames(seq_dataset)
        first_centroid = frames[0][2]
        height, width = image_hw(first_centroid)
        for frame_index, paths in enumerate(frames):
            roles = ('image', 'label', 'centroid')
            for role, path in zip(roles, paths):
                if path == 'None':
                    raise ValueError('Exact evaluation requires %s for %s frame %d.' % (
                        role, name, frame_index,
                    ))
                logical = '%s/%06d/%s' % (name, frame_index, role)
                total_bytes += update_file_content_identity(digest, logical, path)
                total_files += 1
            centroid_hw = image_hw(paths[2])
            if centroid_hw != (height, width):
                raise ValueError(
                    'Centroid dimensions change within %s: %r versus %r.'
                    % (name, (height, width), centroid_hw)
                )
        pixels = len(frames) * height * width
        per_sequence.append({
            'name': name,
            'frames': len(frames),
            'height': height,
            'width': width,
            'pixels': pixels,
        })
    return {
        'identity_scheme': 'sha256(logical-role, size, per-file-content-sha256)',
        'content_sha256': digest.hexdigest(),
        'sequence_list_path': str(Path(sequence_list).resolve()),
        'sequence_list_sha256': sha256_file(sequence_list),
        'selected_file_count': total_files,
        'selected_file_bytes': total_bytes,
        'sequences': per_sequence,
        'total_pixels': int(sum(item['pixels'] for item in per_sequence)),
    }


def clean_model_state_dict(state_dict):
    if state_dict and all(key.startswith('module.') for key in state_dict):
        return {key[len('module.'):]: value for key, value in state_dict.items()}
    return state_dict


def resolve_experiment_dir(logpath, log_dir):
    experiment_root = (Path(logpath).expanduser().resolve() / 'sem_seg')
    experiment_dir = (experiment_root / log_dir).resolve()
    try:
        experiment_dir.relative_to(experiment_root)
    except ValueError as error:
        raise ValueError('--log-dir must remain under %s.' % experiment_root) from error
    if not experiment_dir.is_dir():
        raise FileNotFoundError('Missing experiment directory: %s' % experiment_dir)
    return experiment_dir


def resolve_model_name(checkpoint, experiment_dir):
    model_name = checkpoint.get('model_name')
    if not model_name:
        raise ValueError('Exact evaluator requires checkpoint model_name metadata.')
    model_path = (experiment_dir / ('%s.py' % model_name)).resolve()
    if model_path.parent != experiment_dir or not model_path.is_file():
        raise ValueError('Unsafe or missing model snapshot: %s' % model_path)
    return str(model_name), model_path


def dependency_hashes(experiment_dir, model_path):
    candidates = [
        Path(__file__).resolve(),
        REPO_ROOT / 'ShootingRules.py',
        REPO_ROOT / 'data_utils' / 'TestDataLoader.py',
        REPO_ROOT / 'data_utils' / 'loader_utils.py',
        REPO_ROOT / 'runtime_utils.py',
        REPO_ROOT / 'tools' / 'run_bc_tpro_exact_logit.sh',
        REPO_ROOT / 'networks' / 'layers' / 'basic.py',
        REPO_ROOT / 'networks' / 'layers' / 'TPro.py',
        model_path,
        experiment_dir / 'bc_tpro_adapter.py',
    ]
    result = {}
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError('Missing hashed evaluation dependency: %s' % path)
        try:
            key = str(path.relative_to(REPO_ROOT))
        except ValueError:
            key = str(path)
        result[key] = sha256_file(path)
    return result


def parse_training_namespace(log_path):
    """Read the literal argparse Namespace recorded by train.py."""
    namespaces = []
    for line_number, line in enumerate(
        Path(log_path).read_text(encoding='utf-8').splitlines(), start=1,
    ):
        marker = line.find('Namespace(')
        if marker < 0:
            continue
        try:
            node = ast.parse(line[marker:], mode='eval').body
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                'Cannot parse training Namespace at %s:%d.'
                % (log_path, line_number)
            ) from error
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != 'Namespace'
            or node.args
            or any(keyword.arg is None for keyword in node.keywords)
        ):
            raise ValueError('Unsupported training Namespace representation.')
        namespaces.append({
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
        })
    if len(namespaces) != 1:
        raise ValueError(
            'Expected exactly one training Namespace in %s, found %d.'
            % (log_path, len(namespaces))
        )
    return namespaces[0]


def validate_training_provenance(experiment_dir, args, model_name):
    log_path = experiment_dir / 'logs' / ('%s.txt' % model_name)
    if not log_path.is_file():
        raise FileNotFoundError('Missing training provenance log: %s' % log_path)
    namespace = parse_training_namespace(log_path)
    expected = {
        'model': model_name,
        'structure_variant': args.structure_variant,
        'seed': args.seed,
        'epoch': args.epoch,
        'seqlen': args.seqlen,
        'log_dir': args.log_dir,
        'dataset': 'NUDT-MIRSDT',
        'resume': 'never',
        'resume_checkpoint': None,
        'base_ckpt': '',
        'spatial_ckpt': '',
        'st_ckpt': '',
        'freeze_pretrained': 0,
    }
    for field, expected_value in expected.items():
        if field not in namespace or namespace[field] != expected_value:
            raise ValueError(
                'Training provenance %s mismatch: expected %r, found %r.'
                % (field, expected_value, namespace.get(field))
            )
    return {
        'log_path': str(log_path.resolve()),
        'log_sha256': sha256_file(log_path),
        'verified_fields': expected,
        'scratch_only_verified': True,
    }


class StreamingTopK(object):
    """Bounded exact top-K for finite float32 scores, including ties."""

    def __init__(self, k):
        if k < 0:
            raise ValueError('k must be non-negative.')
        self.k = int(k)
        self._values = np.empty(0, dtype=np.float32)
        self.seen = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            raise FloatingPointError('Top-K input contains non-finite logits.')
        self.seen += int(values.size)
        if self.k == 0 or values.size == 0:
            return
        combined = np.concatenate((self._values, values))
        if combined.size > self.k:
            split = combined.size - self.k
            combined.partition(split)
            self._values = combined[split:]
        else:
            self._values = combined

    def values(self):
        return np.sort(self._values.astype(np.float32, copy=True))


def prepared_frame_events(logits, target):
    """Return exact target peaks and false-region values for one frame."""
    logits = np.asarray(logits, dtype=np.float32)
    target = np.asarray(target)
    if logits.ndim != 2 or target.ndim != 2 or logits.shape != target.shape:
        raise ValueError('Logit and target frames must be same-shape 2-D arrays.')
    if not np.isfinite(logits).all():
        raise FloatingPointError('Prediction contains non-finite raw logits.')
    target_coordinates, box2_map = ShootingRules._prepare_target(target)
    peaks = np.empty(len(target_coordinates), dtype=np.float32)
    height, width = logits.shape
    for target_index, pixel_coords in enumerate(target_coordinates):
        peak = -np.inf
        for row, column in pixel_coords:
            top = max(0, int(row) - 1)
            bottom = min(height, int(row) + 2)
            left = max(0, int(column) - 1)
            right = min(width, int(column) + 2)
            peak = max(peak, float(logits[top:bottom, left:right].max()))
        peaks[target_index] = peak
    if not np.isfinite(peaks).all():
        raise FloatingPointError('A prepared target has no finite 3x3 peak.')
    return peaks, logits[box2_map].reshape(-1)


@dataclass
class PassOneResult:
    target_peaks: np.ndarray
    target_peak_sequence_indices: np.ndarray
    top_background_scores: np.ndarray
    targets_by_sequence: np.ndarray
    false_region_pixels_by_sequence: np.ndarray
    logits_sha256_by_sequence: list
    targets_sha256_by_sequence: list
    global_max_logit: float


def collect_pass_one(sequence_stream, sequence_names, expected_pixels, retain_k):
    topk = StreamingTopK(retain_k)
    peak_parts = []
    peak_owner_parts = []
    targets_by_sequence = np.zeros(len(sequence_names), dtype=np.int64)
    false_pixels_by_sequence = np.zeros(len(sequence_names), dtype=np.int64)
    logits_hashes = []
    target_hashes = []
    global_max = -np.inf
    for sequence_index, (name, logits, targets) in enumerate(sequence_stream):
        if name != sequence_names[sequence_index]:
            raise ValueError('Pass 1 sequence-order mismatch at index %d.' % sequence_index)
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        targets = np.asarray(targets)
        if logits.shape != targets.shape or logits.ndim != 3:
            raise ValueError('Stitched sequence tensors must be same-shape T,H,W.')
        observed_pixels = int(np.prod(logits.shape, dtype=np.int64))
        if observed_pixels != int(expected_pixels[sequence_index]):
            raise ValueError('Pixel-count preflight mismatch for %s.' % name)
        digest = hashlib.sha256()
        digest.update(memoryview(logits).cast('B'))
        logits_hashes.append(digest.hexdigest())
        target_digest = hashlib.sha256()
        contiguous_targets = np.ascontiguousarray(targets)
        target_digest.update(memoryview(contiguous_targets).cast('B'))
        target_hashes.append(target_digest.hexdigest())
        if not np.isfinite(logits).all():
            raise FloatingPointError('Non-finite stitched logits in %s.' % name)
        global_max = max(global_max, float(logits.max()))
        sequence_peaks = []
        for frame_logits, frame_target in zip(logits, targets):
            peaks, false_values = prepared_frame_events(frame_logits, frame_target)
            if peaks.size:
                sequence_peaks.append(peaks)
            topk.update(false_values)
            false_pixels_by_sequence[sequence_index] += false_values.size
        if sequence_peaks:
            sequence_peaks = np.concatenate(sequence_peaks)
        else:
            sequence_peaks = np.empty(0, dtype=np.float32)
        targets_by_sequence[sequence_index] = sequence_peaks.size
        peak_parts.append(sequence_peaks)
        peak_owner_parts.append(np.full(
            sequence_peaks.size, sequence_index, dtype=np.int32,
        ))
    if len(logits_hashes) != len(sequence_names):
        raise ValueError('Pass 1 did not yield every selected sequence.')
    all_peaks = np.concatenate(peak_parts) if peak_parts else np.empty(0, np.float32)
    all_owners = (
        np.concatenate(peak_owner_parts)
        if peak_owner_parts else np.empty(0, np.int32)
    )
    if not np.isfinite(global_max):
        raise FloatingPointError('No finite logits were observed.')
    return PassOneResult(
        target_peaks=all_peaks,
        target_peak_sequence_indices=all_owners,
        top_background_scores=topk.values(),
        targets_by_sequence=targets_by_sequence,
        false_region_pixels_by_sequence=false_pixels_by_sequence,
        logits_sha256_by_sequence=logits_hashes,
        targets_sha256_by_sequence=target_hashes,
        global_max_logit=global_max,
    )


def build_exact_thresholds(pass_one):
    maximum = np.float64(pass_one.global_max_logit)
    sentinel = np.nextafter(maximum, np.float64(np.inf))
    if not np.isfinite(sentinel) or not sentinel > maximum:
        raise FloatingPointError('Cannot construct a finite sentinel above global maximum.')
    values = np.concatenate((
        pass_one.target_peaks.astype(np.float64),
        pass_one.top_background_scores.astype(np.float64),
        np.asarray([0.0, sentinel], dtype=np.float64),
    ))
    thresholds = np.unique(values)
    if not np.isfinite(thresholds).all():
        raise FloatingPointError('Candidate thresholds must all be finite.')
    return thresholds, float(sentinel)


def frame_counts_at_thresholds(logits, target, thresholds):
    peaks, false_values = prepared_frame_events(logits, target)
    sorted_peaks = np.sort(peaks)
    sorted_false = np.sort(false_values)
    true_counts = sorted_peaks.size - np.searchsorted(
        sorted_peaks, thresholds, side='left',
    )
    false_counts = sorted_false.size - np.searchsorted(
        sorted_false, thresholds, side='left',
    )
    return false_counts.astype(np.int64), true_counts.astype(np.int64), peaks


@dataclass
class PassTwoResult:
    false_counts: np.ndarray
    true_counts: np.ndarray
    targets_by_sequence: np.ndarray
    pixels_by_sequence: np.ndarray
    logits_sha256_by_sequence: list
    targets_sha256_by_sequence: list


def collect_pass_two(
    sequence_stream, sequence_names, expected_pixels, expected_targets,
    expected_logits_hashes, expected_target_hashes, thresholds,
):
    false_counts = np.zeros(
        (thresholds.size, len(sequence_names)), dtype=np.int64,
    )
    true_counts = np.zeros_like(false_counts)
    targets_by_sequence = np.zeros(len(sequence_names), dtype=np.int64)
    for sequence_index, (name, logits, targets) in enumerate(sequence_stream):
        if name != sequence_names[sequence_index]:
            raise ValueError('Pass 2 sequence-order mismatch at index %d.' % sequence_index)
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        targets = np.asarray(targets)
        digest = hashlib.sha256()
        digest.update(memoryview(logits).cast('B'))
        observed_hash = digest.hexdigest()
        if observed_hash != expected_logits_hashes[sequence_index]:
            raise RuntimeError(
                'Two-pass bitwise replay mismatch in %s; exact-event claims '
                'would be invalid, so no output is written.' % name
            )
        target_digest = hashlib.sha256()
        contiguous_targets = np.ascontiguousarray(targets)
        target_digest.update(memoryview(contiguous_targets).cast('B'))
        if target_digest.hexdigest() != expected_target_hashes[sequence_index]:
            raise RuntimeError(
                'Two-pass target replay mismatch in %s; no output is written.'
                % name
            )
        observed_pixels = int(np.prod(logits.shape, dtype=np.int64))
        if observed_pixels != int(expected_pixels[sequence_index]):
            raise ValueError('Pixel-count mismatch in pass 2 for %s.' % name)
        sequence_targets = 0
        for frame_logits, frame_target in zip(logits, targets):
            frame_false, frame_true, peaks = frame_counts_at_thresholds(
                frame_logits, frame_target, thresholds,
            )
            false_counts[:, sequence_index] += frame_false
            true_counts[:, sequence_index] += frame_true
            sequence_targets += peaks.size
        targets_by_sequence[sequence_index] = sequence_targets
    if not np.array_equal(targets_by_sequence, expected_targets):
        raise RuntimeError('Target counts changed between pass 1 and pass 2.')
    return PassTwoResult(
        false_counts=false_counts,
        true_counts=true_counts,
        targets_by_sequence=targets_by_sequence,
        pixels_by_sequence=np.asarray(expected_pixels, dtype=np.int64),
        logits_sha256_by_sequence=list(expected_logits_hashes),
        targets_sha256_by_sequence=list(expected_target_hashes),
    )


def loader_options(workers, prefetch_factor):
    options = {'num_workers': workers}
    if workers > 0:
        options.update({
            'persistent_workers': True,
            'prefetch_factor': prefetch_factor,
        })
    return options


def merge_raw_logit_window(stitched_logits, window_logits, start_frame):
    """Max-stitch raw logits without imposing a zero probability floor."""
    stop = min(stitched_logits.shape[0], start_frame + window_logits.shape[0])
    length = stop - start_frame
    if length <= 0:
        raise ValueError('Raw-logit window does not overlap the destination.')
    destination = stitched_logits[start_frame:stop]
    np.maximum(destination, window_logits[:length], out=destination)


def stitched_sequence_stream(
    detector, sequence_names, sequence_datasets, seqlen,
    amp, workers, prefetch_factor,
):
    concatenated = torch.utils.data.ConcatDataset(sequence_datasets)
    dataloader = torch.utils.data.DataLoader(
        concatenated,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        **loader_options(workers, prefetch_factor)
    )
    iterator = iter(dataloader)
    with torch.inference_mode():
        for name, seq_dataset in zip(sequence_names, sequence_datasets):
            storage_length = sequence_storage_length(seq_dataset)
            stitched_logits = None
            stitched_targets = None
            for _ in range(len(seq_dataset)):
                images, _labels, centroids, first_end = next(iterator)
                images = images.float().cuda(non_blocking=True)
                first_frame = int(first_end[0].item())
                with torch.autocast(
                    device_type='cuda', dtype=torch.float16, enabled=amp,
                ):
                    features, window_logits = detector(images)
                del features, images
                if window_logits.shape[-2:] != centroids.shape[-2:]:
                    window_logits = F.interpolate(
                        window_logits,
                        size=tuple(centroids.shape[-2:]),
                        mode='bilinear',
                        align_corners=False,
                    )
                window_logits = window_logits.float().cpu().numpy()[0]
                window_targets = centroids.numpy()[0]
                if stitched_logits is None:
                    stitched_logits = np.full(
                        (storage_length,) + window_logits.shape[-2:],
                        -np.inf, dtype=np.float32,
                    )
                    stitched_targets = np.zeros(
                        (storage_length,) + window_targets.shape[-2:],
                        dtype=window_targets.dtype,
                    )
                stop = min(storage_length, first_frame + window_logits.shape[0])
                length = stop - first_frame
                merge_raw_logit_window(
                    stitched_logits, window_logits, first_frame,
                )
                stitched_targets[first_frame:stop] = window_targets[:length]
                del window_logits, window_targets, centroids, _labels
            if stitched_logits is None or not np.isfinite(stitched_logits).all():
                raise FloatingPointError('Incomplete or non-finite stitched logits in %s.' % name)
            yield name, stitched_logits, stitched_targets
            del stitched_logits, stitched_targets
            if getattr(detector, 'clear_validation_cache_each_sequence', False):
                torch.cuda.empty_cache()


def load_reference(path, args):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Reference exact-logit JSON has wrong schema version.')
    if payload.get('protocol', {}).get('name') != PROTOCOL_NAME:
        raise ValueError('Reference JSON is not an exact-logit protocol output.')
    identity = payload.get('run_identity', {})
    if identity.get('structure_variant') != 'none':
        raise ValueError('Reference JSON must be the B1 variant "none".')
    for field, expected in (
        ('seed', args.seed), ('condition', args.condition),
        ('dataset', args.dataset), ('sequence_length', args.seqlen),
    ):
        if identity.get(field) != expected:
            raise ValueError(
                'Reference %s mismatch: expected %r, found %r.'
                % (field, expected, identity.get(field))
            )
    workpoint = payload.get('workpoint_at_logit_zero')
    required = ('false_pixels', 'true_targets', 'total_targets', 'pixels')
    if not workpoint or any(field not in workpoint for field in required):
        raise ValueError('Reference JSON lacks its logit-zero workpoint.')
    integer_workpoint = {}
    for field in required:
        value = workpoint[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Reference workpoint %s must be a non-negative integer.' % field)
        integer_workpoint[field] = value
    if integer_workpoint['false_pixels'] > integer_workpoint['pixels']:
        raise ValueError('Reference false pixels exceed total pixels.')
    if integer_workpoint['true_targets'] > integer_workpoint['total_targets']:
        raise ValueError('Reference true targets exceed total targets.')
    reference_counts = payload.get('counts') or {}
    reference_npz = Path(reference_counts.get('npz_path', '')).expanduser()
    if not reference_npz.is_file():
        raise FileNotFoundError('Reference JSON points to a missing NPZ: %s' % reference_npz)
    observed_npz_sha = sha256_file(reference_npz)
    if observed_npz_sha != reference_counts.get('npz_sha256'):
        raise ValueError('Reference NPZ SHA256 does not match its JSON metadata.')
    return {
        'path': str(path),
        'sha256': sha256_file(path),
        'npz_path': str(reference_npz.resolve()),
        'npz_sha256': observed_npz_sha,
        'data_identity': payload.get('data_identity'),
        'workpoint': integer_workpoint,
    }


def compute_workpoint_summary(counts, thresholds, reference_workpoint):
    false_total = counts.false_counts.sum(axis=1)
    true_total = counts.true_counts.sum(axis=1)
    targets_total = int(counts.targets_by_sequence.sum())
    pixels_total = int(counts.pixels_by_sequence.sum())
    zero_indices = np.flatnonzero(thresholds == 0.0)
    if zero_indices.size != 1:
        raise RuntimeError('Threshold grid must contain exactly one logit zero.')
    zero_index = int(zero_indices[0])
    at_zero = {
        'threshold_logit': 0.0,
        'false_pixels': int(false_total[zero_index]),
        'true_targets': int(true_total[zero_index]),
        'total_targets': targets_total,
        'pixels': pixels_total,
        'fa': float(false_total[zero_index] / max(pixels_total, 1)),
        'pd': float(true_total[zero_index] / max(targets_total, 1)),
        'false_pixels_by_sequence': counts.false_counts[zero_index].tolist(),
        'true_targets_by_sequence': counts.true_counts[zero_index].tolist(),
    }
    reference = reference_workpoint if reference_workpoint is not None else at_zero
    fixed_fa_eligible = false_total <= int(reference['false_pixels'])
    if not np.any(fixed_fa_eligible):
        raise RuntimeError('No candidate threshold satisfies the reference Fa budget.')
    max_true = int(true_total[fixed_fa_eligible].max())
    fixed_fa_candidates = np.flatnonzero(fixed_fa_eligible & (true_total == max_true))
    fixed_fa_index = int(fixed_fa_candidates[np.argmin(false_total[fixed_fa_candidates])])
    fixed_pd_eligible = true_total >= int(reference['true_targets'])
    if np.any(fixed_pd_eligible):
        min_false = int(false_total[fixed_pd_eligible].min())
        fixed_pd_candidates = np.flatnonzero(fixed_pd_eligible & (false_total == min_false))
        fixed_pd_index = int(fixed_pd_candidates[np.argmax(true_total[fixed_pd_candidates])])
        fixed_pd = {
            'available': True,
            'threshold_logit': float(thresholds[fixed_pd_index]),
            'false_pixels': int(false_total[fixed_pd_index]),
            'true_targets': int(true_total[fixed_pd_index]),
            'fa': float(false_total[fixed_pd_index] / max(pixels_total, 1)),
            'pd': float(true_total[fixed_pd_index] / max(targets_total, 1)),
        }
    else:
        fixed_pd = {'available': False, 'reason': 'reference Pd is unattainable'}
    fixed_fa = {
        'available': True,
        'threshold_logit': float(thresholds[fixed_fa_index]),
        'false_pixels': int(false_total[fixed_fa_index]),
        'true_targets': int(true_total[fixed_fa_index]),
        'fa': float(false_total[fixed_fa_index] / max(pixels_total, 1)),
        'pd': float(true_total[fixed_fa_index] / max(targets_total, 1)),
    }
    return at_zero, fixed_fa, fixed_pd


def low_fa_pauc_from_counts(counts, cap):
    false_total = counts.false_counts.sum(axis=1).astype(np.float64)
    true_total = counts.true_counts.sum(axis=1).astype(np.float64)
    pixels = int(counts.pixels_by_sequence.sum())
    targets = int(counts.targets_by_sequence.sum())
    fa = false_total / max(pixels, 1)
    pd = true_total / max(targets, 1)
    order = np.argsort(fa)
    fa = fa[order]
    pd = pd[order]
    unique_fa, inverse = np.unique(fa, return_inverse=True)
    envelope = np.full(unique_fa.shape, -np.inf, dtype=np.float64)
    np.maximum.at(envelope, inverse, pd)
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
    return {
        'cap': float(cap),
        'normalized_pauc': float(np.trapz(
            envelope[keep], unique_fa[keep],
        ) / cap),
        'empirical_points_at_or_below_cap': int(np.count_nonzero(fa <= cap)),
        'first_empirical_fa_above_cap': (
            float(np.min(fa[fa > cap])) if np.any(fa > cap) else None
        ),
    }


def estimate_resources(retain_k, sequence_count, expected_target_count=4096):
    # During update, the old retained array, concatenation, and retained view
    # can coexist. This deliberately conservative bound is reported/refused.
    topk_working = int((2 * retain_k + 2_000_000) * 4)
    threshold_upper = int(retain_k + expected_target_count + 2)
    # FP, TP, target count, and pixel count as int64 threshold x sequence.
    count_matrix = int(threshold_upper * sequence_count * 4 * 8)
    return {
        'retain_k': int(retain_k),
        'conservative_threshold_count_upper_bound': threshold_upper,
        'estimated_topk_working_bytes': topk_working,
        'estimated_dense_count_matrix_bytes': count_matrix,
    }


def verify_outputs_absent(args):
    json_path = args.output_json.expanduser().resolve()
    npz_path = args.output_npz.expanduser().resolve()
    if json_path == npz_path:
        raise ValueError('JSON and NPZ destinations must differ.')
    if not args.overwrite:
        existing = [str(path) for path in (json_path, npz_path) if path.exists()]
        if existing:
            raise FileExistsError('Refusing to overwrite: %s' % ', '.join(existing))
    return json_path, npz_path


def main(args):
    if not math.isfinite(args.low_fa_cap) or not 0 < args.low_fa_cap < 1:
        raise ValueError('--low-fa-cap must be finite and lie in (0, 1).')
    if args.seqlen <= 0 or args.epoch <= 0:
        raise ValueError('Epoch and sequence length must be positive.')
    if args.test_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError('Invalid DataLoader worker/prefetch settings.')
    if args.max_topk_working_bytes <= 0 or args.max_count_matrix_bytes <= 0:
        raise ValueError('Memory safety limits must be positive.')
    expected_dataset = {
        'clean_val': 'NUDT-MIRSDT',
        'noise8_val': 'NUDT-MIRSDT-Noise8.0_FJY',
    }[args.condition]
    if args.dataset != expected_dataset:
        raise ValueError(
            'Condition %s requires dataset %s, found %s.'
            % (args.condition, expected_dataset, args.dataset)
        )
    output_json, output_npz = verify_outputs_absent(args)

    devices = parse_visible_devices(args.gpu)
    if len(devices) != 1:
        raise ValueError('Exact evaluator uses exactly one GPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
    torch.backends.cudnn.deterministic = bool(args.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    args.datapath = args.datapath.expanduser().resolve()
    args.sequence_list = args.sequence_list.expanduser().resolve()
    if not args.datapath.is_dir() or not args.sequence_list.is_file():
        raise FileNotFoundError('Dataset root or sequence list does not exist.')
    experiment_dir = resolve_experiment_dir(args.logpath, args.log_dir)
    checkpoint_path = experiment_dir / 'checkpoints' / (
        'epoch_%d_model.pth' % args.epoch
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Missing checkpoint: %s' % checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
    model_name, model_path = resolve_model_name(checkpoint, experiment_dir)
    checkpoint_epoch = int(checkpoint.get('epoch', -1)) + 1
    if checkpoint_epoch != args.epoch:
        raise ValueError('Checkpoint epoch metadata mismatch.')
    model_config = dict(checkpoint.get('model_config', {}))
    observed_variant = model_config.get('structure_variant', 'none')
    if observed_variant != args.structure_variant:
        raise ValueError(
            'Checkpoint variant %r does not match --structure-variant %r.'
            % (observed_variant, args.structure_variant)
        )
    training_provenance = validate_training_provenance(
        experiment_dir, args, model_name,
    )

    dataset = TestIRSeqDataLoader(
        args.dataset,
        data_root=str(args.datapath),
        seq_len=args.seqlen,
        cat_len=int(args.seqlen * 0.1),
        transform=None,
        load_annotations=True,
        split='val',
        sequence_list_file=str(args.sequence_list),
    )
    sequence_names = list(dataset.seq_names)
    sequence_datasets = [dataset[index] for index in range(len(dataset))]
    data_identity = build_data_identity(
        args.dataset, sequence_names, sequence_datasets, args.sequence_list,
    )
    pixels_by_sequence = np.asarray(
        [item['pixels'] for item in data_identity['sequences']], dtype=np.int64,
    )
    total_pixels = int(pixels_by_sequence.sum())
    cap_k = int(math.floor(args.low_fa_cap * total_pixels)) + 1
    reference = load_reference(args.reference_json, args)
    if reference is not None:
        reference_identity = reference.get('data_identity') or {}
        for field in ('content_sha256', 'sequence_list_sha256', 'total_pixels'):
            if reference_identity.get(field) != data_identity.get(field):
                raise ValueError('Reference data identity mismatch for %s.' % field)
    if args.reference_retention_mode == 'full-reference-grid':
        if reference is None:
            raise ValueError('full-reference-grid requires --reference-json.')
        retain_k = max(cap_k, int(reference['workpoint']['false_pixels']) + 1)
    else:
        retain_k = cap_k
    resource_estimate = estimate_resources(retain_k, len(sequence_names))
    print('Exact-logit preflight resource estimate: %s' % json.dumps(resource_estimate))
    if resource_estimate['estimated_topk_working_bytes'] > args.max_topk_working_bytes:
        raise MemoryError(
            'Estimated top-K working memory %d exceeds safety limit %d. '
            'Use target-events (exact for the declared workpoints), raise the '
            'explicit limit, or move the run to a larger host.'
            % (
                resource_estimate['estimated_topk_working_bytes'],
                args.max_topk_working_bytes,
            )
        )
    if resource_estimate['estimated_dense_count_matrix_bytes'] > args.max_count_matrix_bytes:
        raise MemoryError(
            'Estimated threshold x sequence matrices %d exceed safety limit %d. '
            'The requested full transition grid is unsafe; target-events is '
            'the exact, memory-bounded workpoint mode.'
            % (
                resource_estimate['estimated_dense_count_matrix_bytes'],
                args.max_count_matrix_bytes,
            )
        )

    sys.path.insert(0, str(experiment_dir))
    model_module = importlib.import_module(model_name)
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
        clean_model_state_dict(checkpoint['model_state_dict']), strict=True,
    )
    del checkpoint
    detector = detector.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def make_stream():
        return stitched_sequence_stream(
            detector, sequence_names, sequence_datasets, args.seqlen,
            args.amp, args.test_workers, args.prefetch_factor,
        )

    start = time.time()
    pass_one_start = time.time()
    pass_one = collect_pass_one(
        make_stream(), sequence_names, pixels_by_sequence, retain_k,
    )
    pass_one_seconds = time.time() - pass_one_start
    if reference is not None:
        if reference['workpoint']['pixels'] != total_pixels:
            raise ValueError('Reference workpoint pixel denominator mismatch.')
        if reference['workpoint']['total_targets'] != int(
            pass_one.targets_by_sequence.sum()
        ):
            raise ValueError('Reference workpoint target denominator mismatch.')
    thresholds, sentinel = build_exact_thresholds(pass_one)
    actual_matrix_bytes = int(thresholds.size * len(sequence_names) * 4 * 8)
    if actual_matrix_bytes > args.max_count_matrix_bytes:
        raise MemoryError(
            'Actual threshold x sequence matrices require an estimated %d bytes, '
            'above the explicit %d-byte limit.'
            % (actual_matrix_bytes, args.max_count_matrix_bytes)
        )
    pass_two_start = time.time()
    pass_two = collect_pass_two(
        make_stream(), sequence_names, pixels_by_sequence,
        pass_one.targets_by_sequence, pass_one.logits_sha256_by_sequence,
        pass_one.targets_sha256_by_sequence, thresholds,
    )
    pass_two_seconds = time.time() - pass_two_start
    elapsed_seconds = time.time() - start

    reference_workpoint = reference['workpoint'] if reference is not None else None
    at_zero, fixed_fa, fixed_pd = compute_workpoint_summary(
        pass_two, thresholds, reference_workpoint,
    )
    low_fa_summary = low_fa_pauc_from_counts(
        pass_two, args.low_fa_cap,
    )
    if reference is None:
        reference_workpoint = {
            field: at_zero[field]
            for field in ('false_pixels', 'true_targets', 'total_targets', 'pixels')
        }

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
    npz_sha = sha256_file(output_npz)
    boundary = (
        float(pass_one.top_background_scores[0])
        if pass_one.top_background_scores.size else None
    )
    false_pixels_at_boundary = None
    if boundary is not None:
        boundary_index = int(np.searchsorted(thresholds, boundary))
        false_total = pass_two.false_counts.sum(axis=1)
        # This count can exceed requested K when the Kth score is tied: pass 2
        # counts every global false-region pixel satisfying score >= boundary.
        false_pixels_at_boundary = int(false_total[boundary_index])

    dependency_sha = dependency_hashes(experiment_dir, model_path)
    payload = {
        'schema_version': SCHEMA_VERSION,
        'protocol': {
            'name': PROTOCOL_NAME,
            'status': 'post_hoc_protocol_amendment',
            'coexists_with': 'preregistered 109-point sigmoid-probability grid',
            'replaces_preregistered_results': False,
            'score_domain': 'raw logits',
            'window_stitch_rule': 'elementwise maximum before scoring',
            'threshold_comparison': 'score >= threshold',
            'passes': 2,
            'prediction_maps_persisted': False,
            'metric_coverage': {
                'low_fa_curve_through_cap': 'exact empirical event grid',
                'pd_at_reference_fa': (
                    'exact discrete maximum; every TP transition is represented '
                    'by all target-peak thresholds'
                ),
                'fa_at_reference_pd': (
                    'exact discrete minimum; every TP transition is represented '
                    'by all target-peak thresholds'
                ),
                'full_roc_auc': 'not covered',
                'high_fa_background_transitions': (
                    'covered through F_ref'
                    if args.reference_retention_mode == 'full-reference-grid'
                    else 'not retained; not needed for the declared exact workpoints'
                ),
            },
        },
        'run_identity': {
            'condition': args.condition,
            'dataset': args.dataset,
            'datapath': str(args.datapath),
            'model': model_name,
            'structure_variant': args.structure_variant,
            'seed': args.seed,
            'checkpoint_epoch': checkpoint_epoch,
            'sequence_length': args.seqlen,
            'log_dir': args.log_dir,
        },
        'checkpoint': {
            'path': str(checkpoint_path.resolve()),
            'sha256': checkpoint_sha,
        },
        'training_provenance': training_provenance,
        'dependencies_sha256': dependency_sha,
        'data_identity': data_identity,
        'reference': reference,
        'reference_workpoint': reference_workpoint,
        'retention': {
            'mode': args.reference_retention_mode,
            'low_fa_cap': args.low_fa_cap,
            'cap_k_floor_plus_one': cap_k,
            'requested_k': retain_k,
            'retained_background_scores': int(pass_one.top_background_scores.size),
            'false_region_scores_seen': int(
                pass_one.false_region_pixels_by_sequence.sum()
            ),
            'topk_boundary_score': boundary,
            'false_pixels_at_boundary_including_all_ties': (
                false_pixels_at_boundary
            ),
            'resource_estimate_before_inference': resource_estimate,
            'actual_dense_count_matrix_bytes': actual_matrix_bytes,
            'max_topk_working_bytes': args.max_topk_working_bytes,
            'max_count_matrix_bytes': args.max_count_matrix_bytes,
        },
        'thresholds': {
            'count': int(thresholds.size),
            'minimum': float(thresholds[0]),
            'maximum_finite_sentinel': sentinel,
            'global_max_raw_logit': pass_one.global_max_logit,
            'includes_logit_zero': True,
            'sources': [
                'all target peaks', 'retained top-K false-region logits',
                'logit zero', 'finite sentinel strictly above global maximum',
            ],
        },
        'counts': {
            'npz_path': str(output_npz),
            'npz_sha256': npz_sha,
            'axes': ['threshold', 'sequence'],
            'shape': [int(thresholds.size), len(sequence_names)],
            'dtype': 'int64',
            'arrays': [
                'false_pixels_by_threshold_sequence',
                'true_targets_by_threshold_sequence',
                'total_targets_by_threshold_sequence',
                'pixel_count_by_threshold_sequence',
            ],
            'targets_by_sequence': pass_two.targets_by_sequence.tolist(),
            'pixels_by_sequence': pass_two.pixels_by_sequence.tolist(),
            'false_region_pixels_by_sequence': (
                pass_one.false_region_pixels_by_sequence.tolist()
            ),
        },
        'workpoint_at_logit_zero': at_zero,
        'low_fa': low_fa_summary,
        'pd_at_reference_fa': fixed_fa,
        'fa_at_reference_pd': fixed_pd,
        'two_pass_replay': {
            'required_match': 'bitwise SHA256 of every stitched float32 logit sequence',
            'matched': True,
            'logits_sha256_by_sequence': dict(zip(
                sequence_names, pass_one.logits_sha256_by_sequence,
            )),
            'targets_sha256_by_sequence': dict(zip(
                sequence_names, pass_one.targets_sha256_by_sequence,
            )),
        },
        'inference': {
            'amp': bool(args.amp),
            'eval_chunk_rows': args.eval_chunk_rows,
            'test_workers': args.test_workers,
            'prefetch_factor': args.prefetch_factor,
            'cudnn_enabled': bool(torch.backends.cudnn.enabled),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'deterministic_algorithms_enabled': bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            'pass_one_seconds': pass_one_seconds,
            'pass_two_seconds': pass_two_seconds,
            'elapsed_seconds': elapsed_seconds,
            'cuda_peak_allocated_gib': (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            ),
            'cuda_peak_reserved_gib': (
                torch.cuda.max_memory_reserved() / (1024 ** 3)
            ),
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
    print('Exact-logit NPZ: %s' % output_npz)
    print('Exact-logit JSON: %s' % output_json)


if __name__ == '__main__':
    main(parse_args())
