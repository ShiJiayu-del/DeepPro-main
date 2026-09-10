#!/usr/bin/env python3
"""Fail-closed semantic checks for the Noise8 BC-TPro stage-1 protocol."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np


EXPECTED_DATASET = 'NUDT-MIRSDT-Noise8.0_FJY'
MODERNIZED_PROFILE = 'modernized'
UPSTREAM_PROFILE = 'upstream8fa1a68_fp32'
PROFILES = (MODERNIZED_PROFILE, UPSTREAM_PROFILE)
EXPECTED_COLUMNS = [
    'run_id', 'wave', 'model', 'structure_variant', 'seed', 'gpu', 'log_dir'
]
VARIANTS = {
    1: ('b1_none', 'none', 'B1'),
    2: ('c0_temporal_control', 'temporal_control', 'C0'),
    3: ('c1_center_multiscale', 'center_multiscale', 'C1'),
    4: ('c2_center_ring', 'center_ring', 'C2'),
}
SEED_GPU = {47: 0, 49: 1, 51: 2}


def fail(message):
    raise ValueError(message)


def relative_parts(raw, source):
    normalized = raw.strip().replace('\\', '/')
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or '..' in path.parts:
        fail('Unsafe or empty entry in %s: %r' % (source, raw))
    parts = tuple(part for part in path.parts if part not in {'', '.'})
    if not parts:
        fail('Empty entry in %s.' % source)
    return parts


def read_official_list(path, expected_entries, expected_sequences):
    lines = [line for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    if len(lines) != expected_entries:
        fail('%s must contain %d entries, found %d.'
             % (path, expected_entries, len(lines)))
    names = [relative_parts(line, path)[0] for line in lines]
    counts = Counter(names)
    if len(counts) != expected_sequences:
        fail('%s must identify %d sequences, found %d.'
             % (path, expected_sequences, len(counts)))
    if set(counts.values()) != {100}:
        fail('%s must contain exactly 100 entries per sequence.' % path)
    return list(dict.fromkeys(names))


def read_split(path, expected_count):
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    if len(lines) != expected_count or len(set(lines)) != expected_count:
        fail('%s must contain %d unique sequence names.' % (path, expected_count))
    for line in lines:
        parts = relative_parts(line, path)
        if len(parts) != 1:
            fail('%s must contain bare sequence names only: %s' % (path, line))
    return lines


def validate_split(data_root, train_list, val_list, split_manifest):
    official_train_path = (data_root / 'train.txt').resolve(strict=True)
    official_test_path = (data_root / 'test.txt').resolve(strict=True)
    for split_path in (train_list, val_list):
        if split_path.resolve(strict=True) == official_test_path:
            fail('Official test.txt is forbidden as a train/validation split.')

    official_train = read_official_list(official_train_path, 8000, 80)
    official_test = read_official_list(official_test_path, 2000, 20)
    if set(official_train) & set(official_test):
        fail('Official train.txt and test.txt sequence sets overlap.')

    train_names = read_split(train_list, 64)
    val_names = read_split(val_list, 16)
    if set(train_names) & set(val_names):
        fail('The fixed 64/16 split overlaps.')
    if set(train_names) | set(val_names) != set(official_train):
        fail('The fixed 64/16 split must partition official Noise8 train.txt.')
    if (set(train_names) | set(val_names)) & set(official_test):
        fail('The fixed split contains an official test sequence.')

    payload = json.loads(split_manifest.read_text(encoding='utf-8'))
    expected = {
        'schema_version': 2,
        'method': 'numpy.PCG64.choice_without_replacement',
        'seed': 20260908,
        'source_list': str(official_train_path),
        'source_entry_count': 8000,
        'source_sequence_count': 80,
        'train_list': train_list.name,
        'train_sequence_count': 64,
        'validation_list': val_list.name,
        'validation_sequence_count': 16,
        'official_test_list': str(official_test_path),
        'official_test_sequence_count': 20,
        'official_test_usage': 'metadata-only isolation check; never train/evaluate',
    }
    if payload != expected:
        fail('split_manifest.json does not exactly match the registered protocol.')

    generator = np.random.Generator(np.random.PCG64(payload['seed']))
    val_indices = set(generator.choice(80, size=16, replace=False).tolist())
    expected_train = [name for index, name in enumerate(official_train)
                      if index not in val_indices]
    expected_val = [name for index, name in enumerate(official_train)
                    if index in val_indices]
    if train_names != expected_train or val_names != expected_val:
        fail('Split contents do not reproduce the registered PCG64 selection.')

    for name in official_train:
        images = data_root / name / 'images'
        masks = data_root / name / 'masks'
        if not images.is_dir() or not masks.is_dir():
            fail('Missing Noise8 training payload directories for %s.' % name)


def expected_manifest_rows(profile=MODERNIZED_PROFILE):
    if profile not in PROFILES:
        fail('Unknown Noise8 protocol profile: %s' % profile)
    log_prefix = (
        'Upstream8fa1a68-FP32-' if profile == UPSTREAM_PROFILE else ''
    )
    rows = []
    for wave, (prefix, variant, label) in VARIANTS.items():
        for seed, gpu in SEED_GPU.items():
            rows.append({
                'run_id': '%s_seed%d' % (prefix, seed),
                'wave': str(wave),
                'model': 'DeepPro-Plus_BCTPro',
                'structure_variant': variant,
                'seed': str(seed),
                'gpu': str(gpu),
                'log_dir': (
                    '2026-09-09/%s__%sSoftIoU-BCTPro-%s_seed%d_E32'
                    % (EXPECTED_DATASET, log_prefix, label, seed)
                ),
            })
    return rows


def validate_manifest(path, profile=MODERNIZED_PROFILE):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != EXPECTED_COLUMNS:
            fail('Unexpected manifest columns: %r' % reader.fieldnames)
        rows = list(reader)
    if len(rows) != 12:
        fail('Manifest must contain exactly 12 runs, found %d.' % len(rows))

    expected_rows = expected_manifest_rows(profile)
    if rows != expected_rows:
        fail('Manifest rows/order differ from the registered 4x3 design.')
    if any('pretrain' in value.lower() or 'test' in value.lower()
           for row in rows for value in row.values()):
        fail('Manifest contains a forbidden pretrained/test token.')


def validate_protocol_config(path, profile):
    if profile == MODERNIZED_PROFILE:
        if path is not None:
            fail('The modernized profile does not use an upstream protocol config.')
        return
    if path is None:
        fail('--protocol-config is required for the upstream profile.')
    payload = json.loads(path.expanduser().resolve(strict=True).read_text(
        encoding='utf-8'
    ))
    expected = {
        'schema_version': 1,
        'profile': UPSTREAM_PROFILE,
        'upstream_repository': 'https://github.com/TinaLRJ/DeepPro.git',
        'upstream_commit': '8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28',
        'training_precision': 'fp32',
        'evaluation_precision': 'fp32',
        'upstream_compat': True,
        'metric_amendment': 'OFFICIAL_METRIC_AMENDMENT_2026-09-10.md',
        'detection_metric_contract': {
            'operating_threshold': 0.5,
            'reported_detection_metrics': [
                'Pd_at_threshold_0.5',
                'Fa_at_threshold_0.5',
                'Pd-Fa_AUC_27_thresholds',
            ],
            'raw_logit_and_dense_grid_role': 'supplemental_only',
        },
        'aligned_semantics': [
            'Pillow-default mask resize followed by positive-pixel binarization',
            'legacy temporal-window endpoints excluding the final frame',
            'legacy crop upper bound size-minus-patch-minus-one',
        ],
        'retained_research_safeguards': [
            'deterministic seeds 47,49,51',
            'fixed official-train-derived 64/16 internal split',
            'scratch-only initialization',
            'fixed epoch-32 checkpoint',
            'official test excluded from Stage1',
        ],
    }
    if payload != expected:
        fail('Upstream protocol config does not exactly match the registered profile.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--expected-data-root', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--train-list', type=Path, required=True)
    parser.add_argument('--val-list', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, default=MODERNIZED_PROFILE)
    parser.add_argument('--protocol-config', type=Path)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve(strict=True)
    expected_root = args.expected_data_root.expanduser().resolve(strict=True)
    if data_root != expected_root:
        fail('Noise8 data root mismatch: %s != %s' % (data_root, expected_root))
    if args.dataset != EXPECTED_DATASET:
        fail('Dataset name must be exactly %s.' % EXPECTED_DATASET)
    validate_split(
        data_root,
        args.train_list.expanduser().resolve(strict=True),
        args.val_list.expanduser().resolve(strict=True),
        args.split_manifest.expanduser().resolve(strict=True),
    )
    validate_manifest(
        args.manifest.expanduser().resolve(strict=True), args.profile,
    )
    validate_protocol_config(args.protocol_config, args.profile)
    print(
        'VALID Noise8 protocol profile=%s: official_train=80 split=64/16 '
        'official_test=20 isolated runs=12 scratch_only=launcher-enforced'
        % args.profile
    )


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit('INVALID Noise8 protocol: %s' % error)
