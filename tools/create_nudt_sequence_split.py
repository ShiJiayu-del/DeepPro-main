#!/usr/bin/env python3
"""Create a fixed, auditable 64/16 NUDT-MIRSDT sequence split."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.loader_utils import read_sequence_names


DEFAULT_SEED = 20260908
EXPECTED_SOURCE_SEQUENCES = 80
VALIDATION_SEQUENCES = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-list', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + '.', suffix='.tmp'
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    source_list = (
        args.source_list.expanduser().resolve()
        if args.source_list is not None
        else data_root / 'train.txt'
    )
    source_names = read_sequence_names(source_list, data_root)
    if len(source_names) != EXPECTED_SOURCE_SEQUENCES:
        raise ValueError(
            'Expected %d source sequences, found %d in %s.'
            % (EXPECTED_SOURCE_SEQUENCES, len(source_names), source_list)
        )

    generator = np.random.Generator(np.random.PCG64(args.seed))
    validation_indices = set(
        generator.choice(
            len(source_names),
            size=VALIDATION_SEQUENCES,
            replace=False,
        ).tolist()
    )
    train_names = [
        name for index, name in enumerate(source_names)
        if index not in validation_indices
    ]
    validation_names = [
        name for index, name in enumerate(source_names)
        if index in validation_indices
    ]
    if set(train_names) & set(validation_names):
        raise RuntimeError('Generated train and validation sequences overlap.')
    if set(train_names) | set(validation_names) != set(source_names):
        raise RuntimeError('Generated split does not cover the source sequences.')

    output_dir = args.output_dir.expanduser().resolve()
    train_path = output_dir / 'train_sequences.txt'
    validation_path = output_dir / 'val_sequences.txt'
    manifest_path = output_dir / 'split_manifest.json'
    atomic_write(train_path, ''.join(name + '\n' for name in train_names))
    atomic_write(
        validation_path,
        ''.join(name + '\n' for name in validation_names),
    )
    manifest = {
        'schema_version': 1,
        'method': 'numpy.PCG64.choice_without_replacement',
        'seed': args.seed,
        'source_list': str(source_list),
        'source_sequence_count': len(source_names),
        'source_sha256': sha256(source_list),
        'train_list': train_path.name,
        'train_sequence_count': len(train_names),
        'train_sha256': sha256(train_path),
        'validation_list': validation_path.name,
        'validation_sequence_count': len(validation_names),
        'validation_sha256': sha256(validation_path),
    }
    atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
