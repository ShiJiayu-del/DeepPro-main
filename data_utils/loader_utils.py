from pathlib import Path, PurePosixPath


SATVIDEO_V1_DATASET = 'SatVideoIRSDT_v1'
SATVIDEO_V1_TRAIN_MEAN = 82.20451526467026
SATVIDEO_V1_TRAIN_STD = 50.753589902516666


def read_sequence_names(sequence_list_file, sequence_root):
    """Read unique sequence names from a sequence- or frame-level list."""
    list_path = Path(sequence_list_file).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError('No such sequence list: %s.' % list_path)

    sequence_names = []
    seen = set()
    for line_number, raw_line in enumerate(
        list_path.read_text(encoding='utf-8').splitlines(), start=1
    ):
        entry = raw_line.strip()
        if not entry:
            continue
        normalized = entry.replace('\\', '/')
        path = PurePosixPath(normalized)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(
                'Sequence-list entries must be relative paths; %s line %d: %s'
                % (list_path, line_number, entry)
            )
        parts = tuple(part for part in path.parts if part not in {'', '.'})
        if not parts:
            continue
        sequence_name = parts[0]
        if sequence_name not in seen:
            seen.add(sequence_name)
            sequence_names.append(sequence_name)

    if not sequence_names:
        raise ValueError('Sequence list is empty: %s.' % list_path)

    root = Path(sequence_root).expanduser().resolve()
    missing = [
        name for name in sequence_names if not (root / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            'Sequence list %s references missing directories under %s: %s'
            % (list_path, root, ', '.join(missing[:10]))
        )
    return sequence_names


def discover_split_sequences(data_root, split):
    """Discover ``<split>/<sequence>/{img,mask}`` sequence directories."""
    split_root = Path(data_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            'Dataset split directory does not exist: %s' % split_root
        )

    sequence_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise ValueError('No sequence directories found in %s.' % split_root)

    invalid = [
        path.name
        for path in sequence_dirs
        if not (path / 'img').is_dir() or not (path / 'mask').is_dir()
    ]
    if invalid:
        raise ValueError(
            'Sequences in %s must contain img and mask directories: %s'
            % (split_root, ', '.join(invalid[:10]))
        )
    return [path.name for path in sequence_dirs]


def validate_frame_pairs(
    image_root,
    label_root,
    images,
    labels,
    sequence_name,
    minimum_frames=1,
):
    if not Path(image_root).is_dir():
        raise FileNotFoundError('Image directory does not exist: %s' % image_root)
    if not Path(label_root).is_dir():
        raise FileNotFoundError('Label directory does not exist: %s' % label_root)
    if len(images) != len(labels):
        raise ValueError(
            '%s has %d images but %d labels.'
            % (sequence_name, len(images), len(labels))
        )
    if len(images) < minimum_frames:
        raise ValueError(
            '%s has %d frames; at least %d are required.'
            % (sequence_name, len(images), minimum_frames)
        )

    image_stems = [Path(name).stem for name in images]
    label_stems = [Path(name).stem for name in labels]
    if image_stems != label_stems:
        mismatch_index = next(
            index
            for index, pair in enumerate(zip(image_stems, label_stems))
            if pair[0] != pair[1]
        )
        raise ValueError(
            '%s image/label names do not match at index %d: %s versus %s.'
            % (
                sequence_name,
                mismatch_index,
                images[mismatch_index],
                labels[mismatch_index],
            )
        )
