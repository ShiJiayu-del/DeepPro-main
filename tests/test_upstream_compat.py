"""CPU regression tests for the opt-in official DeepPro loader profile."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

import train
from data_utils.TrainDataLoader import (
    NEAREST_RESAMPLE,
    TrainIRSeqDataLoader,
    TrainSeqDataLoader,
)


class UpstreamCompatibilityTests(unittest.TestCase):
    def _base_dataset(self, image_path, mask_path, upstream_compat=False):
        return TrainSeqDataLoader(
            'NUDT-MIRSDT',
            str(Path(image_path).parent),
            [[(str(image_path), str(mask_path))]],
            [1.0],
            seq_len=1,
            sample_rate=1.0,
            patch_size=None,
            upstream_compat=upstream_compat,
        )

    def test_nudt_window_count_and_terminal_frame_match_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / 'Sequence1'
            image_root = sequence / 'images'
            mask_root = sequence / 'masks'
            image_root.mkdir(parents=True)
            mask_root.mkdir()
            for frame_index in range(100):
                name = '%03d.png' % frame_index
                (image_root / name).touch()
                (mask_root / name).touch()
            (root / 'train.txt').write_text(
                'Sequence1/Mix/001.mat\n', encoding='utf-8'
            )

            current = TrainIRSeqDataLoader(
                'NUDT-MIRSDT', str(root), seq_len=40, sample_rate=1.0
            )
            upstream = TrainIRSeqDataLoader(
                'NUDT-MIRSDT', str(root), seq_len=40, sample_rate=1.0,
                upstream_compat=True,
            )

            self.assertEqual(len(current.samplelist), 97)
            self.assertEqual(len(upstream.samplelist), 96)
            self.assertTrue(current.samplelist[-1][-1][0].endswith('099.png'))
            self.assertTrue(upstream.samplelist[-1][-1][0].endswith('098.png'))

    def test_upstream_mask_resize_uses_default_pillow_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'image.png'
            mask_path = root / 'mask.png'
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_path)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[2, 2] = 255
            Image.fromarray(mask).save(mask_path)

            current = self._base_dataset(image_path, mask_path)
            upstream = self._base_dataset(
                image_path, mask_path, upstream_compat=True
            )
            _, current_label = current.get_image_label(image_path, mask_path)
            _, upstream_label = upstream.get_image_label(image_path, mask_path)

            with Image.open(mask_path) as mask_image:
                expected_current = np.asarray(mask_image.resize(
                    [256, 256], resample=NEAREST_RESAMPLE
                )) > 0
                expected_upstream = np.asarray(
                    mask_image.resize([256, 256])
                ) > 0
            self.assertTrue(np.array_equal(current_label[0] > 0, expected_current))
            self.assertTrue(np.array_equal(
                upstream_label[0] > 0, expected_upstream
            ))
            self.assertFalse(np.array_equal(current_label, upstream_label))

    def test_crop_upper_bound_is_127_upstream_and_128_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'image.png'
            mask_path = root / 'mask.png'
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(
                image_path
            )
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(
                mask_path
            )

            observed_highs = []

            def choose_last(_low, high, _size):
                observed_highs.append(high)
                return torch.tensor([high - 1])

            for upstream_compat, expected_high in ((True, 128), (False, 129)):
                dataset = self._base_dataset(
                    image_path, mask_path, upstream_compat=upstream_compat
                )
                dataset.patch_size = 128
                observed_highs.clear()
                with mock.patch('numpy.random.choice', return_value=0), mock.patch(
                    'random.uniform', return_value=1.0
                ), mock.patch('torch.randint', side_effect=choose_last):
                    images, labels = dataset.sample_sequence(0)
                self.assertEqual(observed_highs, [expected_high, expected_high])
                self.assertEqual(tuple(images.shape), (1, 1, 128, 128))
                self.assertEqual(tuple(labels.shape), (1, 128, 128))

    def test_skip_validation_returns_before_dataset_or_path_access(self):
        args = SimpleNamespace(
            skip_inprocess_validation=1,
            dataset='NUDT-MIRSDT',
            val_sequence_list='/path/that/must/not/be/read.txt',
            val_workers=0,
            prefetch_factor=2,
        )
        runtime = SimpleNamespace(rank=0, world_size=1)
        with mock.patch.object(
            train,
            'TestIRSeqDataLoader',
            side_effect=AssertionError('validation dataset must not be built'),
        ) as constructor, mock.patch.object(
            train.torch.utils.data,
            'DataLoader',
            side_effect=AssertionError('validation loader must not be built'),
        ) as data_loader:
            result = train.build_validation_data(
                args, runtime, '/path/that/must/not/be/read', 40
            )
        self.assertEqual(result, (None, None, None))
        constructor.assert_not_called()
        data_loader.assert_not_called()

    def test_upstream_compat_cli_is_opt_in(self):
        with mock.patch.object(sys, 'argv', ['train.py']):
            self.assertEqual(train.parse_args().upstream_compat, 0)
        with mock.patch.object(
            sys, 'argv', ['train.py', '--upstream_compat', '1']
        ):
            self.assertEqual(train.parse_args().upstream_compat, 1)


if __name__ == '__main__':
    unittest.main()
