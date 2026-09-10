"""CPU tests for immutable per-run training source snapshots."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import train


class TrainingSourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(train.ROOT_DIR).resolve()
        self.model_source = (
            self.repo_root / 'networks' / 'models' / 'DeepPro-Plus_BCTPro.py'
        )
        self.adapter_source = (
            self.repo_root / 'networks' / 'layers' / 'bc_tpro_adapter.py'
        )

    def test_snapshot_contains_required_tree_and_root_import_files(self):
        expected_relative_paths = (
            'train.py',
            'test.py',
            'ShootingRules.py',
            'write_results.py',
            'data_utils/TrainDataLoader.py',
            'data_utils/TestDataLoader.py',
            'data_utils/loader_utils.py',
            'networks/layers/basic.py',
            'networks/layers/TPro.py',
            'networks/models/DeepPro-Plus_BCTPro.py',
            'networks/layers/bc_tpro_adapter.py',
            'networks/losses/segmentation_losses.py',
        )
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            with mock.patch.object(
                train.shutil, 'copy2', wraps=train.shutil.copy2
            ) as copy2:
                train.snapshot_training_sources(
                    experiment_dir,
                    self.model_source,
                    adapter_sources=(self.adapter_source,),
                )

            self.assertEqual(copy2.call_count, len(expected_relative_paths) + 3)
            for relative_path in expected_relative_paths:
                source = self.repo_root / relative_path
                destination = (
                    experiment_dir / 'source_snapshot' / relative_path
                )
                self.assertEqual(destination.read_bytes(), source.read_bytes())
            for source in (
                self.model_source,
                self.adapter_source,
                self.repo_root / 'networks' / 'losses'
                / 'segmentation_losses.py',
            ):
                self.assertEqual(
                    (experiment_dir / source.name).read_bytes(),
                    source.read_bytes(),
                )

    def test_identical_existing_snapshot_is_reused_without_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            train.snapshot_training_sources(
                experiment_dir,
                self.model_source,
                adapter_sources=(self.adapter_source,),
            )
            with mock.patch.object(
                train.shutil, 'copy2', wraps=train.shutil.copy2
            ) as copy2:
                train.snapshot_training_sources(
                    experiment_dir,
                    self.model_source,
                    adapter_sources=(self.adapter_source,),
                )
            copy2.assert_not_called()

    def test_changed_existing_snapshot_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            train.snapshot_training_sources(
                experiment_dir,
                self.model_source,
                adapter_sources=(self.adapter_source,),
            )
            destination = (
                experiment_dir / 'source_snapshot' / 'networks' / 'layers'
                / 'TPro.py'
            )
            destination.write_text('different\n', encoding='utf-8')

            with mock.patch.object(
                train.shutil, 'copy2', wraps=train.shutil.copy2
            ) as copy2, self.assertRaisesRegex(
                RuntimeError, 'refusing overwrite'
            ):
                train.snapshot_training_sources(
                    experiment_dir,
                    self.model_source,
                    adapter_sources=(self.adapter_source,),
                )
            copy2.assert_not_called()
            self.assertEqual(destination.read_text(encoding='utf-8'), 'different\n')


if __name__ == '__main__':
    unittest.main()
