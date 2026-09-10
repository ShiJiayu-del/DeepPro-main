import argparse
import csv
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    'analyze_bc_tpro_noise8_stage1',
    TOOLS / 'analyze_bc_tpro_noise8_stage1.py',
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
common = analysis.common


def namespace_line(values):
    return repr(argparse.Namespace(**values))


class Noise8Stage1AnalyzerTests(unittest.TestCase):
    def test_expected_manifest_is_exact_12_run_protocol(self):
        rows = analysis.expected_manifest_rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['run_id'] for row in rows}), 12)
        self.assertEqual(
            {(row['structure_variant'], int(row['seed'])) for row in rows},
            {(variant, seed) for variant in analysis.VARIANTS for seed in analysis.SEEDS},
        )
        self.assertTrue(all(analysis.DATASET in row['log_dir'] for row in rows))

    def test_official_metric_contract_is_exact(self):
        contract = analysis.official_metric_contract()
        self.assertEqual(contract['operating_threshold'], 0.5)
        self.assertEqual(len(contract['auc_thresholds']), 27)
        self.assertEqual(
            contract['reported_detection_metrics'],
            [
                'Pd_at_threshold_0.5',
                'Fa_at_threshold_0.5',
                'Pd-Fa_AUC_27_thresholds',
            ],
        )
        self.assertNotIn('f1', repr(contract).lower())

    def test_manifest_rejects_identity_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.tsv'
            rows = analysis.expected_manifest_rows()
            rows[0] = dict(rows[0], seed='48')
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0]), delimiter='\t',
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'frozen 12-run'):
                analysis.load_manifest(path)

    def test_provisional_gate_requires_thresholds_and_ring_attribution(self):
        rows = self._gate_rows()
        status, details = analysis.provisional_gate(rows)
        self.assertEqual(status, 'PROVISIONAL_PASS')
        self.assertEqual(len(details), 7)

        failed = [dict(row) for row in rows]
        target = next(
            row for row in failed
            if row['variant'] == 'center_ring' and row['seed'] == 49
        )
        target['fa_at_0_5'] = 0.009
        target['fa_at_0_5_relative_reduction'] = 0.10
        status, _ = analysis.provisional_gate(failed)
        self.assertEqual(status, 'PROVISIONAL_FAIL')

    def test_aggregate_uses_sample_sd(self):
        rows = self._gate_rows()
        summary = analysis.aggregate_rows(rows)
        baseline = next(item for item in summary if item['variant'] == 'none')
        self.assertEqual(baseline['n'], 3)
        expected = np.std([0.90, 0.91, 0.92], ddof=1)
        self.assertAlmostEqual(baseline['pd_at_fixed_fa_sd'], expected)

    def test_internal_train_is_official_train_minus_val_not_numeric_range(self):
        dataset_root = REPO_ROOT.parent / 'datasets' / analysis.DATASET
        if not (dataset_root / 'train.txt').is_file():
            self.skipTest('Repository Noise8 dataset is unavailable.')
        train_names, official_test = analysis.expected_internal_train_names(dataset_root)
        self.assertEqual(len(train_names), 64)
        self.assertEqual(len(official_test), 20)
        for leaked_name in ('Sequence47', 'Sequence56', 'Sequence59', 'Sequence76'):
            self.assertNotIn(leaked_name, train_names)
            self.assertIn(leaked_name, official_test)

    def test_full_semantic_validation_and_outputs(self):
        dataset_train = REPO_ROOT.parent / 'datasets' / analysis.DATASET / 'train.txt'
        if not dataset_train.is_file():
            self.skipTest('Repository Noise8 dataset is unavailable.')
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            experiment_root = temporary / 'experiment'
            log_root = temporary / 'log' / 'sem_seg'
            self._write_complete_fixture(experiment_root, log_root)
            rows, loaded = analysis.load_rows(experiment_root, log_root)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(loaded), 12)
            analysis.main([
                '--experiment-root', str(experiment_root),
                '--log-root', str(log_root),
                '--bootstrap-replicates', '50',
                '--bootstrap-seed', '9',
            ])
            report = (experiment_root / 'NOISE8_ANALYSIS.md').read_text(
                encoding='utf-8'
            )
            self.assertIn('12/12 run', report)
            self.assertIn('PROVISIONAL', report)
            self.assertIn('raw-logit', report)
            self.assertTrue((experiment_root / 'noise8_results.csv').is_file())
            self.assertNotIn('f1', report.lower())
            result_header = (
                experiment_root / 'noise8_results.csv'
            ).read_text(encoding='utf-8').splitlines()[0]
            summary_header = (
                experiment_root / 'noise8_summary.csv'
            ).read_text(encoding='utf-8').splitlines()[0]
            self.assertNotIn('f1', result_header.lower())
            self.assertNotIn('f1', summary_header.lower())

    def test_training_provenance_rejects_disabled_swanlab_and_wrong_bottleneck(self):
        dataset_train = REPO_ROOT.parent / 'datasets' / analysis.DATASET / 'train.txt'
        if not dataset_train.is_file():
            self.skipTest('Repository Noise8 dataset is unavailable.')
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            experiment_root = temporary / 'experiment'
            log_root = temporary / 'log' / 'sem_seg'
            self._write_complete_fixture(experiment_root, log_root)
            job = analysis.expected_manifest_rows()[0]
            log_path = (
                log_root / job['log_dir'] / 'logs' / (analysis.MODEL + '.txt')
            )
            original = log_path.read_text(encoding='utf-8')
            log_path.write_text(
                original.replace('use_swanlab=1', 'use_swanlab=0'),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'use_swanlab mismatch'):
                analysis.load_rows(experiment_root, log_root)
            log_path.write_text(
                original.replace(
                    'structure_bottleneck_channels=8',
                    'structure_bottleneck_channels=7',
                ),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(
                ValueError, 'structure_bottleneck_channels mismatch'
            ):
                analysis.load_rows(experiment_root, log_root)

    def test_model_state_dict_rejects_empty_and_malformed_state(self):
        with self.assertRaisesRegex(ValueError, 'must be non-empty'):
            analysis.validate_model_state_dict({}, 'none', 'fixture')
        state = self._model_state('none')
        state['bad/key'] = state.pop('conv_in.0.bias')
        with self.assertRaisesRegex(ValueError, 'malformed state key'):
            analysis.validate_model_state_dict(state, 'none', 'fixture')

    def test_model_state_dict_rejects_noncore_shape_change(self):
        state = self._model_state('none')
        self.assertEqual(tuple(state['conv_in.1.weight'].shape), (8,))
        state['conv_in.1.weight'] = torch.zeros(7)
        with self.assertRaisesRegex(
            ValueError, 'complete state shape mismatch for conv_in.1.weight'
        ):
            analysis.validate_model_state_dict(state, 'none', 'fixture')

    @staticmethod
    def _gate_rows():
        rows = []
        for variant in analysis.VARIANTS:
            for offset, seed in enumerate(analysis.SEEDS):
                base_pd = 0.90 + offset * 0.01
                base_fa = 0.010
                if variant == 'none':
                    pd, fa, latency = base_pd, base_fa, 1.0
                elif variant == 'center_multiscale':
                    pd, fa, latency = base_pd, 0.009, 1.1
                elif variant == 'center_ring':
                    pd, fa, latency = base_pd, 0.007, 1.2
                else:
                    pd, fa, latency = base_pd, 0.0095, 1.05
                row = {
                    'run_id': '%s_%d' % (variant, seed),
                    'variant': variant,
                    'variant_label': analysis.VARIANTS[variant][3],
                    'seed': seed,
                    'pd_at_fixed_fa': pd,
                    'fa_at_fixed_pd': fa,
                    'auc': 0.9,
                    'pd_at_0_5': pd,
                    'fa_at_0_5': fa,
                    'pixel_f1_at_0_5': 0.5,
                    'elapsed_seconds': latency,
                    'delta_pd_at_fixed_fa': pd - base_pd,
                    'delta_fa_at_fixed_pd': fa - base_fa,
                    'delta_auc': 0.0,
                    'delta_pd_at_0_5': pd - base_pd,
                    'delta_fa_at_0_5': fa - base_fa,
                    'delta_elapsed_seconds': latency - 1.0,
                    'fa_at_fixed_pd_relative_reduction': (base_fa - fa) / base_fa,
                    'fa_at_0_5_relative_reduction': (base_fa - fa) / base_fa,
                    'latency_ratio': latency,
                }
                rows.append(row)
        return rows

    def _write_complete_fixture(self, experiment_root, log_root):
        (experiment_root / 'splits').mkdir(parents=True)
        (experiment_root / 'metrics').mkdir()
        dataset_root = (REPO_ROOT.parent / 'datasets' / analysis.DATASET).resolve()
        train_names, _ = analysis.expected_internal_train_names(dataset_root)
        (experiment_root / 'splits' / 'train_sequences.txt').write_text(
            '\n'.join(train_names) + '\n', encoding='utf-8'
        )
        val_path = experiment_root / 'splits' / 'val_sequences.txt'
        val_path.write_text('\n'.join(analysis.VAL_NAMES) + '\n', encoding='utf-8')
        manifest = analysis.expected_manifest_rows()
        with (experiment_root / 'manifest.tsv').open(
            'w', newline='', encoding='utf-8'
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(manifest[0]), delimiter='\t'
            )
            writer.writeheader()
            writer.writerows(manifest)

        thresholds = common.EXPECTED_THRESHOLD_GRID
        sequence_count = len(analysis.VAL_NAMES)
        target_counts = np.full(
            (sequence_count, thresholds.size), 10, dtype=np.int64
        )
        true_counts = np.where(
            thresholds[None, :] <= 0.5, 10, 0
        ).repeat(sequence_count, axis=0)
        false_counts = np.floor(
            (1.0 - thresholds[None, :]) * 1000
        ).astype(np.int64).repeat(sequence_count, axis=0)
        pixels = np.full(sequence_count, 10000, dtype=np.int64)
        pd_curve = true_counts.sum(axis=0) / target_counts.sum(axis=0)
        fa_curve = false_counts.sum(axis=0) / pixels.sum()
        paper_indices = [
            int(np.flatnonzero(thresholds == value)[0])
            for value in common.PAPER_THRESHOLDS
        ]
        half = int(np.flatnonzero(thresholds == 0.5)[0])
        paper_auc = float(abs(np.trapz(pd_curve[paper_indices], fa_curve[paper_indices])))
        dense_auc = float(abs(np.trapz(pd_curve, fa_curve)))
        states = {
            variant: self._model_state(variant) for variant in analysis.VARIANTS
        }

        for job in manifest:
            run_dir = log_root / job['log_dir']
            (run_dir / 'checkpoints').mkdir(parents=True)
            (run_dir / 'logs').mkdir()
            checkpoint_path = run_dir / 'checkpoints' / 'epoch_32_model.pth'
            torch.save({
                'epoch': 31,
                'model_name': analysis.MODEL,
                'model_config': {
                    'structure_variant': job['structure_variant'],
                    'structure_bottleneck_channels': 8,
                    'eval_chunk_rows': 32,
                    'spatial_ckpt': None,
                    'st_ckpt': None,
                    'freeze_pretrained': False,
                },
                'model_state_dict': states[job['structure_variant']],
            }, checkpoint_path)
            training = {
                'seed': int(job['seed']), 'gpu': job['gpu'], 'gpu_num': 1,
                'model': analysis.MODEL,
                'structure_variant': job['structure_variant'],
                'structure_bottleneck_channels': 8,
                'dataset': analysis.DATASET, 'batch_size': 4,
                'gradient_accumulation_steps': 1, 'epoch': 32,
                'learning_rate': 0.001, 'optimizer': 'Adam',
                'decay_rate': 0.0001, 'step_size': 10, 'lr_decay': 0.7,
                'seqlen': 40, 'patch_size': 128, 'sample_rate': 0.1,
                'sequence_augmentation': 0, 'loss': 'soft_iou',
                'threshold_eval': 0.5,
                'train_amp': 1, 'eval_amp': 1, 'eval_chunk_rows': 32,
                'eval_interval': 8, 'skip_inprocess_validation': 1,
                'early_stopping_patience': 0, 'train_workers': 4,
                'val_workers': 1, 'prefetch_factor': 2, 'deterministic': 1,
                'log_dir': job['log_dir'], 'resume': 'never',
                'resume_checkpoint': None, 'run_test_after_train': 0,
                'base_ckpt': '', 'spatial_ckpt': '', 'st_ckpt': '',
                'freeze_pretrained': 0, 'datapath': str(dataset_root),
                'use_swanlab': 1, 'swanlab_project': 'DeepPro-BC-TPro',
                'swanlab_group': 'bc-tpro-stage1-noise8-scratch',
                'swanlab_mode': 'cloud', 'swanlab_resume': 'never',
                'train_sequence_list': str(
                    (experiment_root / 'splits' / 'train_sequences.txt').resolve()
                ),
                'val_sequence_list': str(val_path.resolve()),
                'savepath': str(log_root.parent.resolve()),
            }
            (run_dir / 'logs' / (analysis.MODEL + '.txt')).write_text(
                namespace_line(training) + '\n'
                + 'Initialized %s from random weights; no base checkpoint loaded.\n'
                % analysis.MODEL,
                encoding='utf-8',
            )
            metrics_path = (
                experiment_root / 'metrics'
                / (job['run_id'] + analysis.METRICS_SUFFIX)
            ).resolve()
            evaluation = {
                'amp': True, 'batch_size': 1, 'dataset': analysis.DATASET,
                'epoch': 32, 'eval_chunk_rows': 32, 'gpu': job['gpu'],
                'log_dir': job['log_dir'], 'output_only': False,
                'prefetch_factor': 1, 'seqlen': 40, 'sequence_start': 0,
                'sequence_stop': None, 'test_workers': 1,
                'threshold_eval': 0.5, 'threshold_grid_step': 0.01,
                'datapath': str(dataset_root), 'sequence_list': str(val_path.resolve()),
                'logpath': str(log_root.parent.resolve()),
                'metrics_json': str(metrics_path),
            }
            (run_dir / 'eval_epoch-32.txt').write_text(
                namespace_line(evaluation) + '\n'
                + 'Paper-aligned metrics saved to %s.\n' % metrics_path,
                encoding='utf-8',
            )
            payload = {
                'schema_version': 2, 'inference_amp': True,
                'dataset': analysis.DATASET, 'model': analysis.MODEL,
                'checkpoint': str(checkpoint_path.resolve()),
                'checkpoint_epoch': 32, 'sequence_length': 40,
                'sequence_count': sequence_count, 'evaluation_windows': 48,
                'elapsed_seconds': 1.0 + int(job['seed']) / 1000.0,
                'cuda_peak_allocated_gib': 1.0,
                'cuda_peak_reserved_gib': 1.1, 'parameters_m': 0.1,
                'operating_threshold': 0.5,
                'paper_thresholds': common.PAPER_THRESHOLDS.tolist(),
                'all': {
                    'true_targets': float(true_counts[:, half].sum()),
                    'total_targets': float(target_counts[:, half].sum()),
                    'false_pixels': float(false_counts[:, half].sum()),
                    'pixel_count': float(pixels.sum()),
                    'pd': float(pd_curve[half]), 'fa': float(fa_curve[half]),
                    'pd_percent': float(pd_curve[half] * 100),
                    'fa_x1e5': float(fa_curve[half] * 1e5),
                    'auc': paper_auc, 'auc_dense_grid': dense_auc,
                },
                'pixel_iou_at_0_5': 0.5,
                'pixel_precision_at_0_5': 2.0 / 3.0,
                'pixel_recall_at_0_5': 2.0 / 3.0,
                'pixel_f1_at_0_5': 2.0 / 3.0,
                'curve_counts': {
                    'thresholds': thresholds.tolist(),
                    'sequence_names': list(analysis.VAL_NAMES),
                    'false_pixels_by_sequence': false_counts.tolist(),
                    'true_targets_by_sequence': true_counts.tolist(),
                    'total_targets_by_sequence': target_counts.tolist(),
                    'pixel_count_by_sequence': pixels.tolist(),
                },
            }
            metrics_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )

    @staticmethod
    def _model_state(variant):
        model_module = importlib.import_module(
            'networks.models.DeepPro-Plus_BCTPro'
        )
        model = model_module.detector(
            1, 40, 40,
            structure_variant=variant,
            structure_bottleneck_channels=8,
            eval_chunk_rows=32,
        )
        return model.state_dict()


if __name__ == '__main__':
    unittest.main()
