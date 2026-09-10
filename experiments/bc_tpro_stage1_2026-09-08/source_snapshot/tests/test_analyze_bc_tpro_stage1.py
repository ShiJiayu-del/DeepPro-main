"""CPU-only contract tests for the BC-TPro stage-1 analyzer."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tools import analyze_bc_tpro_stage1 as analyzer


def _valid_payload(sequence_names):
    thresholds = analyzer.EXPECTED_THRESHOLD_GRID
    sequence_count = len(sequence_names)
    target_per_sequence = np.arange(
        6, 6 + sequence_count, dtype=np.float64
    )
    targets = np.repeat(
        target_per_sequence[:, None], thresholds.size, axis=1
    )
    true_counts = np.where(
        thresholds[None, :] <= 0.5, targets, 0.0
    )
    false_at_zero = np.arange(
        100, 100 + sequence_count, dtype=np.float64
    )[:, None]
    false_at_half = np.arange(
        10, 10 + sequence_count, dtype=np.float64
    )[:, None]
    false_counts = np.where(
        thresholds[None, :] == 0.0,
        false_at_zero,
        np.where(thresholds[None, :] <= 0.5, false_at_half, 0.0),
    )
    pixels = np.arange(
        1000, 1000 + sequence_count, dtype=np.float64
    )
    half_index = int(np.flatnonzero(thresholds == 0.5)[0])
    true_totals = true_counts.sum(axis=0)
    target_totals = targets.sum(axis=0)
    false_totals = false_counts.sum(axis=0)
    pd_curve = true_totals / target_totals
    fa_curve = false_totals / pixels.sum()
    pd_at_half = pd_curve[half_index]
    fa_at_half = fa_curve[half_index]
    paper_indices = [
        int(np.flatnonzero(thresholds == threshold)[0])
        for threshold in analyzer.PAPER_THRESHOLDS
    ]
    return {
        'schema_version': 2,
        'inference_amp': True,
        'dataset': 'NUDT-MIRSDT',
        'model': 'DeepPro-Plus_BCTPro',
        'checkpoint': '',
        'checkpoint_epoch': 32,
        'sequence_length': 40,
        'sequence_count': sequence_count,
        'evaluation_windows': 48,
        'elapsed_seconds': 1.0,
        'cuda_peak_allocated_gib': 1.0,
        'cuda_peak_reserved_gib': 1.25,
        'parameters_m': 0.1,
        'operating_threshold': 0.5,
        'paper_thresholds': analyzer.PAPER_THRESHOLDS.tolist(),
        'pixel_iou_at_0_5': 1.0 / 3.0,
        'pixel_precision_at_0_5': 0.5,
        'pixel_recall_at_0_5': 0.5,
        'pixel_f1_at_0_5': 0.5,
        'all': {
            'true_targets': int(true_totals[half_index]),
            'total_targets': int(target_totals[half_index]),
            'false_pixels': int(false_totals[half_index]),
            'pixel_count': int(pixels.sum()),
            'pd': float(pd_at_half),
            'fa': float(fa_at_half),
            'pd_percent': float(pd_at_half * 100.0),
            'fa_x1e5': float(fa_at_half * 1e5),
            'auc': float(abs(np.trapz(
                pd_curve[paper_indices], fa_curve[paper_indices]
            ))),
            'auc_dense_grid': float(abs(np.trapz(pd_curve, fa_curve))),
        },
        'curve_counts': {
            'thresholds': thresholds.tolist(),
            'sequence_names': list(sequence_names),
            'false_pixels_by_sequence': false_counts.tolist(),
            'true_targets_by_sequence': true_counts.tolist(),
            'total_targets_by_sequence': targets.tolist(),
            'pixel_count_by_sequence': pixels.tolist(),
        },
    }


def _bootstrap_curve(false_counts):
    return {
        'thresholds': np.asarray([0.0, 0.5, 1.0]),
        'sequence_names': ['S0'],
        'false_counts': np.asarray([false_counts], dtype=np.float64),
        'true_counts': np.asarray([[10.0, 8.0, 0.0]]),
        'target_counts': np.asarray([[10.0, 10.0, 10.0]]),
        'pixels': np.asarray([1000.0]),
        'threshold_half_index': 1,
    }


class CurveValidationTests(unittest.TestCase):
    def test_valid_schema2_curve_and_preregistered_grid(self):
        names = list(analyzer.EXPECTED_VALIDATION_NAMES)
        payload = _valid_payload(names)
        curve = analyzer.validate_curve_counts(payload, names, 'fixture')
        self.assertEqual(analyzer.EXPECTED_THRESHOLD_GRID.size, 109)
        np.testing.assert_array_equal(
            curve['thresholds'], analyzer.EXPECTED_THRESHOLD_GRID
        )
        self.assertEqual(curve['false_counts'].shape, (16, 109))
        recomputed = analyzer.validate_payload_metrics(
            payload, curve, 'fixture'
        )
        self.assertAlmostEqual(
            recomputed['paper_auc'], payload['all']['auc']
        )
        self.assertAlmostEqual(
            recomputed['dense_auc'], payload['all']['auc_dense_grid']
        )

    def test_rejects_corrupt_identity_or_counts(self):
        names = list(analyzer.EXPECTED_VALIDATION_NAMES)
        cases = {}

        wrong_order = _valid_payload(names)
        wrong_order['curve_counts']['sequence_names'][:2] = reversed(
            wrong_order['curve_counts']['sequence_names'][:2]
        )
        cases['sequence order'] = wrong_order

        wrong_grid = _valid_payload(names)
        wrong_grid['curve_counts']['thresholds'][45] += 1e-7
        cases['threshold grid'] = wrong_grid

        wrong_shape = _valid_payload(names)
        wrong_shape['curve_counts']['false_pixels_by_sequence'].pop()
        cases['matrix shape'] = wrong_shape

        negative = _valid_payload(names)
        negative['curve_counts']['false_pixels_by_sequence'][0][0] = -1
        cases['negative count'] = negative

        non_integer = _valid_payload(names)
        non_integer['curve_counts']['true_targets_by_sequence'][0][0] = 0.5
        cases['noninteger count'] = non_integer

        true_exceeds_target = _valid_payload(names)
        true_exceeds_target['curve_counts']['true_targets_by_sequence'][0][0] = 99
        cases['true exceeds target'] = true_exceeds_target

        varying_targets = _valid_payload(names)
        varying_targets['curve_counts']['total_targets_by_sequence'][0][1] += 1
        cases['target varies'] = varying_targets

        increasing_true = _valid_payload(names)
        increasing_true['curve_counts']['true_targets_by_sequence'][0][-1] = 1
        cases['true increases'] = increasing_true

        increasing_false = _valid_payload(names)
        increasing_false['curve_counts']['false_pixels_by_sequence'][0][-1] = 1
        cases['false increases'] = increasing_false

        zero_pixels = _valid_payload(names)
        zero_pixels['curve_counts']['pixel_count_by_sequence'][0] = 0
        cases['zero pixels'] = zero_pixels

        wrong_schema = _valid_payload(names)
        wrong_schema['schema_version'] = 3
        cases['schema must equal two'] = wrong_schema

        for label, payload in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    analyzer.validate_curve_counts(payload, names, label)

    def test_recomputed_aggregate_and_diagnostic_fields_are_strict(self):
        names = list(analyzer.EXPECTED_VALIDATION_NAMES)
        base = _valid_payload(names)
        curve = analyzer.validate_curve_counts(base, names, 'base')
        cases = {}

        bad_auc = copy.deepcopy(base)
        bad_auc['all']['auc'] += 0.01
        cases['paper AUC'] = bad_auc

        bad_total = copy.deepcopy(base)
        bad_total['all']['false_pixels'] += 1
        cases['all totals'] = bad_total

        bad_units = copy.deepcopy(base)
        bad_units['all']['fa_x1e5'] += 1
        cases['Fa units'] = bad_units

        bad_paper_grid = copy.deepcopy(base)
        bad_paper_grid['paper_thresholds'][1] = 1e-19
        cases['paper thresholds'] = bad_paper_grid

        bad_amp = copy.deepcopy(base)
        bad_amp['inference_amp'] = False
        cases['AMP'] = bad_amp

        bad_pixel = copy.deepcopy(base)
        bad_pixel['pixel_iou_at_0_5'] = 1.1
        cases['pixel range'] = bad_pixel

        bad_elapsed = copy.deepcopy(base)
        bad_elapsed['elapsed_seconds'] = 0
        cases['elapsed'] = bad_elapsed

        bad_peak = copy.deepcopy(base)
        bad_peak['cuda_peak_reserved_gib'] = 0.5
        cases['peak ordering'] = bad_peak

        for label, payload in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    analyzer.validate_payload_metrics(payload, curve, label)

    def test_fixed_validation_split_is_not_replaceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_root = Path(temporary)
            split_dir = experiment_root / 'splits'
            split_dir.mkdir()
            split_path = split_dir / 'val_sequences.txt'
            split_path.write_text(
                '\n'.join(analyzer.EXPECTED_VALIDATION_NAMES) + '\n',
                encoding='utf-8',
            )
            self.assertEqual(
                analyzer.read_expected_validation_names(experiment_root),
                list(analyzer.EXPECTED_VALIDATION_NAMES),
            )
            changed = list(analyzer.EXPECTED_VALIDATION_NAMES)
            changed[-1] = 'Sequence85'
            split_path.write_text('\n'.join(changed) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                analyzer.read_expected_validation_names(experiment_root)


class ProtocolValidationTests(unittest.TestCase):
    def test_manifest_requires_exact_preregistered_mapping(self):
        repo_root = Path(analyzer.__file__).resolve().parents[1]
        manifest = (
            repo_root / 'experiments' / 'bc_tpro_stage1_2026-09-08'
            / 'manifest.tsv'
        )
        rows = analyzer.load_manifest(manifest)
        self.assertEqual(len(rows), 12)
        with tempfile.TemporaryDirectory() as temporary:
            corrupt = Path(temporary) / 'manifest.tsv'
            text = manifest.read_text(encoding='utf-8').replace(
                'c1_center_multiscale_seed47',
                'c1_center_multiscale_typo_seed47',
            )
            corrupt.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                analyzer.load_manifest(corrupt)

    def test_split_sha256_is_fixed(self):
        repo_root = Path(analyzer.__file__).resolve().parents[1]
        source_splits = (
            repo_root / 'experiments' / 'bc_tpro_stage1_2026-09-08'
            / 'splits'
        )
        with tempfile.TemporaryDirectory() as temporary:
            experiment_root = Path(temporary)
            split_dir = experiment_root / 'splits'
            split_dir.mkdir()
            for name in ('train_sequences.txt', 'val_sequences.txt'):
                (split_dir / name).write_bytes((source_splits / name).read_bytes())
            analyzer.validate_split_artifacts(experiment_root)
            with (split_dir / 'train_sequences.txt').open('a') as handle:
                handle.write('Sequence999\n')
            with self.assertRaises(ValueError):
                analyzer.validate_split_artifacts(experiment_root)

    def test_gate_requires_all_controls_exact_seeds_and_valid_latency(self):
        rows = []
        for variant in analyzer.VARIANT_LABELS:
            for seed in analyzer.EXPECTED_SEEDS:
                for condition in analyzer.CONDITIONS:
                    rows.append({
                        'variant': variant,
                        'seed': seed,
                        'condition': condition,
                        'elapsed_seconds': 1.0,
                        'latency_ratio': 1.0,
                        'fa_at_fixed_pd_relative_reduction': (
                            0.25 if variant == 'center_ring' else 0.0
                        ),
                        'delta_pd_at_fixed_fa': 0.0,
                    })
        status, details = analyzer.gate_status(rows)
        self.assertEqual(status, 'PASS', details)
        self.assertTrue(any('analysis amendment' in item for item in details))

        without_control = [
            row for row in rows if row['variant'] != 'temporal_control'
        ]
        self.assertEqual(
            analyzer.gate_status(without_control)[0], 'NOT_EVALUABLE'
        )
        bad_latency = copy.deepcopy(rows)
        bad_latency[0]['latency_ratio'] = float('nan')
        self.assertEqual(
            analyzer.gate_status(bad_latency)[0], 'NOT_EVALUABLE'
        )


class IdentityValidationTests(unittest.TestCase):
    @staticmethod
    def _namespace_line(values):
        arguments = ', '.join(
            '%s=%r' % item for item in values.items()
        )
        return 'INFO Namespace(%s)\n' % arguments

    def _fixture(self, root):
        names = list(analyzer.EXPECTED_VALIDATION_NAMES)
        experiment_root = root / 'experiment'
        log_root = root / 'log' / 'sem_seg'
        job = {
            'run_id': 'c2_center_ring_seed47',
            'model': 'DeepPro-Plus_BCTPro',
            'structure_variant': 'center_ring',
            'seed': '47',
            'gpu': '0',
            'log_dir': '2026-09-08/C2_seed47',
        }
        run_dir = log_root / job['log_dir']
        checkpoint_path = run_dir / 'checkpoints' / 'epoch_32_model.pth'
        checkpoint_path.parent.mkdir(parents=True)
        torch.save({
            'epoch': 31,
            'model_name': job['model'],
            'model_config': {'structure_variant': 'center_ring'},
        }, checkpoint_path)
        log_path = run_dir / 'logs' / (job['model'] + '.txt')
        log_path.parent.mkdir()
        repo_root = Path(analyzer.__file__).resolve().parents[1]
        train_namespace = {
            'seed': 47,
            'gpu': '0',
            'gpu_num': 1,
            'model': 'DeepPro-Plus_BCTPro',
            'structure_variant': 'center_ring',
            'dataset': 'NUDT-MIRSDT',
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
            'train_amp': 1,
            'eval_amp': 1,
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
            'use_swanlab': 1,
            'swanlab_project': 'DeepPro-BC-TPro',
            'swanlab_group': 'bc-tpro-stage1-scratch',
            'swanlab_mode': 'cloud',
            'swanlab_resume': 'never',
            'base_ckpt': '',
            'spatial_ckpt': '',
            'st_ckpt': '',
            'freeze_pretrained': 0,
            'datapath': str(repo_root.parent / 'datasets' / 'NUDT-MIRSDT'),
            'train_sequence_list': str(
                experiment_root / 'splits' / 'train_sequences.txt'
            ),
            'val_sequence_list': str(
                experiment_root / 'splits' / 'val_sequences.txt'
            ),
            'savepath': str(log_root.parent),
        }
        log_path.write_text(
            self._namespace_line(train_namespace), encoding='utf-8'
        )

        metrics_root = experiment_root / 'metrics'
        metrics_root.mkdir(parents=True)
        metrics_paths = {
            'clean_val': metrics_root / 'c2_center_ring_seed47__clean_val.json',
            'noise8_val': metrics_root / 'c2_center_ring_seed47__noise8_val.json',
        }
        eval_lines = []
        for condition, metrics_path in metrics_paths.items():
            dataset = analyzer.CONDITION_DATASETS[condition]
            eval_namespace = {
                'amp': True,
                'batch_size': 1,
                'dataset': dataset,
                'epoch': 32,
                'eval_chunk_rows': 32,
                'gpu': '0',
                'log_dir': job['log_dir'],
                'output_only': False,
                'prefetch_factor': 1,
                'seqlen': 40,
                'sequence_start': 0,
                'sequence_stop': None,
                'test_workers': 1,
                'threshold_eval': 0.5,
                'threshold_grid_step': 0.01,
                'datapath': str(repo_root.parent / 'datasets' / dataset),
                'sequence_list': str(
                    experiment_root / 'splits' / 'val_sequences.txt'
                ),
                'logpath': str(log_root.parent),
                'metrics_json': str(metrics_path),
            }
            eval_lines.append(self._namespace_line(eval_namespace))
            eval_lines.append(
                'INFO Paper-aligned metrics saved to %s.\n'
                % metrics_path.resolve()
            )
        eval_path = run_dir / 'eval_epoch-32.txt'
        eval_path.write_text(''.join(eval_lines), encoding='utf-8')
        payload = _valid_payload(names)
        payload['checkpoint'] = str(checkpoint_path)
        return {
            'names': names,
            'job': job,
            'payload': payload,
            'checkpoint_path': checkpoint_path,
            'log_path': log_path,
            'eval_path': eval_path,
            'experiment_root': experiment_root,
            'log_root': log_root,
            'metrics_paths': metrics_paths,
            'train_namespace': train_namespace,
        }

    def test_checkpoint_log_and_payload_identity_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            curve = analyzer.validate_payload_identity(
                fixture['payload'], fixture['job'], 'clean_val',
                fixture['names'], fixture['experiment_root'],
                fixture['log_root'], fixture['metrics_paths']['clean_val'],
            )
            self.assertEqual(curve['threshold_half_index'], 58)

            noise_payload = copy.deepcopy(fixture['payload'])
            noise_payload['dataset'] = 'NUDT-MIRSDT-Noise8.0_FJY'
            analyzer.validate_payload_identity(
                noise_payload, fixture['job'], 'noise8_val',
                fixture['names'], fixture['experiment_root'],
                fixture['log_root'], fixture['metrics_paths']['noise8_val'],
            )

    def test_rejects_noninteger_checkpoint_epoch_and_nonscratch_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            torch.save({
                'epoch': 31.0,
                'model_name': fixture['job']['model'],
                'model_config': {'structure_variant': 'center_ring'},
            }, fixture['checkpoint_path'])
            with self.assertRaises(ValueError):
                analyzer.validate_payload_identity(
                    fixture['payload'], fixture['job'], 'clean_val',
                    fixture['names'], fixture['experiment_root'],
                    fixture['log_root'], fixture['metrics_paths']['clean_val'],
                )

            torch.save({
                'epoch': 31,
                'model_name': fixture['job']['model'],
                'model_config': {'structure_variant': 'center_ring'},
            }, fixture['checkpoint_path'])
            bad_namespace = dict(fixture['train_namespace'])
            bad_namespace['base_ckpt'] = '/tmp/pretrained.pth'
            fixture['log_path'].write_text(
                self._namespace_line(bad_namespace), encoding='utf-8'
            )
            with self.assertRaises(ValueError):
                analyzer.validate_payload_identity(
                    fixture['payload'], fixture['job'], 'clean_val',
                    fixture['names'], fixture['experiment_root'],
                    fixture['log_root'], fixture['metrics_paths']['clean_val'],
                )

    def test_duplicate_eval_namespace_for_metrics_path_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            records = analyzer.parse_namespace_records(fixture['eval_path'])
            duplicate = self._namespace_line(records[0]['values'])
            with fixture['eval_path'].open('a', encoding='utf-8') as handle:
                handle.write(duplicate)
            with self.assertRaises(ValueError):
                analyzer.validate_payload_identity(
                    fixture['payload'], fixture['job'], 'clean_val',
                    fixture['names'], fixture['experiment_root'],
                    fixture['log_root'], fixture['metrics_paths']['clean_val'],
                )

    def test_wrong_training_gpu_and_eval_amp_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            bad_training = dict(fixture['train_namespace'])
            bad_training['gpu'] = '1'
            fixture['log_path'].write_text(
                self._namespace_line(bad_training), encoding='utf-8'
            )
            with self.assertRaises(ValueError):
                analyzer.validate_payload_identity(
                    fixture['payload'], fixture['job'], 'clean_val',
                    fixture['names'], fixture['experiment_root'],
                    fixture['log_root'], fixture['metrics_paths']['clean_val'],
                )

            fixture['log_path'].write_text(
                self._namespace_line(fixture['train_namespace']),
                encoding='utf-8',
            )
            eval_text = fixture['eval_path'].read_text(encoding='utf-8')
            fixture['eval_path'].write_text(
                eval_text.replace('Namespace(amp=True', 'Namespace(amp=False', 1),
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                analyzer.validate_payload_identity(
                    fixture['payload'], fixture['job'], 'clean_val',
                    fixture['names'], fixture['experiment_root'],
                    fixture['log_root'], fixture['metrics_paths']['clean_val'],
                )

    def test_ambiguous_namespace_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'train.log'
            namespace = "Namespace(seed=47)\n"
            path.write_text(namespace + namespace, encoding='utf-8')
            with self.assertRaises(ValueError):
                analyzer.parse_training_namespace(path)


class PairedBootstrapTests(unittest.TestCase):
    def test_exact_paired_deltas_and_one_shared_resample(self):
        loaded = {}
        rows = []
        expected = {
            'delta_pd_at_fixed_fa': 0.0,
            'delta_fa_at_fixed_pd': -0.005,
            'fa_at_fixed_pd_relative_reduction': 0.5,
            'delta_low_fa_pauc': 0.045,
        }
        for condition in analyzer.CONDITIONS:
            for training_seed in analyzer.EXPECTED_SEEDS:
                for variant, false_counts in (
                    ('none', [100, 10, 0]),
                    ('center_ring', [80, 5, 0]),
                ):
                    run_id = '%s_%d' % (variant, training_seed)
                    loaded[(run_id, condition)] = {
                        'job': {
                            'run_id': run_id,
                            'structure_variant': variant,
                            'seed': str(training_seed),
                        },
                        'condition': condition,
                        'curve': _bootstrap_curve(false_counts),
                    }
                rows.append({
                    'variant': 'center_ring',
                    'seed': training_seed,
                    'condition': condition,
                    **expected,
                })

        with mock.patch.object(
            analyzer,
            '_bootstrap_weights',
            wraps=analyzer._bootstrap_weights,
        ) as weight_builder:
            summaries = analyzer.paired_video_bootstrap(
                loaded,
                rows,
                replicates=37,
                seed=20260908,
                low_fa_cap=0.1,
            )
        weight_builder.assert_called_once_with(37, 1, 20260908)
        self.assertEqual(len(summaries), 8)
        for item in summaries:
            expected_value = expected[item['metric']]
            with self.subTest(
                condition=item['condition'], metric=item['metric']
            ):
                self.assertAlmostEqual(item['point_estimate'], expected_value)
                self.assertAlmostEqual(
                    item['bootstrap_median'], expected_value
                )
                self.assertAlmostEqual(item['ci95_low'], expected_value)
                self.assertAlmostEqual(item['ci95_high'], expected_value)
                self.assertEqual(item['valid_replicates'], 37)
                self.assertEqual(item['valid_rate'], 1.0)
                self.assertTrue(item['ci_available'])

    def test_relative_fa_ci_requires_95_percent_validity(self):
        below = analyzer._bootstrap_summary_row(
            'center_ring',
            'clean_val',
            'fa_at_fixed_pd_relative_reduction',
            0.1,
            np.asarray([0.1] * 94 + [np.nan] * 6),
            100,
        )
        self.assertFalse(below['ci_available'])
        self.assertTrue(np.isnan(below['ci95_low']))

        boundary = analyzer._bootstrap_summary_row(
            'center_ring',
            'clean_val',
            'fa_at_fixed_pd_relative_reduction',
            0.1,
            np.asarray([0.1] * 95 + [np.nan] * 5),
            100,
        )
        self.assertTrue(boundary['ci_available'])
        self.assertAlmostEqual(boundary['ci95_low'], 0.1)
        self.assertAlmostEqual(boundary['ci95_high'], 0.1)

        absolute_below = analyzer._bootstrap_summary_row(
            'center_ring',
            'clean_val',
            'delta_pd_at_fixed_fa',
            0.1,
            np.asarray([0.1] * 94 + [np.nan] * 6),
            100,
        )
        self.assertFalse(absolute_below['ci_available'])
        self.assertTrue(np.isnan(absolute_below['ci95_high']))

    def test_zero_target_resample_is_nan_for_pd_and_pauc(self):
        curve = {
            'true_counts': np.asarray([[0.0, 0.0], [2.0, 1.0]]),
            'target_counts': np.asarray([[0.0, 0.0], [2.0, 2.0]]),
            'false_counts': np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            'pixels': np.asarray([100.0, 100.0]),
        }
        weights = np.asarray([[2, 0], [0, 2]], dtype=np.int16)
        resampled = analyzer._resampled_curves(curve, weights)
        self.assertTrue(np.isnan(resampled['pd'][0]).all())
        self.assertTrue(np.isfinite(resampled['pd'][1]).all())
        pauc = analyzer._bootstrap_pauc(
            resampled['pd'], resampled['fa'], 0.1
        )
        self.assertTrue(np.isnan(pauc[0]))
        self.assertTrue(np.isfinite(pauc[1]))

    def test_working_point_and_low_fa_support_diagnostics(self):
        thresholds = np.asarray([0.0, 0.5, 1.0])
        pd_curve = np.asarray([1.0, 0.8, 0.0])
        fa_curve = np.asarray([0.1, 0.01, 0.0])
        pd_diagnostic = analyzer.working_point_diagnostics(
            thresholds, pd_curve, fa_curve, 0.01, 'max_pd_at_fa'
        )
        self.assertEqual(pd_diagnostic['selected_threshold'], 0.5)
        self.assertFalse(pd_diagnostic['selected_is_endpoint'])

        no_internal = analyzer.low_fa_support_diagnostics(
            np.asarray([0.2, 0.1]), 5e-5
        )
        self.assertEqual(
            no_internal['low_fa_support'], 'origin-interpolation-only'
        )
        self.assertEqual(no_internal['low_fa_internal_unique_points'], 0)

    def test_invalid_caps_and_replicates_fail(self):
        for cap in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    analyzer.low_fa_pauc(
                        np.asarray([0.0, 1.0]),
                        np.asarray([0.0, 1.0]),
                        cap,
                    )
        with self.assertRaises(ValueError):
            analyzer._bootstrap_weights(0, 16, 1)


if __name__ == '__main__':
    unittest.main()
