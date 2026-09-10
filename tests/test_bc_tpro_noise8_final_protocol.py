import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from data_utils.TestDataLoader import TestIRSeqDataLoader
from tools import analyze_bc_tpro_noise8_final as final_analysis
from tools import analyze_bc_tpro_noise8_paper as selector
from tools import validate_bc_tpro_noise8_final as protocol


def fallback_lock(stage1_root, log_root):
    decisions = []
    for variant in ('temporal_control', 'center_multiscale', 'center_ring'):
        decisions.append({
            'variant': variant,
            'variant_label': variant,
            'qualified': False,
            'mean_pd_at_0_5': 0.8,
            'mean_fa_at_0_5': 1e-5,
            'mean_paper_auc': 0.9,
            'mean_delta_paper_auc': -0.01,
            'mean_fa_at_0_5_relative_reduction': 0.0,
            'mean_delta_pd_at_0_5': 0.0,
            'joint_nonworse_seed_count': 0,
            'max_latency_ratio': 1.0,
            'parameters_m': 0.1,
            'complete_finite_metrics': True,
            'mean_delta_pd_at_0_5_ge_minus_1pp': True,
            'mean_fa_at_0_5_relative_reduction_gt_zero': False,
            'at_least_two_joint_nonworse_seeds': False,
            'mean_paper_auc_not_below_b1': False,
            'every_seed_latency_le_1p3': True,
            'identity_verified': True,
        })
    selection = {
        'status': 'B1_FALLBACK',
        'selected_variant': 'none',
        'qualified_ranking': [],
        'reason': 'fixture',
    }
    return selector.build_locked_candidate_payload(
        selection, decisions, stage1_root, log_root, protocol.PROFILE,
    )


class FinalNoise8ProtocolTests(unittest.TestCase):
    def test_lock_schema_and_b1_only_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1_root = root / 'stage1'
            log_root = root / 'log' / 'sem_seg'
            payload = fallback_lock(stage1_root, log_root)
            selected = protocol.validate_lock_schema(
                payload, stage1_root, log_root, protocol.PROFILE,
            )
            self.assertEqual(selected, 'none')
            jobs = protocol.build_final_jobs(selected, protocol.PROFILE)
            self.assertEqual(len(jobs), 3)
            self.assertEqual([job['gpu'] for job in jobs], ['0', '1', '2'])
            self.assertTrue(all(job['role'] == 'B1' for job in jobs))

    def test_candidate_plan_always_keeps_b1(self):
        jobs = protocol.build_final_jobs('center_ring', protocol.PROFILE)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(
            [(job['wave'], job['structure_variant']) for job in jobs],
            [(1, 'none')] * 3 + [(2, 'center_ring')] * 3,
        )

    def test_lock_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1_root = root / 'stage1'
            log_root = root / 'log' / 'sem_seg'
            payload = fallback_lock(stage1_root, log_root)
            payload['detection_metric_contract']['operating_threshold'] = 0.4
            with self.assertRaisesRegex(ValueError, 'metric contract'):
                protocol.validate_lock_schema(
                    payload, stage1_root, log_root, protocol.PROFILE,
                )

    def test_modernized_profile_is_never_accepted(self):
        with self.assertRaisesRegex(ValueError, 'explicit upstream profile'):
            protocol.build_final_jobs('none', 'modernized')

    def test_official_split_metadata_without_payload_reads(self):
        if not (protocol.EXPECTED_DATA_ROOT / 'test.txt').is_file():
            self.skipTest('Local Noise8 split text is unavailable.')
        train_names, test_names = protocol.validate_official_split_metadata(
            protocol.EXPECTED_DATA_ROOT,
        )
        self.assertEqual(len(train_names), 80)
        self.assertEqual(tuple(test_names), protocol.EXPECTED_TEST_NAMES)
        self.assertFalse(set(train_names) & set(test_names))
        loader = TestIRSeqDataLoader(
            protocol.DATASET,
            data_root=str(protocol.EXPECTED_DATA_ROOT),
            seq_len=40,
            cat_len=4,
            split='test',
        )
        self.assertEqual(tuple(loader.seq_names), protocol.EXPECTED_TEST_NAMES)

    def test_official_list_rejects_duplicate_or_misnamed_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'split.txt'
            path.write_text(
                'Sequence1/Mix/00001.mat\n' * 100,
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'each unique'):
                protocol.read_official_sequence_list(path, 100, 1)

            path.write_text(
                ''.join(
                    'Sequence1/Mix/%05d.mat\n' % frame
                    for frame in range(1, 101)
                ).replace('00050.mat', '00050.png'),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'each unique'):
                protocol.read_official_sequence_list(path, 100, 1)

    def test_frame_layout_checks_exact_image_mask_and_centroid_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'Noise8'
            clean = root.parent / 'NUDT-MIRSDT'
            for directory in (
                root / 'Sequence1' / 'images',
                root / 'Sequence1' / 'masks',
                clean / 'Sequence1' / 'masks_centroid',
            ):
                directory.mkdir(parents=True)
                for frame in range(1, 101):
                    (directory / ('%05d.png' % frame)).touch()
            protocol.validate_sequence_frame_layout(
                root, ['Sequence1'], centroids=True,
            )
            (root / 'Sequence1' / 'masks' / '00100.png').unlink()
            with self.assertRaisesRegex(ValueError, 'exact unique'):
                protocol.validate_sequence_frame_layout(root, ['Sequence1'])

    def test_c3_gate_pass_refuses_c0_c2_candidate_lock(self):
        with mock.patch.object(
            selector.grid_analysis, 'provisional_gate',
            return_value=('PROVISIONAL_PASS', []),
        ):
            with self.assertRaisesRegex(selector.C3RequiredError, 'C3_REQUIRED'):
                selector.require_c3_gate_resolved([])

    def test_final_replay_independently_enforces_c3_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1_root = root / 'stage1'
            log_root = root / 'log' / 'sem_seg'
            payload = fallback_lock(stage1_root, log_root)
            with mock.patch.object(
                selector, 'load_complete_evidence',
                return_value=([], {}),
            ), mock.patch.object(
                selector, 'require_c3_gate_resolved',
                side_effect=selector.C3RequiredError('C3_REQUIRED'),
            ):
                with self.assertRaisesRegex(selector.C3RequiredError, 'C3_REQUIRED'):
                    protocol.replay_and_validate_lock(
                        payload, stage1_root, log_root, protocol.PROFILE,
                    )

    def test_final_protocol_lock_is_publish_once_and_detects_plan_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / 'data'
            stage1_root = root / 'stage1'
            log_root = root / 'log' / 'sem_seg'
            final_root = root / 'final'
            for path in (data_root, stage1_root, log_root, final_root):
                path.mkdir(parents=True)
            candidate = fallback_lock(stage1_root, log_root)
            jobs = protocol.build_final_jobs('none', protocol.PROFILE)
            patches = (
                mock.patch.object(protocol, 'EXPECTED_DATA_ROOT', data_root),
                mock.patch.object(protocol, 'EXPECTED_STAGE1_ROOT', stage1_root),
                mock.patch.object(protocol, 'EXPECTED_LOG_ROOT', log_root),
                mock.patch.object(protocol, 'EXPECTED_FINAL_ROOT', final_root),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                payload = protocol.build_final_protocol_lock(
                    candidate, jobs, data_root, stage1_root, log_root,
                    final_root, protocol.PROFILE,
                )
                lock_path = final_root / protocol.FINAL_PROTOCOL_LOCK_NAME
                manifest_path = final_root / protocol.FINAL_MANIFEST_NAME
                protocol._publish_text_once(
                    lock_path,
                    json.dumps(payload, indent=2, sort_keys=True) + '\n',
                )
                protocol._publish_text_once(
                    manifest_path, protocol.jobs_tsv(jobs),
                )
                with mock.patch.object(
                    protocol, 'load_and_validate_plan',
                    return_value=(candidate, jobs),
                ):
                    protocol.load_and_validate_frozen_plan(
                        stage1_root / 'LOCKED_CANDIDATE.json',
                        stage1_root, log_root, data_root, final_root,
                        protocol.PROFILE,
                    )
                changed = dict(candidate)
                changed['status'] = 'CANDIDATE_LOCKED'
                with mock.patch.object(
                    protocol, 'load_and_validate_plan',
                    return_value=(changed, jobs),
                ):
                    with self.assertRaisesRegex(
                        ValueError, 'Immutable final protocol lock differs',
                    ):
                        protocol.load_and_validate_frozen_plan(
                            stage1_root / 'LOCKED_CANDIDATE.json',
                            stage1_root, log_root, data_root, final_root,
                            protocol.PROFILE,
                        )
                with self.assertRaisesRegex(ValueError, 'Immutable protocol'):
                    protocol._publish_text_once(lock_path, '{}\n')

    def test_paper_curve_requires_exact_27_thresholds_and_integer_counts(self):
        names = list(protocol.EXPECTED_TEST_NAMES)
        thresholds = final_analysis.PAPER_THRESHOLDS
        targets = np.full((20, len(thresholds)), 10, dtype=np.int64)
        true = np.where(thresholds[None, :] <= 0.5, 10, 0).repeat(20, axis=0)
        false = np.floor((1.0 - thresholds[None, :]) * 1000).astype(
            np.int64
        ).repeat(20, axis=0)
        payload = {
            'schema_version': 2,
            'curve_counts': {
                'thresholds': thresholds.tolist(),
                'sequence_names': names,
                'false_pixels_by_sequence': false.tolist(),
                'true_targets_by_sequence': true.tolist(),
                'total_targets_by_sequence': targets.tolist(),
                'pixel_count_by_sequence': [10000] * 20,
            },
        }
        curve = final_analysis.validate_curve_counts(payload, names, 'fixture')
        self.assertEqual(curve['false'].shape, (20, 27))
        payload['curve_counts']['thresholds'] = np.linspace(0, 1, 27).tolist()
        with self.assertRaisesRegex(ValueError, '27-threshold'):
            final_analysis.validate_curve_counts(payload, names, 'fixture')

    def test_complete_metric_validation_returns_both_snr_auc_fields(self):
        names = list(protocol.EXPECTED_TEST_NAMES)
        thresholds = final_analysis.PAPER_THRESHOLDS
        targets = np.full((20, len(thresholds)), 10, dtype=np.int64)
        true = np.where(thresholds[None, :] <= 0.5, 10, 0).repeat(20, axis=0)
        false = np.floor((1.0 - thresholds[None, :]) * 1000).astype(
            np.int64
        ).repeat(20, axis=0)
        curve_payload = {
            'schema_version': 2,
            'curve_counts': {
                'thresholds': thresholds.tolist(),
                'sequence_names': names,
                'false_pixels_by_sequence': false.tolist(),
                'true_targets_by_sequence': true.tolist(),
                'total_targets_by_sequence': targets.tolist(),
                'pixel_count_by_sequence': [10000] * 20,
            },
        }
        curve = final_analysis.validate_curve_counts(
            curve_payload, names, 'fixture',
        )
        all_summary = final_analysis._summary_for_indices(
            curve, np.arange(20, dtype=np.int64),
        )
        low = np.asarray([
            index for index, name in enumerate(names)
            if name in final_analysis.LOW_SNR_NAMES
        ])
        high = np.asarray([
            index for index, name in enumerate(names)
            if name in final_analysis.HIGH_SNR_NAMES
        ])
        low_summary = final_analysis._summary_for_indices(curve, low)
        high_summary = final_analysis._summary_for_indices(curve, high)
        job = protocol.build_final_jobs('none', protocol.PROFILE)[0]
        model_module = __import__(
            'networks.models.DeepPro-Plus_BCTPro', fromlist=['detector'],
        )
        model = model_module.detector(
            1, 40, 40, structure_variant='none',
            structure_bottleneck_channels=8, eval_chunk_rows=32,
        )
        parameters_m = sum(p.numel() for p in model.parameters()) / 1e6
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_root = root / 'log' / 'sem_seg'
            log_root.mkdir(parents=True)
            metrics_path = root / 'metric.json'
            metrics_path.write_text('{}\n', encoding='utf-8')
            checkpoint_path = (
                log_root / job['log_dir'] / 'checkpoints'
                / 'epoch_32_model.pth'
            ).resolve()
            payload = {
                **curve_payload,
                'dataset': protocol.DATASET,
                'model': protocol.MODEL,
                'checkpoint': str(checkpoint_path),
                'checkpoint_epoch': 32,
                'sequence_length': 40,
                'sequence_count': 20,
                'evaluation_windows': 60,
                'inference_amp': False,
                'paper_thresholds': thresholds.tolist(),
                'operating_threshold': 0.5,
                'all': all_summary,
                'low_snr': low_summary,
                'high_snr': high_summary,
                'elapsed_seconds': 1.0,
                'cuda_peak_allocated_gib': 1.0,
                'cuda_peak_reserved_gib': 1.5,
                'parameters_m': parameters_m,
            }
            with mock.patch.object(
                protocol, 'validate_training_checkpoint',
                return_value={'model_state_dict': model.state_dict()},
            ), mock.patch.object(final_analysis, '_validate_eval_namespace'):
                row = final_analysis.validate_one_metric(
                    payload, metrics_path, job, names, log_root,
                )
        self.assertEqual(row['low_snr_auc'], low_summary['auc'])
        self.assertEqual(row['high_snr_auc'], high_summary['auc'])

    def test_launcher_seals_test_until_training_barrier(self):
        source = (
            protocol.REPO_ROOT / 'tools' / 'run_bc_tpro_noise8_final.sh'
        ).read_text(encoding='utf-8')
        self.assertIn('--skip_inprocess_validation 1', source)
        self.assertIn('--run_test_after_train 0', source)
        self.assertIn('--split test', source)
        self.assertIn('--profile "$PROFILE"', source)
        self.assertIn('--train_amp 0', source)
        self.assertIn('--eval_amp 0', source)
        self.assertIn('--upstream_compat 1', source)
        self.assertNotIn('--amp', source)
        self.assertIn('verify-frozen', source)
        self.assertNotIn('--train_sequence_list', source)
        self.assertNotIn('--val_sequence_list', source)
        self.assertNotIn('--sequence_list', source)
        evaluation_body = source.split('run_evaluation() {', 1)[1].split(
            '\n}\n', 1,
        )[0]
        self.assertLess(
            evaluation_body.index('verify_frozen_plan'),
            evaluation_body.index('> "$attempted_file"'),
        )
        self.assertLess(
            source.rfind('verify_all_training'),
            source.rfind('run_evaluation "$row"'),
        )

    def test_final_protocol_code_contains_no_hash_generation_or_checks(self):
        paths = (
            'tools/run_bc_tpro_noise8_final.sh',
            'tools/validate_bc_tpro_noise8_final.py',
            'tools/analyze_bc_tpro_noise8_final.py',
        )
        forbidden = ('hashlib', 'sha256', 'sha-256', 'md5', 'checksum')
        for relative in paths:
            source = (protocol.REPO_ROOT / relative).read_text(
                encoding='utf-8',
            ).lower()
            for token in forbidden:
                self.assertNotIn(
                    token, source, '%s contains %s' % (relative, token),
                )

    def test_final_analyzer_does_not_consume_legacy_detection_metrics(self):
        source = (
            protocol.REPO_ROOT / 'tools' / 'analyze_bc_tpro_noise8_final.py'
        ).read_text(encoding='utf-8').lower()
        for token in ('pixel_f1', 'pixel_iou', 'auc_dense_grid'):
            self.assertNotIn(token, source)


if __name__ == '__main__':
    unittest.main()
