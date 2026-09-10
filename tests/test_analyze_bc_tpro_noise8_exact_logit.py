#!/usr/bin/env python3
"""CPU-only tests for the Noise8 exact-logit gate analyzer."""

import copy
import unittest

import numpy as np

from tools import analyze_bc_tpro_noise8_exact_logit as analysis


def fake_output():
    arrays = {
        'thresholds': np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
        'sequence_names': np.asarray(['Sequence9']),
        'false_pixels_by_threshold_sequence': np.asarray([[4], [2], [0]], dtype=np.int64),
        'true_targets_by_threshold_sequence': np.asarray([[2], [1], [0]], dtype=np.int64),
        'total_targets_by_threshold_sequence': np.asarray([[2], [2], [2]], dtype=np.int64),
        'pixel_count_by_threshold_sequence': np.asarray([[9], [9], [9]], dtype=np.int64),
    }
    payload = {
        'workpoint_at_logit_zero': {
            'false_pixels': 2, 'true_targets': 1, 'total_targets': 2,
            'pixels': 9, 'false_pixels_by_sequence': [2],
            'true_targets_by_sequence': [1],
        },
        'reference_workpoint': {
            'false_pixels': 2, 'true_targets': 1, 'total_targets': 2, 'pixels': 9,
        },
        'pd_at_reference_fa': {'available': True, 'false_pixels': 2, 'true_targets': 1},
        'fa_at_reference_pd': {'available': True, 'false_pixels': 2, 'true_targets': 1},
    }
    return {'arrays': arrays, 'payload': payload}


class Noise8ExactLogitAnalysisTests(unittest.TestCase):
    def test_reference_workpoint_cannot_be_rewritten(self):
        payload = fake_output()['payload']
        payload['reference'] = {'workpoint': dict(payload['reference_workpoint'])}
        analysis.validate_embedded_reference_budget(payload)
        payload['reference_workpoint']['false_pixels'] = 1
        with self.assertRaisesRegex(ValueError, 'declared B1 budget'):
            analysis.validate_embedded_reference_budget(payload)

    def test_frozen_payload_rejects_non_amp(self):
        payload = {
            'inference': {
                'amp': False, 'eval_chunk_rows': 32, 'test_workers': 1,
                'prefetch_factor': 1, 'cudnn_deterministic': False,
                'cudnn_benchmark': False,
            },
            'retention': {'low_fa_cap': 5e-5},
            'checkpoint': {'model_config': {
                'eval_chunk_rows': 32, 'structure_variant': 'none',
                'structure_bottleneck_channels': 8,
            }},
        }
        with self.assertRaisesRegex(ValueError, 'amp mismatch'):
            analysis.validate_frozen_evaluation_payload(payload, 'none')

    def test_repeat_comparison_is_field_by_field(self):
        primary = fake_output()
        repeated = copy.deepcopy(primary)
        matched, differences = analysis.compare_repeat_counts(primary, repeated)
        self.assertTrue(matched)
        self.assertEqual(differences, [])
        repeated['arrays']['true_targets_by_threshold_sequence'][1, 0] = 0
        matched, differences = analysis.compare_repeat_counts(primary, repeated)
        self.assertFalse(matched)
        self.assertIn('true_targets_by_threshold_sequence values', differences)

    def test_c2_gate_requires_all_declared_checks(self):
        rows = []
        for variant in analysis.VARIANTS:
            for seed in analysis.SEEDS:
                if variant == 'none':
                    pd, fa = 0.90, 1.0e-4
                elif variant == 'center_multiscale':
                    pd, fa = 0.91, 8.5e-5
                elif variant == 'center_ring':
                    pd, fa = 0.92, 7.0e-5
                else:
                    pd, fa = 0.90, 9.0e-5
                rows.append({
                    'variant': variant, 'seed': seed,
                    'pd_at_fixed_fa': pd, 'fa_at_fixed_pd': fa,
                    'delta_pd_at_fixed_fa': pd - 0.90,
                    'fa_at_fixed_pd_relative_reduction': (1.0e-4 - fa) / 1.0e-4,
                    'latency_ratio': 1.1,
                })
        status, details = analysis.c2_gate(rows, reproducible=True)
        self.assertEqual(status, 'PASS')
        self.assertTrue(all(detail.endswith('PASS') for detail in details))
        status, _ = analysis.c2_gate(rows, reproducible=False)
        self.assertEqual(status, 'FAIL')

    def test_exact_output_names_keep_repeats_separate(self):
        primary = analysis.exact_stem('none', 47)
        repeated = analysis.exact_stem('none', 47, repeat_index=1)
        self.assertNotEqual(primary, repeated)
        self.assertTrue(repeated.endswith('__repeat1'))


if __name__ == '__main__':
    unittest.main()
