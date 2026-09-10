#!/usr/bin/env python3
"""CPU-only tests for the dedicated Noise8 exact-logit evaluator."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ShootingRules import ShootingRules
from tools import evaluate_bc_tpro_noise8_exact_logit as exact


class Noise8ExactLogitEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.logits = np.asarray([
            [[-2.0, 0.2, -1.0], [0.7, -0.4, -0.8], [-3.0, -2.0, -1.0]],
            [[-1.0, -0.5, -0.3], [-0.2, 1.2, -0.1], [0.4, -2.0, -3.0]],
        ], dtype=np.float32)
        self.targets = np.zeros_like(self.logits)
        self.targets[0, 2, 2] = 1
        self.targets[1, 1, 1] = 1

    def test_two_pass_counts_match_shooting_rules(self):
        stream = [('Sequence9', self.logits, self.targets)]
        pixels = np.asarray([self.logits.size], dtype=np.int64)
        first = exact.collect_pass_one(iter(stream), ['Sequence9'], pixels, 4)
        thresholds, sentinel = exact.build_exact_thresholds(first)
        second = exact.collect_pass_two(
            iter(stream), ['Sequence9'], pixels, first.targets_by_sequence,
            thresholds,
        )
        expected_false = np.zeros(thresholds.size, dtype=np.int64)
        expected_true = np.zeros(thresholds.size, dtype=np.int64)
        for logits, target in zip(self.logits, self.targets):
            false, true, _total = ShootingRules().evaluate_thresholds(
                logits[None], target[None], thresholds,
            )
            expected_false += false
            expected_true += true
        np.testing.assert_array_equal(second.false_counts[:, 0], expected_false)
        np.testing.assert_array_equal(second.true_counts[:, 0], expected_true)
        endpoint = int(np.flatnonzero(thresholds == sentinel)[0])
        self.assertEqual(int(second.false_counts[endpoint, 0]), 0)
        self.assertEqual(int(second.true_counts[endpoint, 0]), 0)

    def test_temporary_cache_replays_identical_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = list(exact.cache_inference_stream(
                iter([('Sequence9', self.logits, self.targets)]),
                ['Sequence9'], Path(directory),
            ))
            replayed = list(exact.cached_sequence_stream(
                Path(directory), ['Sequence9'],
            ))
        np.testing.assert_array_equal(cached[0][1], replayed[0][1])
        np.testing.assert_array_equal(cached[0][2], replayed[0][2])

    def test_all_target_events_make_fixed_workpoints_exact(self):
        stream = [('Sequence9', self.logits, self.targets)]
        pixels = np.asarray([self.logits.size], dtype=np.int64)
        first = exact.collect_pass_one(iter(stream), ['Sequence9'], pixels, 1)
        target_thresholds, _ = exact.build_exact_thresholds(first)
        target_counts = exact.collect_pass_two(
            iter(stream), ['Sequence9'], pixels, first.targets_by_sequence,
            target_thresholds,
        )
        all_background = []
        for logits, target in zip(self.logits, self.targets):
            _peaks, false_values = exact.prepared_frame_events(logits, target)
            all_background.append(false_values)
        exhaustive = exact.PassOneResult(
            target_peaks=first.target_peaks,
            target_peak_sequence_indices=first.target_peak_sequence_indices,
            top_background_scores=np.concatenate(all_background),
            targets_by_sequence=first.targets_by_sequence,
            false_region_pixels_by_sequence=first.false_region_pixels_by_sequence,
            global_max_logit=first.global_max_logit,
        )
        exhaustive_thresholds, _ = exact.build_exact_thresholds(exhaustive)
        exhaustive_counts = exact.collect_pass_two(
            iter(stream), ['Sequence9'], pixels, first.targets_by_sequence,
            exhaustive_thresholds,
        )
        reference = {
            'false_pixels': 2,
            'true_targets': 1,
            'total_targets': int(first.targets_by_sequence.sum()),
            'pixels': int(pixels.sum()),
        }
        _, target_fa, target_pd = exact.compute_workpoint_summary(
            target_counts, target_thresholds, reference,
        )
        _, exhaustive_fa, exhaustive_pd = exact.compute_workpoint_summary(
            exhaustive_counts, exhaustive_thresholds, reference,
        )
        self.assertEqual(target_fa['true_targets'], exhaustive_fa['true_targets'])
        self.assertEqual(target_fa['false_pixels'], exhaustive_fa['false_pixels'])
        self.assertEqual(target_pd['true_targets'], exhaustive_pd['true_targets'])
        self.assertEqual(target_pd['false_pixels'], exhaustive_pd['false_pixels'])

    def test_pass_two_rejects_changed_target_count(self):
        pixels = np.asarray([self.logits.size], dtype=np.int64)
        first = exact.collect_pass_one(
            iter([('Sequence9', self.logits, self.targets)]),
            ['Sequence9'], pixels, 2,
        )
        thresholds, _ = exact.build_exact_thresholds(first)
        changed = np.zeros_like(self.targets)
        with self.assertRaisesRegex(RuntimeError, 'Target counts changed'):
            exact.collect_pass_two(
                iter([('Sequence9', self.logits, changed)]), ['Sequence9'],
                pixels, first.targets_by_sequence, thresholds,
            )

    def test_frozen_evaluation_rejects_non_amp(self):
        args = SimpleNamespace(
            epoch=32, seqlen=40, eval_chunk_rows=32, test_workers=1,
            prefetch_factor=1, cudnn_deterministic=0, cudnn_benchmark=0,
            amp=False, low_fa_cap=5e-5,
        )
        with self.assertRaisesRegex(ValueError, 'requires AMP'):
            exact.validate_frozen_evaluation_args(args)


if __name__ == '__main__':
    unittest.main()
