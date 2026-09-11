#!/usr/bin/env python3
"""Pure-CPU checks for the exact-event BC-TPro logit evaluator."""

import unittest

import numpy as np

from ShootingRules import ShootingRules
from tools import evaluate_bc_tpro_exact_logit as exact


class ExactLogitEvaluatorTests(unittest.TestCase):
    def test_shooting_rule_casts_threshold_like_official_scalar_comparison(self):
        output = np.zeros((1, 10, 10), dtype=np.float32)
        output[0, 5, 5] = np.float32(0.35)
        target = np.zeros_like(output)
        false_counts, true_counts, target_counts = (
            ShootingRules().evaluate_thresholds(
                output, target, np.asarray([0.35], dtype=np.float64)
            )
        )
        self.assertEqual(false_counts.tolist(), [1])
        self.assertEqual(true_counts.tolist(), [0])
        self.assertEqual(target_counts.tolist(), [0])

    def test_target_peaks_and_counts_match_shooting_rules(self):
        logits = np.asarray([
            [-3.0, -2.0, -1.0, -4.0, -5.0],
            [-2.0, 0.4, 0.4, -3.0, -5.0],
            [-1.0, 0.1, 0.2, -2.0, 0.8],
            [-4.0, -3.0, -2.0, -1.0, 0.8],
            [-5.0, -5.0, -4.0, -3.0, -2.0],
        ], dtype=np.float32)
        target = np.zeros_like(logits)
        target[2, 2] = 1
        thresholds = np.asarray([-1.0, 0.4, 0.8, 0.81], dtype=np.float64)

        peaks, false_values = exact.prepared_frame_events(logits, target)
        self.assertEqual(peaks.tolist(), [np.float32(0.4)])
        self.assertLess(false_values.size, logits.size)
        false_counts, true_counts, _ = exact.frame_counts_at_thresholds(
            logits, target, thresholds,
        )
        expected_false, expected_true, _ = ShootingRules().evaluate_thresholds(
            logits[None], target[None], thresholds,
        )
        np.testing.assert_array_equal(false_counts, expected_false)
        np.testing.assert_array_equal(true_counts, expected_true)

    def test_streaming_topk_merge_preserves_ties(self):
        accumulator = exact.StreamingTopK(5)
        accumulator.update(np.asarray([1, 7, 5, 5], dtype=np.float32))
        accumulator.update(np.asarray([9, 5, -2, 8], dtype=np.float32))
        accumulator.update(np.asarray([5, 4], dtype=np.float32))
        self.assertEqual(accumulator.seen, 10)
        np.testing.assert_array_equal(
            accumulator.values(),
            np.asarray([5, 5, 7, 8, 9], dtype=np.float32),
        )

        tie_accumulator = exact.StreamingTopK(3)
        tie_accumulator.update(np.asarray([2, 2, 2, 2, 1], dtype=np.float32))
        np.testing.assert_array_equal(
            tie_accumulator.values(), np.asarray([2, 2, 2], dtype=np.float32),
        )

    def test_raw_logit_stitch_uses_max_even_when_all_values_are_negative(self):
        stitched = np.full((4, 1, 2), -np.inf, dtype=np.float32)
        first = np.asarray([
            [[-4.0, -3.0]], [[-5.0, -2.0]], [[-8.0, -7.0]],
        ], dtype=np.float32)
        second = np.asarray([
            [[-6.0, -1.0]], [[-9.0, -6.0]], [[-3.0, -4.0]],
        ], dtype=np.float32)
        exact.merge_raw_logit_window(stitched, first, 0)
        exact.merge_raw_logit_window(stitched, second, 1)
        np.testing.assert_array_equal(
            stitched[:, 0],
            np.asarray([
                [-4.0, -3.0], [-5.0, -1.0], [-8.0, -6.0], [-3.0, -4.0],
            ], dtype=np.float32),
        )

    def test_topk_boundary_threshold_counts_all_global_ties(self):
        logits = np.full((1, 15, 15), np.float32(0.7), dtype=np.float32)
        targets = np.zeros_like(logits)
        targets[0, 7, 7] = 1
        stream = [('ties', logits, targets)]
        pass_one = exact.collect_pass_one(
            iter(stream), ['ties'], np.asarray([logits.size]), retain_k=3,
        )
        self.assertEqual(pass_one.top_background_scores.size, 3)
        thresholds, _ = exact.build_exact_thresholds(pass_one)
        counts = exact.collect_pass_two(
            iter(stream), ['ties'], np.asarray([logits.size]),
            pass_one.targets_by_sequence, pass_one.logits_sha256_by_sequence,
            pass_one.targets_sha256_by_sequence, thresholds,
        )
        boundary = float(pass_one.top_background_scores[0])
        boundary_index = int(np.flatnonzero(thresholds == boundary)[0])
        self.assertGreater(int(counts.false_counts[boundary_index, 0]), 3)

    def test_empty_target_and_finite_sentinel_endpoints(self):
        logits = np.asarray([[-3.0, -2.0], [-1.0, -4.0]], dtype=np.float32)
        target = np.zeros_like(logits)
        stream = [('empty', logits[None], target[None])]
        pass_one = exact.collect_pass_one(
            iter(stream), ['empty'], np.asarray([4]), retain_k=2,
        )
        thresholds, sentinel = exact.build_exact_thresholds(pass_one)
        self.assertTrue(np.isfinite(sentinel))
        self.assertGreater(sentinel, float(logits.max()))
        self.assertIn(0.0, thresholds)
        pass_two = exact.collect_pass_two(
            iter(stream), ['empty'], np.asarray([4]), np.asarray([0]),
            pass_one.logits_sha256_by_sequence,
            pass_one.targets_sha256_by_sequence, thresholds,
        )
        np.testing.assert_array_equal(pass_two.true_counts, 0)
        sentinel_index = int(np.flatnonzero(thresholds == sentinel)[0])
        self.assertEqual(int(pass_two.false_counts[sentinel_index, 0]), 0)

    def test_two_pass_sequence_pairing_and_integer_counts_are_exact(self):
        logits_a = np.asarray([
            [[0.5, 0.5, -1.0], [0.1, -0.2, -0.3], [-2.0, -3.0, -4.0]],
            [[-0.5, -0.4, -0.3], [-0.2, 0.9, -0.1], [-1.0, -2.0, -3.0]],
        ], dtype=np.float32)
        target_a = np.zeros_like(logits_a)
        target_a[0, 2, 2] = 1
        target_a[1, 1, 1] = 1
        logits_b = np.asarray([
            [[-1.0, 0.2], [0.2, 1.2]],
        ], dtype=np.float32)
        target_b = np.zeros_like(logits_b)
        stream = [
            ('a', logits_a, target_a),
            ('b', logits_b, target_b),
        ]
        pixels = np.asarray([logits_a.size, logits_b.size], dtype=np.int64)
        pass_one = exact.collect_pass_one(
            iter(stream), ['a', 'b'], pixels, retain_k=5,
        )
        thresholds, sentinel = exact.build_exact_thresholds(pass_one)
        pass_two = exact.collect_pass_two(
            iter(stream), ['a', 'b'], pixels, pass_one.targets_by_sequence,
            pass_one.logits_sha256_by_sequence,
            pass_one.targets_sha256_by_sequence, thresholds,
        )
        self.assertEqual(pass_two.false_counts.dtype, np.int64)
        self.assertEqual(pass_two.true_counts.dtype, np.int64)
        self.assertEqual(pass_two.false_counts.shape, (thresholds.size, 2))
        for sequence_index, (_name, logits, targets) in enumerate(stream):
            expected_false = np.zeros(thresholds.size, dtype=np.int64)
            expected_true = np.zeros(thresholds.size, dtype=np.int64)
            for frame_logits, frame_target in zip(logits, targets):
                peaks, false_values = exact.prepared_frame_events(
                    frame_logits, frame_target
                )
                expected_false += np.count_nonzero(
                    false_values.astype(np.float64)[:, None]
                    >= thresholds[None, :],
                    axis=0,
                )
                expected_true += np.count_nonzero(
                    peaks.astype(np.float64)[:, None]
                    >= thresholds[None, :],
                    axis=0,
                )
            np.testing.assert_array_equal(
                pass_two.false_counts[:, sequence_index], expected_false,
            )
            np.testing.assert_array_equal(
                pass_two.true_counts[:, sequence_index], expected_true,
            )
            self.assertEqual(
                int(pass_two.targets_by_sequence[sequence_index]),
                sum(
                    exact.prepared_frame_events(frame_logits, frame_target)[0].size
                    for frame_logits, frame_target in zip(logits, targets)
                ),
            )
        sentinel_index = int(np.flatnonzero(thresholds == sentinel)[0])
        self.assertEqual(int(pass_two.false_counts[sentinel_index].sum()), 0)
        self.assertEqual(int(pass_two.true_counts[sentinel_index].sum()), 0)

    def test_second_pass_rejects_nonidentical_inference_replay(self):
        logits = np.asarray([[[0.1, 0.2]]], dtype=np.float32)
        targets = np.zeros_like(logits)
        first = exact.collect_pass_one(
            iter([('s', logits, targets)]), ['s'], np.asarray([2]), retain_k=1,
        )
        thresholds, _ = exact.build_exact_thresholds(first)
        changed = logits.copy()
        changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(np.inf))
        with self.assertRaisesRegex(RuntimeError, 'bitwise replay mismatch'):
            exact.collect_pass_two(
                iter([('s', changed, targets)]), ['s'], np.asarray([2]),
                first.targets_by_sequence, first.logits_sha256_by_sequence,
                first.targets_sha256_by_sequence, thresholds,
            )

    def test_target_event_grid_matches_exhaustive_high_fa_workpoints(self):
        rng = np.random.RandomState(20260909)
        logits = rng.normal(size=(3, 15, 15)).astype(np.float32)
        targets = np.zeros_like(logits)
        targets[0, 2, 2] = 1
        targets[1, 7, 7] = 1
        targets[2, 12, 12] = 1
        stream = [('s', logits, targets)]
        pixels = np.asarray([logits.size], dtype=np.int64)
        pass_one = exact.collect_pass_one(
            iter(stream), ['s'], pixels, retain_k=3,
        )
        targeted_thresholds, _ = exact.build_exact_thresholds(pass_one)
        targeted_counts = exact.collect_pass_two(
            iter(stream), ['s'], pixels, pass_one.targets_by_sequence,
            pass_one.logits_sha256_by_sequence,
            pass_one.targets_sha256_by_sequence, targeted_thresholds,
        )

        all_false_parts = []
        for frame_logits, frame_target in zip(logits, targets):
            _peaks, false_values = exact.prepared_frame_events(
                frame_logits, frame_target,
            )
            all_false_parts.append(false_values)
        exhaustive_pass = exact.PassOneResult(
            target_peaks=pass_one.target_peaks,
            target_peak_sequence_indices=pass_one.target_peak_sequence_indices,
            top_background_scores=np.concatenate(all_false_parts),
            targets_by_sequence=pass_one.targets_by_sequence,
            false_region_pixels_by_sequence=(
                pass_one.false_region_pixels_by_sequence
            ),
            logits_sha256_by_sequence=pass_one.logits_sha256_by_sequence,
            targets_sha256_by_sequence=pass_one.targets_sha256_by_sequence,
            global_max_logit=pass_one.global_max_logit,
        )
        exhaustive_thresholds, _ = exact.build_exact_thresholds(exhaustive_pass)
        exhaustive_counts = exact.collect_pass_two(
            iter(stream), ['s'], pixels, pass_one.targets_by_sequence,
            pass_one.logits_sha256_by_sequence,
            pass_one.targets_sha256_by_sequence, exhaustive_thresholds,
        )
        exhaustive_zero = int(np.flatnonzero(exhaustive_thresholds == 0.0)[0])
        reference = {
            'false_pixels': int(exhaustive_counts.false_counts[exhaustive_zero].sum()),
            'true_targets': int(exhaustive_counts.true_counts[exhaustive_zero].sum()),
            'total_targets': int(exhaustive_counts.targets_by_sequence.sum()),
            'pixels': int(exhaustive_counts.pixels_by_sequence.sum()),
        }
        _zero, targeted_fixed_fa, targeted_fixed_pd = (
            exact.compute_workpoint_summary(
                targeted_counts, targeted_thresholds, reference,
            )
        )
        _zero, exhaustive_fixed_fa, exhaustive_fixed_pd = (
            exact.compute_workpoint_summary(
                exhaustive_counts, exhaustive_thresholds, reference,
            )
        )
        self.assertEqual(
            targeted_fixed_fa['true_targets'],
            exhaustive_fixed_fa['true_targets'],
        )
        self.assertEqual(
            targeted_fixed_pd['false_pixels'],
            exhaustive_fixed_pd['false_pixels'],
        )


if __name__ == '__main__':
    unittest.main()
