"""CPU validation of parameter-matched, fixed-filter BC-TPro variants."""

import importlib
import math
import unittest

import torch

from networks.layers.bc_tpro_adapter import (
    BCTProAdapter,
    _cross_spatial_smooth,
    _unit_l2_temporal_bandpass,
)


VARIANTS = (
    "center_ring_difference",
    "temporal_bandpass",
    "center_spatial_smooth",
)


class BCTProNongateTests(unittest.TestCase):
    def test_parameter_count_and_common_initialization_match_c1(self):
        module = importlib.import_module("networks.models.DeepPro-Plus_BCTPro")
        torch.manual_seed(47)
        reference = module.detector(
            1, seqlen=40, out_len=40, structure_variant="center_multiscale"
        )
        reference_state = reference.state_dict()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(47)
                candidate = module.detector(
                    1, seqlen=40, out_len=40, structure_variant=variant
                )
                self.assertEqual(
                    sum(parameter.numel() for parameter in candidate.parameters()),
                    71233,
                )
                self.assertEqual(candidate.state_dict().keys(), reference_state.keys())
                for name, value in candidate.state_dict().items():
                    self.assertTrue(torch.equal(value, reference_state[name]), name)

    def test_constant_background_and_padding_have_zero_evidence(self):
        torch.manual_seed(8)
        frame = torch.randn(1, 1, 1, 6, 7)
        raw = frame.expand(-1, -1, 19, -1, -1).clone()
        raw[:, :, [0, 1, 9, 18]] = 0
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                _, auxiliary = BCTProAdapter(variant).prepare_evidence(
                    raw, return_aux=True
                )
                torch.testing.assert_close(
                    auxiliary["evidence"], torch.zeros_like(auxiliary["evidence"]),
                    atol=3e-6, rtol=0,
                )
                self.assertFalse(auxiliary["valid_mask"][:, :, [0, 1, 9, 18]].any())
                self.assertEqual(
                    torch.count_nonzero(auxiliary["evidence"][:, :, [0, 1, 9, 18]]),
                    0,
                )

    def test_short_and_all_padding_sequences_are_finite(self):
        for variant in VARIANTS:
            for length in (1, 2, 3):
                for all_padding in (False, True):
                    with self.subTest(variant=variant, length=length, padding=all_padding):
                        raw = torch.randn(1, 1, length, 1, 2)
                        if all_padding:
                            raw.zero_()
                        adapter = BCTProAdapter(variant)
                        _, auxiliary = adapter.prepare_evidence(raw, return_aux=True)
                        self.assertEqual(auxiliary["evidence"].shape, (1, 3, length, 1, 2))
                        self.assertTrue(torch.isfinite(auxiliary["evidence"]).all())
                        if all_padding or length == 1:
                            self.assertEqual(torch.count_nonzero(auxiliary["evidence"]), 0)
                        if all_padding:
                            self.assertFalse(auxiliary["valid_mask"].any())

    def test_bandpass_coefficients_match_actual_valid_sets(self):
        length = 19
        # Each batch element is one basis vector: output across the batch is
        # the complete coefficient vector for the queried temporal sample.
        basis = torch.eye(length, dtype=torch.float64).reshape(length, 1, length, 1, 1)
        patterns = (
            list(range(length)),
            list(range(2, 18)),
            [0, 2, 3, 6, 9, 10, 15, 18],
            [9],
        )
        for valid_indices in patterns:
            valid = torch.zeros(length, 1, length, 1, 1, dtype=torch.bool)
            valid[:, :, valid_indices] = True
            for kernel_size in (5, 9, 17):
                response, usable = _unit_l2_temporal_bandpass(basis, valid, kernel_size)
                for query in range(length):
                    with self.subTest(valid=valid_indices, kernel=kernel_size, query=query):
                        short = [t for t in valid_indices if abs(t - query) <= 1]
                        long = [t for t in valid_indices if abs(t - query) <= kernel_size // 2]
                        expected = torch.zeros(length, dtype=torch.float64)
                        expected_usable = query in valid_indices and len(long) > len(short) > 0
                        if expected_usable:
                            norm = math.sqrt(1.0 / len(short) - 1.0 / len(long))
                            expected[short] += 1.0 / len(short) / norm
                            expected[long] -= 1.0 / len(long) / norm
                        coefficients = response[:, 0, query, 0, 0]
                        torch.testing.assert_close(coefficients, expected, atol=1e-12, rtol=1e-12)
                        self.assertEqual(bool(usable[0, 0, query, 0, 0]), expected_usable)
                        self.assertAlmostEqual(coefficients.sum().item(), 0.0, places=12)
                        self.assertAlmostEqual(
                            coefficients.square().sum().item(), float(expected_usable), places=12
                        )

    def test_cross_filter_weights_and_replicate_boundaries(self):
        impulse = torch.zeros(1, 1, 1, 5, 5)
        impulse[..., 2, 2] = 1
        expected = impulse * 0.5
        expected[..., 1, 2] = 0.125
        expected[..., 3, 2] = 0.125
        expected[..., 2, 1] = 0.125
        expected[..., 2, 3] = 0.125
        self.assertTrue(torch.equal(_cross_spatial_smooth(impulse), expected))
        corner = torch.zeros_like(impulse)
        corner[..., 0, 0] = 1
        expected.zero_()
        expected[..., 0, 0] = 0.75
        expected[..., 1, 0] = 0.125
        expected[..., 0, 1] = 0.125
        self.assertTrue(torch.equal(_cross_spatial_smooth(corner), expected))
        for shape in ((2, 1, 3, 4, 6), (1, 1, 2, 1, 1)):
            constant = torch.full(shape, 2.5)
            self.assertTrue(torch.equal(_cross_spatial_smooth(constant), constant))

    def test_ring_difference_matches_existing_c2_difference_channels(self):
        raw = torch.randn(1, 1, 7, 6, 8)
        raw[:, :, 0] = 0
        _, old = BCTProAdapter("center_ring").prepare_evidence(raw, return_aux=True)
        _, new = BCTProAdapter("center_ring_difference").prepare_evidence(raw, return_aux=True)
        self.assertTrue(torch.equal(new["evidence"], old["evidence"][:, [2, 5, 8]]))
        self.assertTrue(torch.equal(new["valid_mask"], old["valid_mask"][:, [2, 5, 8]]))

    def test_zero_initial_logits_match_baseline_elementwise(self):
        baseline_module = importlib.import_module("networks.models.DeepPro-Plus")
        candidate_module = importlib.import_module("networks.models.DeepPro-Plus_BCTPro")
        raw = torch.randn(1, 1, 5, 7, 8)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(47)
                baseline = baseline_module.detector(1, seqlen=5, out_len=5).eval()
                torch.manual_seed(47)
                candidate = candidate_module.detector(
                    1, seqlen=5, out_len=5, structure_variant=variant
                ).eval()
                with torch.no_grad():
                    baseline_features, baseline_logits = baseline(raw)
                    candidate_features, candidate_logits = candidate(raw)
                self.assertTrue(torch.equal(baseline_features, candidate_features))
                self.assertTrue(torch.equal(baseline_logits, candidate_logits))

    def test_projection_then_upstream_fusion_receive_gradients(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(47)
                adapter = BCTProAdapter(variant)
                optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
                raw = torch.randn(2, 1, 7, 6, 8)
                features = torch.randn(2, 32, 7, 6, 8)
                target = torch.randn_like(features)
                fusion_weight = adapter.evidence_fusion[0].weight
                projection_weight = adapter.residual_projection.weight
                optimizer.zero_grad()
                (adapter(features, raw) - target).square().mean().backward()
                self.assertTrue(torch.isfinite(projection_weight.grad).all())
                self.assertGreater(projection_weight.grad.abs().sum(), 0)
                self.assertEqual(fusion_weight.grad.abs().sum(), 0)
                optimizer.step()
                optimizer.zero_grad()
                (adapter(features, raw) - target).square().mean().backward()
                self.assertTrue(torch.isfinite(fusion_weight.grad).all())
                self.assertGreater(fusion_weight.grad.abs().sum(), 0)

    def test_nonzero_projection_full_and_row_chunks_match(self):
        module = importlib.import_module("networks.models.DeepPro-Plus_BCTPro")
        raw = torch.randn(1, 1, 5, 8, 9)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                full = module.detector(
                    1, seqlen=5, out_len=5, structure_variant=variant, eval_chunk_rows=0
                ).eval()
                chunked = module.detector(
                    1, seqlen=5, out_len=5, structure_variant=variant, eval_chunk_rows=3
                ).eval()
                torch.nn.init.normal_(full.bc_tpro.residual_projection.weight, std=0.02)
                torch.nn.init.normal_(full.bc_tpro.residual_projection.bias, std=0.02)
                chunked.load_state_dict(full.state_dict())
                with torch.no_grad():
                    full_features, full_logits = full(raw)
                    chunk_features, chunk_logits = chunked(raw)
                torch.testing.assert_close(full_features, chunk_features, atol=3e-6, rtol=1e-5)
                torch.testing.assert_close(full_logits, chunk_logits, atol=3e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
