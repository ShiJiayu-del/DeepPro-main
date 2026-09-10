"""CPU checks for the controlled stage-1 raw-evidence BC-TPro models."""

import importlib
import math
import unittest

import torch

from networks.layers.bc_tpro_adapter import (
    BCTProAdapter,
    _ring_average,
    _unit_l2_zero_dc_response,
)


VARIANTS = (
    "none",
    "temporal_control",
    "center_multiscale",
    "center_ring",
)


def _model_module(name):
    return importlib.import_module("networks.models." + name)


class BCTProStage1Tests(unittest.TestCase):
    def test_adapter_shape_and_finite(self):
        raw = torch.randn(2, 1, 7, 9, 11)
        features = torch.randn(2, 32, 7, 9, 11)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                adapter = BCTProAdapter(variant)
                output, auxiliary = adapter(
                    features, raw, return_aux=True
                )
                self.assertEqual(output.shape, features.shape)
                self.assertTrue(torch.isfinite(output).all())
                delta = auxiliary["residual_delta"]
                if delta is not None:
                    self.assertTrue(torch.isfinite(delta).all())

    def test_evidence_depends_on_raw_frames_not_post_tpro_features(self):
        torch.manual_seed(5)
        adapter = BCTProAdapter("center_ring")
        torch.nn.init.normal_(adapter.residual_projection.weight, std=0.03)
        raw = torch.randn(1, 1, 7, 8, 9)
        first_features = torch.randn(1, 32, 7, 8, 9)
        second_features = torch.randn_like(first_features)
        first_output, first_aux = adapter(
            first_features, raw, return_aux=True
        )
        second_output, second_aux = adapter(
            second_features, raw, return_aux=True
        )
        self.assertTrue(torch.equal(
            first_aux["evidence"], second_aux["evidence"]
        ))
        self.assertTrue(torch.equal(
            first_aux["fused_evidence"],
            second_aux["fused_evidence"],
        ))
        self.assertTrue(torch.equal(
            first_aux["residual_delta"],
            second_aux["residual_delta"],
        ))
        self.assertTrue(torch.allclose(
            first_output - first_features,
            second_output - second_features,
            atol=2e-7,
            rtol=1e-5,
        ))

    def test_fixed_kernel_is_zero_dc_and_unit_l2(self):
        signal = torch.zeros(1, 1, 9, 1, 1)
        signal[:, :, 4] = 1.0
        valid = torch.ones(1, 1, 9, 1, 1, dtype=torch.bool)
        response, usable = _unit_l2_zero_dc_response(
            signal, valid, kernel_size=5
        )
        expected_kernel = torch.tensor([
            -1.0 / math.sqrt(20.0),
            -1.0 / math.sqrt(20.0),
            math.sqrt(4.0 / 5.0),
            -1.0 / math.sqrt(20.0),
            -1.0 / math.sqrt(20.0),
        ])
        observed_kernel = response[0, 0, 2:7, 0, 0]
        self.assertTrue(torch.allclose(
            observed_kernel, expected_kernel, atol=1e-6, rtol=1e-6
        ))
        self.assertAlmostEqual(observed_kernel.sum().item(), 0.0, places=6)
        self.assertAlmostEqual(
            observed_kernel.square().sum().item(), 1.0, places=6
        )
        self.assertTrue(usable.all())

        boundary_signal = torch.zeros(1, 1, 5, 1, 1)
        boundary_signal[:, :, 0] = 1.0
        boundary_valid = torch.ones(
            1, 1, 5, 1, 1, dtype=torch.bool
        )
        boundary_response, _ = _unit_l2_zero_dc_response(
            boundary_signal, boundary_valid, kernel_size=5
        )
        self.assertAlmostEqual(
            boundary_response[0, 0, 0, 0, 0].item(),
            math.sqrt(2.0 / 3.0),
            places=6,
        )

    def test_c1_c2_constant_raw_sequence_has_zero_response(self):
        raw_frame = torch.randn(1, 1, 1, 8, 9)
        raw = raw_frame.expand(-1, -1, 7, -1, -1).clone()
        features = torch.randn(1, 32, 7, 8, 9)
        for variant in ("center_multiscale", "center_ring"):
            with self.subTest(variant=variant):
                _, auxiliary = BCTProAdapter(variant)(
                    features, raw, return_aux=True
                )
                self.assertTrue(torch.allclose(
                    auxiliary["evidence"],
                    torch.zeros_like(auxiliary["evidence"]),
                    atol=3e-6,
                    rtol=0,
                ))

    def test_ring_average_has_no_constant_boundary_artifact(self):
        constant = torch.full((1, 1, 3, 6, 7), 2.5)
        ring = _ring_average(constant)
        self.assertTrue(torch.allclose(ring, constant, atol=1e-6, rtol=0))
        edge_impulse = torch.zeros(1, 1, 3, 6, 7)
        edge_impulse[..., 0, 0] = 1
        ring = _ring_average(edge_impulse)
        self.assertEqual(ring.shape, edge_impulse.shape)
        self.assertTrue(torch.isfinite(ring).all())

    def test_raw_evidence_is_time_reversal_equivariant(self):
        raw = torch.randn(1, 1, 9, 8, 9)
        variants = (
            "temporal_control",
            "center_multiscale",
            "center_ring",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                torch.manual_seed(7)
                adapter = BCTProAdapter(variant)
                forward = adapter.prepare_evidence(raw)
                reversed_output = adapter.prepare_evidence(
                    raw.flip(2)
                ).flip(2)
                self.assertTrue(torch.allclose(
                    forward,
                    reversed_output,
                    atol=3e-6,
                    rtol=1e-5,
                ))

    def test_common_initialization_matches_deeppro_plus_elementwise(self):
        baseline_module = _model_module("DeepPro-Plus")
        bctpro_module = _model_module("DeepPro-Plus_BCTPro")
        torch.manual_seed(47)
        baseline = baseline_module.detector(1, seqlen=5, out_len=5)
        torch.manual_seed(47)
        candidate = bctpro_module.detector(
            1,
            seqlen=5,
            out_len=5,
            structure_variant="center_ring",
        )
        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        common_prefixes = (
            "conv_in.",
            "layer1.",
            "TPro.",
            "conv_out1.",
            "conv_out2.",
        )
        common_keys = [
            key for key in baseline_state if key.startswith(common_prefixes)
        ]
        self.assertTrue(common_keys)
        for key in common_keys:
            self.assertIn(key, candidate_state)
            self.assertTrue(torch.equal(
                baseline_state[key], candidate_state[key]
            ), key)

    def test_zero_initialized_adapter_preserves_initial_logits(self):
        baseline_module = _model_module("DeepPro-Plus")
        bctpro_module = _model_module("DeepPro-Plus_BCTPro")
        images = torch.randn(1, 1, 5, 7, 8)
        variants = (
            "temporal_control",
            "center_multiscale",
            "center_ring",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                torch.manual_seed(49)
                baseline = baseline_module.detector(
                    1, seqlen=5, out_len=5
                ).eval()
                torch.manual_seed(49)
                candidate = bctpro_module.detector(
                    1,
                    seqlen=5,
                    out_len=5,
                    structure_variant=variant,
                ).eval()
                with torch.no_grad():
                    baseline_features, baseline_logits = baseline(images)
                    candidate_features, candidate_logits = candidate(images)
                self.assertTrue(torch.equal(
                    baseline_features, candidate_features
                ))
                self.assertTrue(torch.equal(
                    baseline_logits, candidate_logits
                ))

    def test_two_steps_open_zero_initialized_projection_gradient_path(self):
        torch.manual_seed(51)
        adapter = BCTProAdapter("center_ring")
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        raw = torch.randn(2, 1, 7, 8, 9)
        features = torch.randn(2, 32, 7, 8, 9)
        target = torch.randn_like(features)
        fusion_weight = adapter.evidence_fusion[0].weight
        projection_weight = adapter.residual_projection.weight

        optimizer.zero_grad()
        loss = (adapter(features, raw) - target).square().mean()
        loss.backward()
        self.assertGreater(projection_weight.grad.abs().sum(), 0)
        self.assertEqual(fusion_weight.grad.abs().sum(), 0)
        optimizer.step()

        optimizer.zero_grad()
        loss = (adapter(features, raw) - target).square().mean()
        loss.backward()
        self.assertGreater(fusion_weight.grad.abs().sum(), 0)

    def test_eval_full_and_chunked_rows_match(self):
        module = _model_module("DeepPro-Plus_BCTPro")
        images = torch.randn(1, 1, 5, 8, 9)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(53)
                full = module.detector(
                    1,
                    seqlen=5,
                    out_len=5,
                    structure_variant=variant,
                    eval_chunk_rows=0,
                ).eval()
                chunked = module.detector(
                    1,
                    seqlen=5,
                    out_len=5,
                    structure_variant=variant,
                    eval_chunk_rows=3,
                ).eval()
                chunked.load_state_dict(full.state_dict())
                if variant != "none":
                    torch.nn.init.normal_(
                        full.bc_tpro.residual_projection.weight,
                        std=0.02,
                    )
                    full.bc_tpro.residual_projection.bias.data.normal_(
                        std=0.02
                    )
                    chunked.load_state_dict(full.state_dict())
                with torch.no_grad():
                    full_features, full_logits = full(images)
                    chunk_features, chunk_logits = chunked(images)
                self.assertTrue(torch.allclose(
                    full_features,
                    chunk_features,
                    atol=3e-6,
                    rtol=1e-5,
                ))
                self.assertTrue(torch.allclose(
                    full_logits,
                    chunk_logits,
                    atol=3e-6,
                    rtol=1e-5,
                ))

    def test_c0_and_c1_have_identical_trainable_parameter_count(self):
        c0 = BCTProAdapter("temporal_control")
        c1 = BCTProAdapter("center_multiscale")
        c0_count = sum(parameter.numel() for parameter in c0.parameters())
        c1_count = sum(parameter.numel() for parameter in c1.parameters())
        self.assertEqual(c0_count, c1_count)


if __name__ == "__main__":
    unittest.main()
