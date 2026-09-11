import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import train


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainingMetricPolicyTests(unittest.TestCase):
    def test_new_bctpro_internal_split_requires_every_epoch_validation(self):
        values = dict(
            model='DeepPro-Plus_BCTPro',
            dataset='NUDT-MIRSDT-Noise8.0_FJY',
            train_sequence_list='/tmp/train.txt',
            val_sequence_list='/tmp/val.txt',
            eval_interval=1,
            skip_inprocess_validation=0,
            validation_safe_cudnn=1,
            validation_overlap_policy='official_window',
            eval_chunk_rows=32,
            early_stopping_patience=0,
            early_stopping_metric='eval_iou',
            run_test_after_train=0,
        )
        train.validate_bctpro_validation_schedule(
            Namespace(**values), environment={}
        )
        train.validate_bctpro_validation_schedule(
            Namespace(
                **dict(values, model='DeepPro-Plus_BCTPro_Experimental')
            ),
            environment={},
        )
        for changed in (
            dict(eval_interval=8),
            dict(skip_inprocess_validation=1),
            dict(validation_safe_cudnn=0),
            dict(validation_overlap_policy='sequence_max'),
            dict(eval_chunk_rows=0),
            dict(early_stopping_patience=2),
            dict(early_stopping_metric='eval_f1'),
            dict(run_test_after_train=1),
        ):
            with self.subTest(changed=changed):
                invalid = dict(values, **changed)
                with self.assertRaisesRegex(
                    ValueError, 'must use --eval_interval 1'
                ):
                    train.validate_bctpro_validation_schedule(
                        Namespace(**invalid), environment={}
                    )

    def test_frozen_external_only_and_final80_are_explicit_exceptions(self):
        frozen = Namespace(
            model='DeepPro-Plus_BCTPro',
            dataset='NUDT-MIRSDT-Noise8.0_FJY',
            train_sequence_list='/tmp/train.txt',
            val_sequence_list='/tmp/val.txt',
            eval_interval=8,
            skip_inprocess_validation=1,
            validation_safe_cudnn=0,
            validation_overlap_policy='official_window',
            eval_chunk_rows=32,
            early_stopping_patience=0,
            early_stopping_metric='eval_iou',
            run_test_after_train=0,
        )
        train.validate_bctpro_validation_schedule(
            frozen,
            environment={'CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION': '1'},
        )
        invalid_frozen = Namespace(
            **dict(vars(frozen), early_stopping_patience=2)
        )
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            train.validate_bctpro_validation_schedule(
                invalid_frozen,
                environment={
                    'CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION': '1'
                },
            )
        final80 = Namespace(
            model='DeepPro-Plus_BCTPro',
            dataset='NUDT-MIRSDT-Noise8.0_FJY',
            train_sequence_list=None,
            val_sequence_list=None,
            eval_interval=8,
            skip_inprocess_validation=1,
            validation_safe_cudnn=0,
            validation_overlap_policy='official_window',
            eval_chunk_rows=32,
            early_stopping_patience=0,
            early_stopping_metric='eval_iou',
            run_test_after_train=0,
        )
        train.validate_bctpro_validation_schedule(final80, environment={})
        unsafe_followup_test = Namespace(
            **dict(vars(final80), run_test_after_train=1)
        )
        with self.assertRaisesRegex(ValueError, 'test-isolated launcher'):
            train.validate_bctpro_validation_schedule(
                unsafe_followup_test, environment={}
            )

        unsafe_default_split = Namespace(
            model='DeepPro-Plus_BCTPro',
            dataset='NUDT-MIRSDT-Noise8.0_FJY',
            train_sequence_list=None,
            val_sequence_list=None,
            eval_interval=1,
            skip_inprocess_validation=0,
            validation_safe_cudnn=1,
            validation_overlap_policy='official_window',
            eval_chunk_rows=32,
            early_stopping_patience=0,
            early_stopping_metric='eval_iou',
            run_test_after_train=0,
        )
        with self.assertRaisesRegex(
            ValueError, 'requires explicit --train_sequence_list'
        ):
            train.validate_bctpro_validation_schedule(
                unsafe_default_split, environment={}
            )
        non_nudt = Namespace(
            **dict(vars(unsafe_default_split), dataset='SatVideoIRSTD')
        )
        train.validate_bctpro_validation_schedule(non_nudt, environment={})

    def test_train_defaults_validate_every_epoch(self):
        with mock.patch.object(sys, 'argv', ['train.py']):
            args = train.parse_args()
        self.assertEqual(args.eval_interval, 1)
        self.assertEqual(args.skip_inprocess_validation, 0)
        self.assertEqual(args.validation_overlap_policy, 'official_window')
        self.assertEqual(args.early_stopping_metric, 'eval_iou')

    def test_checkpoint_records_best_validation_epoch_and_current_metrics(self):
        detector = train.torch.nn.Linear(2, 1)
        optimizer = train.torch.optim.SGD(detector.parameters(), lr=0.1)
        grad_scaler = mock.Mock()
        grad_scaler.state_dict.return_value = {}
        state = train.make_checkpoint_state(
            detector,
            optimizer,
            grad_scaler,
            epoch=8,
            best_iou=0.42,
            args=Namespace(model='DeepPro-Plus_BCTPro'),
            config={},
            best_epoch=7,
            validation_metrics={
                'epoch': 9,
                'iou': 0.40,
                'overlap_policy': 'official_window',
            },
        )
        self.assertEqual(
            state['checkpoint_selection'],
            {
                'metric': 'eval_iou',
                'mode': 'max',
                'overlap_policy': 'official_window',
                'best_value': 0.42,
                'best_epoch': 7,
            },
        )
        self.assertEqual(
            state['validation_metrics'],
            {
                'epoch': 9,
                'iou': 0.40,
                'overlap_policy': 'official_window',
            },
        )

    def test_official_validation_repeats_overlap_frames(self):
        class QueuedDetector(train.torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.outputs = [
                    train.torch.tensor([[[[10.0]], [[-10.0]]]]),
                    train.torch.tensor([[[[10.0]], [[-10.0]]]]),
                ]

            def forward(self, _images):
                return None, self.outputs.pop(0)

        class ZeroLoss:
            def __call__(self, logits, _targets, **_kwargs):
                return logits.sum() * 0.0

        def validation_inputs():
            images = train.torch.zeros(1, 1, 2, 1, 1)
            centroids = train.torch.zeros(1, 2, 1, 1)
            return [
                (
                    images.clone(),
                    train.torch.tensor([[[[0.0]], [[1.0]]]]),
                    centroids.clone(),
                    [train.torch.tensor([0]), train.torch.tensor([1])],
                ),
                (
                    images.clone(),
                    train.torch.tensor([[[[1.0]], [[0.0]]]]),
                    centroids.clone(),
                    [train.torch.tensor([1]), train.torch.tensor([2])],
                ),
            ]

        def counts(policy):
            return train.evaluate_sequences(
                QueuedDetector(),
                ZeroLoss(),
                [[None, None]],
                validation_inputs(),
                train.torch.device('cpu'),
                threshold=0.5,
                epoch=0,
                show_progress=False,
                overlap_policy=policy,
            )[2].tolist()

        self.assertEqual(counts('official_window'), [1, 2, 2])
        self.assertEqual(counts('sequence_max'), [1, 2, 1])

    def test_safe_validation_cudnn_context_restores_training_flags(self):
        original_deterministic = train.torch.backends.cudnn.deterministic
        original_benchmark = train.torch.backends.cudnn.benchmark
        try:
            train.torch.backends.cudnn.deterministic = True
            train.torch.backends.cudnn.benchmark = False
            with train.validation_cudnn_context(True):
                self.assertFalse(train.torch.backends.cudnn.deterministic)
                self.assertFalse(train.torch.backends.cudnn.benchmark)
            self.assertTrue(train.torch.backends.cudnn.deterministic)
            self.assertFalse(train.torch.backends.cudnn.benchmark)
        finally:
            train.torch.backends.cudnn.deterministic = original_deterministic
            train.torch.backends.cudnn.benchmark = original_benchmark

    def test_safe_validation_cudnn_context_restores_flags_after_error(self):
        original_deterministic = train.torch.backends.cudnn.deterministic
        original_benchmark = train.torch.backends.cudnn.benchmark
        try:
            train.torch.backends.cudnn.deterministic = True
            train.torch.backends.cudnn.benchmark = True
            with self.assertRaisesRegex(RuntimeError, 'validation failure'):
                with train.validation_cudnn_context(True):
                    self.assertFalse(train.torch.backends.cudnn.deterministic)
                    self.assertFalse(train.torch.backends.cudnn.benchmark)
                    raise RuntimeError('validation failure')
            self.assertTrue(train.torch.backends.cudnn.deterministic)
            self.assertTrue(train.torch.backends.cudnn.benchmark)
        finally:
            train.torch.backends.cudnn.deterministic = original_deterministic
            train.torch.backends.cudnn.benchmark = original_benchmark

    def test_evaluation_call_uses_and_restores_safe_cudnn_policy(self):
        original_deterministic = train.torch.backends.cudnn.deterministic
        original_benchmark = train.torch.backends.cudnn.benchmark

        def assert_safe_policy(*_args, **_kwargs):
            self.assertFalse(train.torch.backends.cudnn.deterministic)
            self.assertFalse(train.torch.backends.cudnn.benchmark)
            return 'validation-result'

        try:
            train.torch.backends.cudnn.deterministic = True
            train.torch.backends.cudnn.benchmark = True
            with mock.patch.object(
                train,
                'evaluate_sequences',
                side_effect=assert_safe_policy,
            ):
                result = train.evaluate_sequences_with_cudnn_policy(
                    True, 'detector'
                )
            self.assertEqual(result, 'validation-result')
            self.assertTrue(train.torch.backends.cudnn.deterministic)
            self.assertTrue(train.torch.backends.cudnn.benchmark)
        finally:
            train.torch.backends.cudnn.deterministic = original_deterministic
            train.torch.backends.cudnn.benchmark = original_benchmark

    def test_validation_loader_does_not_advance_training_rng(self):
        sequence = train.torch.utils.data.TensorDataset(
            train.torch.tensor([1])
        )
        args = Namespace(
            skip_inprocess_validation=0,
            dataset='unit-test',
            val_sequence_list='/tmp/val.txt',
            val_workers=0,
            prefetch_factor=2,
            seed=47,
        )
        runtime = Namespace(rank=0, world_size=1)
        original_state = train.torch.get_rng_state()
        try:
            train.torch.manual_seed(1234)
            state_before = train.torch.get_rng_state().clone()
            with mock.patch.object(
                train, 'TestIRSeqDataLoader', return_value=[sequence]
            ):
                _, _, loader = train.build_validation_data(
                    args, runtime, '/tmp', 40
                )
                list(loader)
            self.assertTrue(
                train.torch.equal(state_before, train.torch.get_rng_state())
            )
        finally:
            train.torch.set_rng_state(original_state)

    def test_early_stopping_defaults_to_iou(self):
        with mock.patch.object(sys, 'argv', ['train.py']):
            self.assertEqual(train.parse_args().early_stopping_metric, 'eval_iou')

    def test_upstream_swanlab_omits_pixel_classification_diagnostics(self):
        metrics = {
            'train/loss': 0.4,
            'train/iou': 0.5,
            'train/precision': 0.6,
            'train/recall': 0.7,
            'train/f1': 0.65,
            'train/lr': 0.001,
            'train/loss_component/soft_iou': 0.4,
            'eval/loss': 0.3,
            'eval/iou': 0.55,
            'eval/precision': 0.61,
            'eval/recall': 0.71,
            'eval/f1': 0.66,
            'eval/best_iou': 0.55,
        }

        filtered = train.filter_swanlab_metrics_for_protocol(
            metrics, upstream_compat=True
        )

        self.assertEqual(
            set(filtered),
            {
                'train/loss',
                'train/iou',
                'train/lr',
                'train/loss_component/soft_iou',
                'eval/loss',
                'eval/iou',
                'eval/best_iou',
            },
        )
        self.assertEqual(len(metrics), 13)

    def test_non_upstream_swanlab_payload_is_unchanged(self):
        metrics = {
            'train/loss': 0.4,
            'train/iou': 0.5,
            'train/f1': 0.65,
        }

        filtered = train.filter_swanlab_metrics_for_protocol(
            metrics, upstream_compat=False
        )

        self.assertEqual(filtered, metrics)
        self.assertIsNot(filtered, metrics)

    def test_current_launchers_explicitly_select_iou(self):
        for relative_path in (
            'tools/run_bc_tpro_stage1_noise8.sh',
            'tools/run_bc_tpro_noise8_final.sh',
        ):
            script = (REPO_ROOT / relative_path).read_text(encoding='utf-8')
            self.assertIn('--early_stopping_metric eval_iou', script)

    def test_completed_stage1_launchers_mark_the_frozen_schedule(self):
        for relative_path in (
            'tools/run_bc_tpro_stage1.sh',
            'tools/run_bc_tpro_stage1_noise8.sh',
        ):
            script = (REPO_ROOT / relative_path).read_text(encoding='utf-8')
            self.assertIn(
                'CSIG_ALLOW_FROZEN_EXTERNAL_ONLY_VALIDATION=1', script
            )


if __name__ == '__main__':
    unittest.main()
