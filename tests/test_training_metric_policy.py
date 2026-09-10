import sys
import unittest
from pathlib import Path
from unittest import mock

import train


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainingMetricPolicyTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
