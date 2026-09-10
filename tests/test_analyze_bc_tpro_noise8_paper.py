import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import analyze_bc_tpro_noise8_paper as paper


def synthetic_rows():
    probability_rows = []
    parameters = {}
    settings = {
        'none': (0.900, 0.00010, 1.00, 0.900, 0.070913),
        'temporal_control': (0.910, 0.00008, 1.10, 0.920, 0.071233),
        'center_multiscale': (0.925, 0.00005, 1.15, 0.930, 0.071233),
        'center_ring': (0.920, 0.00006, 1.20, 0.910, 0.071281),
    }
    for variant in paper.VARIANTS:
        pd, fa, latency, auc, params = settings[variant]
        for seed in paper.SEEDS:
            run_id = next(
                row['run_id'] for row in paper.grid_analysis.expected_manifest_rows()
                if row['structure_variant'] == variant and int(row['seed']) == seed
            )
            probability_rows.append({
                'run_id': run_id, 'variant': variant, 'seed': seed,
                'pd_at_0_5': pd, 'fa_at_0_5': fa,
                'delta_pd_at_0_5': pd - 0.900,
                'delta_fa_at_0_5': fa - 0.00010,
                'fa_at_0_5_relative_reduction': (0.00010 - fa) / 0.00010,
                'auc': auc, 'delta_auc': auc - 0.900,
                'latency_ratio': latency,
                # Deliberately present but forbidden from selection.
                'pixel_f1_at_0_5': 0.01 if variant == 'center_multiscale' else 0.99,
            })
            parameters[run_id] = params
    return probability_rows, parameters


def synthetic_log(epochs=32, duplicate_loss_epoch=None):
    lines = []
    for epoch in range(1, epochs + 1):
        lines.extend([
            'INFO **** Epoch %d/32 ****' % epoch,
            'INFO Training mean loss: %.6f' % (1.0 / epoch),
            'INFO Training accuracy (IoU) of prediction: %.6f' % (epoch / 64.0),
            'INFO Training pixel F1: %.6f' % (epoch / 40.0),
        ])
        if epoch == duplicate_loss_epoch:
            lines.append('INFO Training mean loss: 0.123456')
    return '\n'.join(lines) + '\n'


class Noise8PaperAnalysisTests(unittest.TestCase):
    def test_incomplete_barrier_refuses_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                paper.IncompleteEvidenceError, 'refusing to read partial'
            ):
                paper.require_complete_artifacts(
                    root / 'experiment', root / 'log' / 'sem_seg'
                )

    def test_curve_parser_requires_exactly_32_complete_epochs(self):
        job = paper.grid_analysis.expected_manifest_rows()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.log'
            path.write_text(synthetic_log(), encoding='utf-8')
            rows = paper.parse_training_curve(path, job)
            self.assertEqual(len(rows), 32)
            self.assertEqual(rows[-1]['epoch'], 32)
            self.assertNotIn('train_pixel_f1', rows[-1])

            path.write_text(synthetic_log(epochs=31), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Epoch 1..32'):
                paper.parse_training_curve(path, job)

            path.write_text(
                synthetic_log(duplicate_loss_epoch=7), encoding='utf-8'
            )
            with self.assertRaisesRegex(ValueError, 'exactly one train_loss'):
                paper.parse_training_curve(path, job)

    def test_curve_summary_uses_three_seed_sample_sd_and_plots(self):
        rows = []
        for variant in paper.VARIANTS:
            for epoch in paper.EXPECTED_EPOCHS:
                for offset, seed in enumerate(paper.SEEDS):
                    rows.append({
                        'variant': variant,
                        'variant_label': paper.grid_analysis.VARIANTS[variant][3],
                        'seed': seed,
                        'epoch': epoch,
                        'train_loss': float(epoch + offset),
                        'train_iou': 0.1 + 0.01 * offset,
                    })
        summary = paper.summarize_training_curves(rows)
        self.assertEqual(len(summary), 4 * 32)
        first = summary[0]
        self.assertAlmostEqual(first['train_loss_mean'], 2.0)
        self.assertAlmostEqual(first['train_loss_sd'], 1.0)
        self.assertAlmostEqual(
            first['train_iou_sd'], np.std([0.10, 0.11, 0.12], ddof=1)
        )
        with tempfile.TemporaryDirectory() as directory:
            png = Path(directory) / 'curves.png'
            pdf = Path(directory) / 'curves.pdf'
            paper.plot_training_curves(summary, png, pdf)
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)

    def test_selection_uses_joint_pareto_objectives_and_never_pixel_f1(self):
        probability_rows, parameters = synthetic_rows()
        decisions = paper.evaluate_candidate_eligibility(
            probability_rows, parameters, identity_verified=True,
        )
        selection = paper.rank_qualified_candidates(decisions)
        self.assertEqual(selection['status'], 'CANDIDATE_LOCKED')
        self.assertEqual(selection['selected_variant'], 'center_multiscale')
        # C1 has deliberately worst Pixel F1 but dominates all official metrics.
        self.assertEqual(selection['qualified_ranking'][0], 'center_multiscale')

    def test_single_good_seed_cannot_qualify_candidate(self):
        probability_rows, parameters = synthetic_rows()
        c2 = [row for row in probability_rows if row['variant'] == 'center_ring']
        for row in c2[1:]:
            row['delta_pd_at_0_5'] = -0.005
            row['delta_fa_at_0_5'] = 0.00001
            row['pd_at_0_5'] = 0.895
            row['fa_at_0_5'] = 0.00011
        decisions = paper.evaluate_candidate_eligibility(
            probability_rows, parameters, identity_verified=True,
        )
        decision = next(
            row for row in decisions if row['variant'] == 'center_ring'
        )
        self.assertEqual(decision['joint_nonworse_seed_count'], 1)
        self.assertFalse(decision['qualified'])

    def test_missing_seed_is_rejected_before_selection(self):
        probability_rows, parameters = synthetic_rows()
        removed = probability_rows.pop()
        with self.assertRaisesRegex(ValueError, 'all 12'):
            paper.evaluate_candidate_eligibility(
                probability_rows, parameters, identity_verified=True,
            )
        self.assertEqual(removed['seed'], 51)

    def test_no_eligible_candidate_falls_back_to_b1(self):
        probability_rows, parameters = synthetic_rows()
        for row in probability_rows:
            if row['variant'] != 'none':
                row['delta_auc'] = -0.1
        decisions = paper.evaluate_candidate_eligibility(
            probability_rows, parameters, identity_verified=True,
        )
        selection = paper.rank_qualified_candidates(decisions)
        self.assertEqual(selection['status'], 'B1_FALLBACK')
        self.assertEqual(selection['selected_variant'], 'none')

    def test_equal_metrics_remain_unresolved_regardless_of_parameters(self):
        probability_rows, parameters = synthetic_rows()
        for row in probability_rows:
            if row['variant'] in ('temporal_control', 'center_multiscale'):
                row['pd_at_0_5'] = 0.91
                row['fa_at_0_5'] = 0.00008
                row['delta_pd_at_0_5'] = 0.01
                row['delta_fa_at_0_5'] = -0.00002
                row['fa_at_0_5_relative_reduction'] = 0.2
                row['auc'] = 0.92
                row['delta_auc'] = 0.02
        for run_id in list(parameters):
            if 'center_multiscale' in run_id:
                parameters[run_id] = 0.071500
        decisions = paper.evaluate_candidate_eligibility(
            probability_rows, parameters, identity_verified=True,
        )
        selection = paper.rank_qualified_candidates(decisions)
        self.assertEqual(selection['status'], 'UNRESOLVED_TRADEOFF')
        self.assertIsNone(selection['selected_variant'])
        self.assertIn('temporal_control', selection['nondominated_candidates'])
        self.assertIn('center_multiscale', selection['nondominated_candidates'])

    def test_report_and_selector_table_exclude_f1(self):
        probability_rows, parameters = synthetic_rows()
        decisions = paper.evaluate_candidate_eligibility(
            probability_rows, parameters, identity_verified=True,
        )
        selection = paper.rank_qualified_candidates(decisions)
        curves = []
        for variant in paper.VARIANTS:
            for epoch in (1, 32):
                curves.append({
                    'variant': variant,
                    'train_loss_mean': 1.0 / epoch,
                    'train_iou_mean': epoch / 64.0,
                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / 'report.md'
            table = root / 'qualification.csv'
            paper.write_report(report, decisions, selection, curves)
            paper.write_csv(table, decisions)
            self.assertNotIn('f1', report.read_text(encoding='utf-8').lower())
            self.assertNotIn(
                'f1', table.read_text(encoding='utf-8').splitlines()[0].lower()
            )

    def test_locked_candidate_schema_is_fail_closed_and_machine_readable(self):
        probability_rows, parameters = synthetic_rows()
        decisions = paper.evaluate_candidate_eligibility(
            probability_rows, parameters, identity_verified=True,
        )
        selection = paper.rank_qualified_candidates(decisions)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = paper.build_locked_candidate_payload(
                selection, decisions, root / 'experiment',
                root / 'log' / 'sem_seg',
            )
        self.assertEqual(payload['schema_version'], 2)
        self.assertTrue(payload['selection_complete'])
        self.assertEqual(payload['model'], 'DeepPro-Plus_BCTPro')
        self.assertEqual(payload['selected_variant'], 'center_multiscale')
        self.assertEqual(payload['locked_model']['role'], 'candidate')
        self.assertEqual(payload['baseline']['selected_variant'], 'none')
        self.assertEqual(payload['seeds'], [47, 49, 51])
        self.assertFalse(payload['selection_uses_single_seed'])
        self.assertFalse(payload['official_test_accessed'])
        self.assertEqual(
            payload['detection_metric_contract'],
            paper.grid_analysis.official_metric_contract(),
        )
        self.assertNotIn('f1', repr(payload).lower())
        self.assertTrue(payload['eligible_candidates'])
        self.assertTrue(payload['selector_inputs'])
        self.assertEqual(
            set(payload), {
                'schema_version', 'status', 'selection_complete',
                'protocol_path', 'locked_model', 'model',
                'selected_variant', 'baseline', 'seeds',
                'eligible_candidates', 'candidate_qualification',
                'detection_metric_contract', 'selection_rule',
                'selector_inputs', 'selection_uses_single_seed',
                'official_test_accessed',
            }
        )
        self.assertEqual(len(payload['selector_inputs']), 36)

        incomplete = dict(selection, status='UNRESOLVED_TRADEOFF')
        with self.assertRaisesRegex(ValueError, 'completed candidate'):
            paper.build_locked_candidate_payload(
                incomplete, decisions, '/tmp/experiment', '/tmp/log/sem_seg'
            )

        inconsistent = dict(selection, selected_variant='center_ring')
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            paper.build_locked_candidate_payload(
                inconsistent, decisions, '/tmp/experiment', '/tmp/log/sem_seg'
            )


if __name__ == '__main__':
    unittest.main()
