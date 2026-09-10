import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
EXPERIMENT = (
    REPO_ROOT / 'experiments'
    / 'bc_tpro_stage1_noise8_upstream_2026-09-09'
)
DATA_ROOT = REPO_ROOT.parent / 'datasets' / 'NUDT-MIRSDT-Noise8.0_FJY'


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


setup = load_module(
    'validate_bc_tpro_noise8_setup_upstream_test',
    'validate_bc_tpro_noise8_setup.py',
)
analysis = load_module(
    'analyze_bc_tpro_noise8_stage1_upstream_test',
    'analyze_bc_tpro_noise8_stage1.py',
)
exact_evaluator = load_module(
    'evaluate_bc_tpro_noise8_exact_logit_upstream_test',
    'evaluate_bc_tpro_noise8_exact_logit.py',
)
exact_analysis = load_module(
    'analyze_bc_tpro_noise8_exact_logit_upstream_test',
    'analyze_bc_tpro_noise8_exact_logit.py',
)
paper_analysis = load_module(
    'analyze_bc_tpro_noise8_paper_upstream_test',
    'analyze_bc_tpro_noise8_paper.py',
)


class Noise8UpstreamProfileTests(unittest.TestCase):
    def test_profiles_have_disjoint_log_identity(self):
        modern = setup.expected_manifest_rows(setup.MODERNIZED_PROFILE)
        upstream = setup.expected_manifest_rows(setup.UPSTREAM_PROFILE)
        self.assertEqual(len(modern), 12)
        self.assertEqual(len(upstream), 12)
        self.assertTrue(all('Upstream8fa1a68-FP32' not in row['log_dir']
                            for row in modern))
        self.assertTrue(all('Upstream8fa1a68-FP32' in row['log_dir']
                            for row in upstream))
        self.assertTrue(
            set(row['log_dir'] for row in modern).isdisjoint(
                row['log_dir'] for row in upstream
            )
        )

    def test_registered_config_and_manifest_validate_only_as_upstream(self):
        setup.validate_protocol_config(
            EXPERIMENT / 'UPSTREAM_PROTOCOL.json', setup.UPSTREAM_PROFILE,
        )
        setup.validate_manifest(
            EXPERIMENT / 'manifest.tsv', setup.UPSTREAM_PROFILE,
        )
        with self.assertRaisesRegex(ValueError, '4x3 design'):
            setup.validate_manifest(
                EXPERIMENT / 'manifest.tsv', setup.MODERNIZED_PROFILE,
            )

    def test_analyzer_profile_requires_fp32_and_upstream_flag(self):
        job = analysis.expected_manifest_rows(analysis.UPSTREAM_PROFILE)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_list = root / 'train.txt'
            val_list = root / 'val.txt'
            training_log = root / 'training.txt'
            train_list.touch()
            val_list.touch()
            values = {
                'seed': int(job['seed']), 'gpu': job['gpu'], 'gpu_num': 1,
                'model': analysis.MODEL,
                'structure_variant': job['structure_variant'],
                'structure_bottleneck_channels': 8,
                'dataset': analysis.DATASET, 'batch_size': 4,
                'gradient_accumulation_steps': 1, 'epoch': 32,
                'learning_rate': 0.001, 'optimizer': 'Adam',
                'decay_rate': 0.0001, 'step_size': 10, 'lr_decay': 0.7,
                'seqlen': 40, 'patch_size': 128, 'sample_rate': 0.1,
                'sequence_augmentation': 0, 'loss': 'soft_iou',
                'threshold_eval': 0.5, 'train_amp': 0, 'eval_amp': 0,
                'eval_chunk_rows': 32, 'eval_interval': 8,
                'skip_inprocess_validation': 1,
                'early_stopping_patience': 0, 'train_workers': 4,
                'val_workers': 1, 'prefetch_factor': 2, 'deterministic': 1,
                'log_dir': job['log_dir'], 'resume': 'never',
                'resume_checkpoint': None, 'run_test_after_train': 0,
                'base_ckpt': '', 'spatial_ckpt': '', 'st_ckpt': '',
                'freeze_pretrained': 0, 'use_swanlab': 1,
                'swanlab_project': 'DeepPro-BC-TPro',
                'swanlab_group': (
                    'bc-tpro-stage1-noise8-upstream8fa1a68-fp32-scratch'
                ),
                'swanlab_mode': 'cloud', 'swanlab_resume': 'never',
                'upstream_compat': 1, 'datapath': str(DATA_ROOT.resolve()),
                'train_sequence_list': str(train_list.resolve()),
                'val_sequence_list': str(val_list.resolve()),
                'savepath': str(root.resolve()),
            }
            training_log.write_text(
                repr(argparse.Namespace(**values)) + '\n'
                + 'Initialized %s from random weights; no base checkpoint loaded.\n'
                % analysis.MODEL,
                encoding='utf-8',
            )
            analysis.validate_training_provenance(
                values, training_log, job, DATA_ROOT.resolve(),
                train_list.resolve(), val_list.resolve(),
                (root / 'sem_seg').resolve(), analysis.UPSTREAM_PROFILE,
            )
            wrong = dict(values, train_amp=1)
            with self.assertRaisesRegex(ValueError, 'train_amp mismatch'):
                analysis.validate_training_provenance(
                    wrong, training_log, job, DATA_ROOT.resolve(),
                    train_list.resolve(), val_list.resolve(),
                    (root / 'sem_seg').resolve(), analysis.UPSTREAM_PROFILE,
                )
            missing = dict(values)
            missing.pop('upstream_compat')
            with self.assertRaisesRegex(ValueError, 'lacks upstream_compat'):
                analysis.validate_training_provenance(
                    missing, training_log, job, DATA_ROOT.resolve(),
                    train_list.resolve(), val_list.resolve(),
                    (root / 'sem_seg').resolve(), analysis.UPSTREAM_PROFILE,
                )

    def test_dry_run_is_fp32_and_never_launches_training(self):
        if not (DATA_ROOT / 'train.txt').is_file():
            self.skipTest('Noise8 dataset is unavailable.')
        environment = os.environ.copy()
        environment['DRY_RUN'] = '1'
        result = subprocess.run(
            ['bash', str(TOOLS / 'run_bc_tpro_stage1_noise8_upstream.sh')],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        train_lines = [line for line in result.stdout.splitlines()
                       if line.startswith('TRAIN ')]
        eval_lines = [line for line in result.stdout.splitlines()
                      if line.startswith('EVAL ')]
        self.assertEqual(len(train_lines), 12)
        self.assertEqual(len(eval_lines), 12)
        self.assertTrue(all('--upstream_compat 1' in line for line in train_lines))
        self.assertTrue(all('--train_amp 0' in line and '--eval_amp 0' in line
                            for line in train_lines))
        self.assertTrue(all('Upstream8fa1a68-FP32' in line
                            for line in train_lines + eval_lines))
        self.assertTrue(all(' --amp' not in line for line in eval_lines))
        self.assertIn('profile=upstream8fa1a68_fp32', result.stdout)

    def test_exact_evaluator_precision_is_profile_locked(self):
        values = dict(
            epoch=32, seqlen=40, eval_chunk_rows=32, test_workers=1,
            prefetch_factor=1, cudnn_deterministic=0, cudnn_benchmark=0,
            low_fa_cap=5e-5,
        )
        exact_evaluator.validate_frozen_evaluation_args(SimpleNamespace(
            profile=analysis.UPSTREAM_PROFILE, amp=False, **values
        ))
        with self.assertRaisesRegex(ValueError, 'requires amp=False'):
            exact_evaluator.validate_frozen_evaluation_args(SimpleNamespace(
                profile=analysis.UPSTREAM_PROFILE, amp=True, **values
            ))
        exact_evaluator.validate_frozen_evaluation_args(SimpleNamespace(
            profile=analysis.MODERNIZED_PROFILE, amp=True, **values
        ))

    def test_exact_analyzer_rejects_cross_profile_precision(self):
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
        exact_analysis.validate_frozen_evaluation_payload(
            payload, 'none', analysis.UPSTREAM_PROFILE,
        )
        with self.assertRaisesRegex(ValueError, 'amp mismatch'):
            exact_analysis.validate_frozen_evaluation_payload(
                payload, 'none', analysis.MODERNIZED_PROFILE,
            )

    def test_paper_artifact_log_paths_are_profile_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modern = paper_analysis.required_artifacts(
                root / 'modern', root / 'logs', analysis.MODERNIZED_PROFILE,
            )
            upstream = paper_analysis.required_artifacts(
                root / 'upstream', root / 'logs', analysis.UPSTREAM_PROFILE,
            )
        modern_logs = [str(path) for role, path in modern
                       if role.startswith('training log')]
        upstream_logs = [str(path) for role, path in upstream
                         if role.startswith('training log')]
        self.assertEqual(len(modern), 36)
        self.assertEqual(len(upstream), 36)
        self.assertTrue(all('Upstream8fa1a68-FP32' not in path
                            for path in modern_logs))
        self.assertTrue(all('Upstream8fa1a68-FP32' in path
                            for path in upstream_logs))
        self.assertTrue(set(modern_logs).isdisjoint(upstream_logs))

    def test_upstream_paper_lock_binds_upstream_plan_and_inputs(self):
        decisions = []
        for index, variant in enumerate(paper_analysis.CANDIDATES):
            decisions.append({
                'variant': variant,
                'variant_label': analysis.VARIANTS[variant][3],
                'qualified': False,
                'mean_pd_at_0_5': 0.8,
                'mean_fa_at_0_5': 1e-5,
                'mean_paper_auc': 0.5,
                'mean_delta_paper_auc': -0.1,
                'mean_fa_at_0_5_relative_reduction': 0.1,
                'mean_delta_pd_at_0_5': 0.0,
                'joint_nonworse_seed_count': 3,
                'max_latency_ratio': 1.0,
                'parameters_m': 0.07 + index * 0.001,
                'complete_finite_metrics': True,
                'mean_delta_pd_at_0_5_ge_minus_1pp': True,
                'mean_fa_at_0_5_relative_reduction_gt_zero': True,
                'at_least_two_joint_nonworse_seeds': True,
                'mean_paper_auc_not_below_b1': False,
                'every_seed_latency_le_1p3': True,
                'identity_verified': True,
            })
        selection = paper_analysis.rank_qualified_candidates(decisions)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = paper_analysis.build_locked_candidate_payload(
                selection, decisions, root / 'upstream', root / 'logs',
                analysis.UPSTREAM_PROFILE,
            )
        self.assertTrue(payload['protocol_path'].endswith(
            '/OFFICIAL_METRIC_AMENDMENT_2026-09-10.md'
        ))
        self.assertEqual(
            payload['detection_metric_contract'],
            analysis.official_metric_contract(),
        )
        training_inputs = [
            item['path'] for item in payload['selector_inputs']
            if item['role'].startswith('training log')
        ]
        self.assertEqual(len(training_inputs), 12)
        self.assertTrue(all('Upstream8fa1a68-FP32' in path
                            for path in training_inputs))

    def test_exact_logit_dry_run_is_upstream_fp32(self):
        if not (DATA_ROOT / 'train.txt').is_file():
            self.skipTest('Noise8 dataset is unavailable.')
        environment = os.environ.copy()
        environment['DRY_RUN'] = '1'
        result = subprocess.run(
            ['bash', str(TOOLS / 'run_bc_tpro_noise8_exact_logit_upstream.sh')],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        exact_lines = [line for line in result.stdout.splitlines()
                       if line.startswith('EXACT_LOGIT ')]
        analyze_lines = [line for line in result.stdout.splitlines()
                         if line.startswith('ANALYZE ')]
        self.assertEqual(len(exact_lines), 18)
        self.assertEqual(len(analyze_lines), 1)
        self.assertTrue(all('--profile upstream8fa1a68_fp32' in line
                            for line in exact_lines + analyze_lines))
        self.assertTrue(all('Upstream8fa1a68-FP32' in line
                            for line in exact_lines))
        self.assertTrue(all(' --amp' not in line for line in exact_lines))


if __name__ == '__main__':
    unittest.main()
