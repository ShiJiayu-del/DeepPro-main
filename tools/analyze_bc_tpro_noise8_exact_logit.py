#!/usr/bin/env python3
"""Validate Noise8 exact-logit outputs, compare gates, and report results."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = REPO_ROOT / 'tools'
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools import analyze_bc_tpro_noise8_stage1 as grid  # noqa: E402
from tools import analyze_bc_tpro_stage1 as common  # noqa: E402
from tools import evaluate_bc_tpro_noise8_exact_logit as exact  # noqa: E402


SEEDS = (47, 49, 51)
VARIANTS = {
    'none': ('b1', 'B1 DeepPro-Plus'),
    'temporal_control': ('c0', 'C0 temporal control'),
    'center_multiscale': ('c1', 'C1 center multiscale'),
    'center_ring': ('c2', 'C2 center+ring'),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--experiment-root', type=Path,
        default=REPO_ROOT / 'experiments' / 'bc_tpro_stage1_noise8_2026-09-09',
    )
    parser.add_argument(
        '--log-root', type=Path, default=REPO_ROOT / 'log' / 'sem_seg',
    )
    parser.add_argument(
        '--profile', choices=tuple(grid.PROFILES), default=grid.MODERNIZED_PROFILE,
    )
    return parser.parse_args(argv)


def expected_run_id(variant, seed):
    return '%s_%s_seed%d' % (VARIANTS[variant][0], variant, seed)


def exact_stem(variant, seed, repeat_index=0):
    suffix = '__repeat1' if repeat_index else ''
    return '%s__noise8_internal_val__exact_logit%s' % (
        expected_run_id(variant, seed), suffix,
    )


def _require_dict_counts_equal(context, observed, expected):
    fields = ('false_pixels', 'true_targets', 'total_targets', 'pixels')
    for field in fields:
        if field in expected and observed.get(field) != expected[field]:
            raise ValueError('%s %s mismatch.' % (context, field))
    for field in ('false_pixels_by_sequence', 'true_targets_by_sequence'):
        if field in expected and observed.get(field) != expected[field]:
            raise ValueError('%s %s mismatch.' % (context, field))
    for field in ('threshold_logit', 'fa', 'pd'):
        if field in expected:
            value = observed.get(field)
            if not isinstance(value, (int, float)) or not math.isclose(
                float(value), float(expected[field]), rel_tol=0.0, abs_tol=1e-15,
            ):
                raise ValueError('%s %s mismatch.' % (context, field))
    if 'available' in expected and observed.get('available') != expected['available']:
        raise ValueError('%s availability mismatch.' % context)


def validate_frozen_evaluation_payload(
    payload, variant, profile=grid.MODERNIZED_PROFILE,
):
    inference = payload.get('inference', {})
    expected_inference = {
        'amp': grid.PROFILES[profile]['inference_amp'],
        'eval_chunk_rows': 32,
        'test_workers': 1,
        'prefetch_factor': 1,
        'cudnn_deterministic': False,
        'cudnn_benchmark': False,
    }
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            raise ValueError('Frozen inference field %s mismatch.' % field)
    retention = payload.get('retention', {})
    if not math.isclose(
        float(retention.get('low_fa_cap', float('nan'))),
        5e-5, rel_tol=0.0, abs_tol=0.0,
    ):
        raise ValueError('Frozen low-Fa cap mismatch.')
    expected_config = {
        'eval_chunk_rows': 32,
        'structure_variant': variant,
        'structure_bottleneck_channels': 8,
    }
    if payload.get('checkpoint', {}).get('model_config') != expected_config:
        raise ValueError('Frozen checkpoint model_config mismatch.')


def validate_embedded_reference_budget(payload):
    reference = payload.get('reference')
    if reference is None:
        source = payload.get('workpoint_at_logit_zero', {})
    else:
        source = reference.get('workpoint', {})
    fields = ('false_pixels', 'true_targets', 'total_targets', 'pixels')
    expected = {field: source.get(field) for field in fields}
    if payload.get('reference_workpoint') != expected:
        raise ValueError('reference_workpoint differs from its declared B1 budget.')


def load_exact_output(
    path, npz_path, variant, seed, repeat_index, log_root=None,
    checkpoint_cache=None, profile=grid.MODERNIZED_PROFILE,
):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != exact.SCHEMA_VERSION:
        raise ValueError('%s has the wrong schema version.' % path)
    if payload.get('protocol', {}).get('name') != exact.PROTOCOL_NAME:
        raise ValueError('%s has the wrong protocol.' % path)
    payload_profile = payload.get('protocol', {}).get(
        'profile', grid.MODERNIZED_PROFILE,
    )
    if payload_profile != profile:
        raise ValueError('%s has the wrong protocol profile.' % path)
    if payload.get('protocol', {}).get('official_test_accessed') is not False:
        raise ValueError('%s does not attest official-test isolation.' % path)
    validate_frozen_evaluation_payload(payload, variant, profile)
    validate_embedded_reference_budget(payload)
    identity = payload.get('run_identity', {})
    expected_identity = {
        'profile': profile,
        'run_id': expected_run_id(variant, seed),
        'dataset': exact.DATASET,
        'model': exact.MODEL,
        'structure_variant': variant,
        'seed': seed,
        'checkpoint_epoch': 32,
        'sequence_length': 40,
        'repeat_index': repeat_index,
    }
    expected_job = {
        row['run_id']: row for row in grid.expected_manifest_rows(profile)
    }[expected_run_id(variant, seed)]
    expected_identity['log_dir'] = expected_job['log_dir']
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError('%s identity %s mismatch.' % (path, field))
    if Path(identity.get('datapath', '')).expanduser().resolve() != exact.EXPECTED_DATA_ROOT:
        raise ValueError('%s uses the wrong data root.' % path)
    if log_root is not None:
        expected_checkpoint = (
            Path(log_root).expanduser().resolve() / expected_job['log_dir']
            / 'checkpoints' / 'epoch_32_model.pth'
        ).resolve()
        if Path(payload.get('checkpoint', {}).get('path', '')).expanduser().resolve() != expected_checkpoint:
            raise ValueError('%s uses the wrong checkpoint path.' % path)
        if not expected_checkpoint.is_file():
            raise FileNotFoundError('Missing evaluated checkpoint: %s' % expected_checkpoint)
        checkpoint = common._load_checkpoint(
            expected_checkpoint,
            {} if checkpoint_cache is None else checkpoint_cache,
        )
        if checkpoint.get('model_name') != exact.MODEL:
            raise ValueError('%s checkpoint has the wrong model_name.' % path)
        if checkpoint.get('epoch') != 31:
            raise ValueError('%s checkpoint has the wrong epoch.' % path)
        expected_config = {
            'eval_chunk_rows': 32,
            'structure_variant': variant,
            'structure_bottleneck_channels': 8,
        }
        if checkpoint.get('model_config') != expected_config:
            raise ValueError('%s actual checkpoint model_config mismatch.' % path)
        grid.validate_model_state_dict(
            checkpoint.get('model_state_dict'), variant,
            '%s actual checkpoint' % path,
        )
    provenance = payload.get('training_provenance', {})
    provenance_profile = provenance.get('profile', grid.MODERNIZED_PROFILE)
    if provenance_profile != profile:
        raise ValueError('%s has the wrong training profile.' % path)
    if provenance.get('scratch_only_verified') is not True:
        raise ValueError('%s does not verify scratch-only training.' % path)
    descriptor = payload.get('data_descriptor', {})
    if descriptor.get('sequence_names') != list(exact.VAL_NAMES):
        raise ValueError('%s uses the wrong validation sequence order.' % path)
    if Path(descriptor.get('sequence_list_path', '')).expanduser().resolve() != (
        Path(path).parents[1] / 'splits' / 'val_sequences.txt'
    ).resolve():
        raise ValueError('%s uses the wrong validation-list path.' % path)
    if Path(payload.get('counts', {}).get('npz_path', '')).expanduser().resolve() != Path(npz_path).resolve():
        raise ValueError('%s points to the wrong count table.' % path)

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {
        'thresholds', 'sequence_names',
        'false_pixels_by_threshold_sequence',
        'true_targets_by_threshold_sequence',
        'total_targets_by_threshold_sequence',
        'pixel_count_by_threshold_sequence',
    }
    if not required.issubset(arrays):
        raise ValueError('%s count table lacks required arrays.' % npz_path)
    thresholds = arrays['thresholds']
    names = arrays['sequence_names'].tolist()
    false_counts = arrays['false_pixels_by_threshold_sequence']
    true_counts = arrays['true_targets_by_threshold_sequence']
    target_matrix = arrays['total_targets_by_threshold_sequence']
    pixel_matrix = arrays['pixel_count_by_threshold_sequence']
    if names != list(exact.VAL_NAMES):
        raise ValueError('%s count table has the wrong sequence order.' % npz_path)
    if thresholds.ndim != 1 or thresholds.size < 2:
        raise ValueError('%s has an invalid threshold vector.' % npz_path)
    if not np.isfinite(thresholds).all() or not np.all(np.diff(thresholds) > 0):
        raise ValueError('%s thresholds must be finite and strictly increasing.' % npz_path)
    expected_shape = (thresholds.size, len(exact.VAL_NAMES))
    for name, value in (
        ('false counts', false_counts), ('true counts', true_counts),
        ('target counts', target_matrix), ('pixel counts', pixel_matrix),
    ):
        if value.shape != expected_shape or value.dtype != np.int64:
            raise ValueError('%s %s has the wrong shape or dtype.' % (npz_path, name))
        if np.any(value < 0):
            raise ValueError('%s %s contains negative values.' % (npz_path, name))
    if not np.all(np.diff(false_counts, axis=0) <= 0):
        raise ValueError('%s false counts are not monotone.' % npz_path)
    if not np.all(np.diff(true_counts, axis=0) <= 0):
        raise ValueError('%s true counts are not monotone.' % npz_path)
    if not np.all(target_matrix == target_matrix[0]):
        raise ValueError('%s target denominators vary by threshold.' % npz_path)
    if not np.all(pixel_matrix == pixel_matrix[0]):
        raise ValueError('%s pixel denominators vary by threshold.' % npz_path)
    if not np.all(true_counts <= target_matrix) or not np.all(false_counts <= pixel_matrix):
        raise ValueError('%s contains impossible counts.' % npz_path)

    counts = exact.PassTwoResult(
        false_counts=false_counts,
        true_counts=true_counts,
        targets_by_sequence=target_matrix[0],
        pixels_by_sequence=pixel_matrix[0],
    )
    reference = payload.get('reference_workpoint')
    at_zero, fixed_fa, fixed_pd = exact.compute_workpoint_summary(
        counts, thresholds, reference,
    )
    _require_dict_counts_equal('logit-zero workpoint', payload.get('workpoint_at_logit_zero', {}), at_zero)
    _require_dict_counts_equal('Pd@B1-Fa workpoint', payload.get('pd_at_reference_fa', {}), fixed_fa)
    if fixed_pd.get('available'):
        _require_dict_counts_equal('Fa@B1-Pd workpoint', payload.get('fa_at_reference_pd', {}), fixed_pd)
    elif payload.get('fa_at_reference_pd', {}).get('available') is not False:
        raise ValueError('%s unreachable fixed-Pd status mismatch.' % path)
    return {'payload': payload, 'arrays': arrays, 'path': Path(path), 'npz': Path(npz_path)}


def compare_repeat_counts(primary, repeated):
    """Compare complete threshold/count tables field by field."""
    keys = (
        'thresholds', 'sequence_names',
        'false_pixels_by_threshold_sequence',
        'true_targets_by_threshold_sequence',
        'total_targets_by_threshold_sequence',
        'pixel_count_by_threshold_sequence',
    )
    differences = []
    for key in keys:
        first = primary['arrays'][key]
        second = repeated['arrays'][key]
        if first.shape != second.shape or first.dtype != second.dtype:
            differences.append('%s shape/dtype' % key)
        elif not np.array_equal(first, second):
            differences.append('%s values' % key)
    sections = (
        'workpoint_at_logit_zero', 'reference_workpoint',
        'pd_at_reference_fa', 'fa_at_reference_pd',
    )
    for section in sections:
        left = primary['payload'].get(section, {})
        right = repeated['payload'].get(section, {})
        fields = (
            'available', 'false_pixels', 'true_targets', 'total_targets', 'pixels',
            'false_pixels_by_sequence', 'true_targets_by_sequence',
        )
        for field in fields:
            if left.get(field) != right.get(field):
                differences.append('%s.%s' % (section, field))
    return not differences, differences


def load_probability_rows(
    experiment_root, log_root=None, profile=grid.MODERNIZED_PROFILE,
):
    if log_root is not None:
        rows, _loaded = grid.load_rows(experiment_root, log_root, profile)
        return rows
    metrics_root = experiment_root / 'metrics'
    loaded = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            run_id = expected_run_id(variant, seed)
            path = metrics_root / (run_id + '__noise8_internal_val.json')
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('dataset') != exact.DATASET or payload.get('model') != exact.MODEL:
                raise ValueError('%s has the wrong probability-grid identity.' % path)
            curve = common.validate_curve_counts(payload, exact.VAL_NAMES, str(path))
            common.validate_payload_metrics(
                payload, curve, str(path),
                expected_inference_amp=grid.PROFILES[profile]['inference_amp'],
                validate_pixel_diagnostics=False,
            )
            pd_curve, fa_curve = grid._curve_totals(curve)
            loaded[(variant, seed)] = {
                'payload': payload, 'pd': pd_curve, 'fa': fa_curve,
            }
    half = int(np.flatnonzero(common.EXPECTED_THRESHOLD_GRID == 0.5)[0])
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            entry = loaded[(variant, seed)]
            baseline = loaded[('none', seed)]
            row = {
                'run_id': expected_run_id(variant, seed),
                'variant': variant,
                'seed': seed,
                'pd_at_fixed_fa': common.max_pd_at_fa(
                    entry['pd'], entry['fa'], float(baseline['fa'][half]),
                ),
                'fa_at_fixed_pd': common.min_fa_at_pd(
                    entry['pd'], entry['fa'], float(baseline['pd'][half]),
                ),
                'elapsed_seconds': float(entry['payload']['elapsed_seconds']),
            }
            rows.append(row)
    index = {(row['variant'], row['seed']): row for row in rows}
    for row in rows:
        baseline = index[('none', row['seed'])]
        row['delta_pd_at_fixed_fa'] = row['pd_at_fixed_fa'] - baseline['pd_at_fixed_fa']
        row['delta_fa_at_fixed_pd'] = row['fa_at_fixed_pd'] - baseline['fa_at_fixed_pd']
        row['fa_at_fixed_pd_relative_reduction'] = (
            (baseline['fa_at_fixed_pd'] - row['fa_at_fixed_pd'])
            / baseline['fa_at_fixed_pd']
            if baseline['fa_at_fixed_pd'] > 0 else float('nan')
        )
        row['latency_ratio'] = (
            row['elapsed_seconds'] / baseline['elapsed_seconds']
            if baseline['elapsed_seconds'] > 0 else float('nan')
        )
    return rows


def make_exact_rows(primary, probability_rows):
    latency = {
        (row['variant'], row['seed']): row['latency_ratio']
        for row in probability_rows
    }
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            payload = primary[(variant, seed)]['payload']
            fixed_fa = payload['pd_at_reference_fa']
            fixed_pd = payload['fa_at_reference_pd']
            rows.append({
                'run_id': expected_run_id(variant, seed),
                'variant': variant,
                'variant_label': VARIANTS[variant][1],
                'seed': seed,
                'pd_at_fixed_fa': float(fixed_fa['pd']),
                'fa_at_fixed_pd': (
                    float(fixed_pd['fa']) if fixed_pd.get('available') else float('nan')
                ),
                'latency_ratio': float(latency[(variant, seed)]),
            })
    index = {(row['variant'], row['seed']): row for row in rows}
    for row in rows:
        baseline = index[('none', row['seed'])]
        row['delta_pd_at_fixed_fa'] = row['pd_at_fixed_fa'] - baseline['pd_at_fixed_fa']
        row['delta_fa_at_fixed_pd'] = row['fa_at_fixed_pd'] - baseline['fa_at_fixed_pd']
        row['fa_at_fixed_pd_relative_reduction'] = (
            (baseline['fa_at_fixed_pd'] - row['fa_at_fixed_pd'])
            / baseline['fa_at_fixed_pd']
            if baseline['fa_at_fixed_pd'] > 0
            and math.isfinite(row['fa_at_fixed_pd']) else float('nan')
        )
    return rows


def c2_gate(rows, reproducible=True):
    index = {(row['variant'], row['seed']): row for row in rows}
    c2 = [index[('center_ring', seed)] for seed in SEEDS]
    c1 = [index[('center_multiscale', seed)] for seed in SEEDS]
    finite = all(
        math.isfinite(row[field]) for row in c2
        for field in (
            'fa_at_fixed_pd_relative_reduction',
            'delta_pd_at_fixed_fa', 'latency_ratio',
        )
    )
    if not finite:
        return 'FAIL', ['C2 gate values are all finite: FAIL']
    mean_fa_reduction = float(np.mean([
        row['fa_at_fixed_pd_relative_reduction'] for row in c2
    ]))
    mean_pd_delta = float(np.mean([row['delta_pd_at_fixed_fa'] for row in c2]))
    pareto = all(
        c2_row['fa_at_fixed_pd'] <= c1_row['fa_at_fixed_pd']
        and c2_row['pd_at_fixed_fa'] >= c1_row['pd_at_fixed_fa']
        and (
            c2_row['fa_at_fixed_pd'] < c1_row['fa_at_fixed_pd']
            or c2_row['pd_at_fixed_fa'] > c1_row['pd_at_fixed_fa']
        )
        for c2_row, c1_row in zip(c2, c1)
    )
    checks = [
        ('C2 mean Fa@B1-Pd relative reduction >=20%', mean_fa_reduction >= 0.20),
        ('C2 mean Pd@B1-Fa delta >=-1pp', mean_pd_delta >= -0.01),
        ('Every seed Fa reduction >=-20%', min(
            row['fa_at_fixed_pd_relative_reduction'] for row in c2
        ) >= -0.20),
        ('Every seed Pd delta >=-3pp', min(
            row['delta_pd_at_fixed_fa'] for row in c2
        ) >= -0.03),
        ('Every seed latency ratio <=1.3', all(
            0 < row['latency_ratio'] <= 1.30 for row in c2
        )),
        ('C2 Pareto-improves C1 in every paired seed', pareto),
        ('B1/C2 exact counts repeat field-by-field', reproducible),
    ]
    return (
        'PASS' if all(passed for _, passed in checks) else 'FAIL',
        ['%s: %s' % (label, 'PASS' if passed else 'FAIL') for label, passed in checks],
    )


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['variant']].append(row)
    result = []
    for variant in VARIANTS:
        group = grouped[variant]
        item = {'variant': variant, 'variant_label': VARIANTS[variant][1], 'n': len(group)}
        for field in (
            'pd_at_fixed_fa', 'fa_at_fixed_pd',
            'delta_pd_at_fixed_fa', 'fa_at_fixed_pd_relative_reduction',
            'latency_ratio',
        ):
            values = [row[field] for row in group]
            item[field + '_mean'] = common.mean(values)
            item[field + '_sd'] = common.sample_sd(values)
        result.append(item)
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summaries, exact_status, exact_details, probability_status,
                 probability_details, repeat_details,
                 profile=grid.MODERNIZED_PROFILE):
    exact_pass = exact_status == 'PASS'
    probability_pass = probability_status == 'PROVISIONAL_PASS'
    agreement = exact_pass == probability_pass
    final_status = 'CONFIRMED_PASS' if exact_pass and probability_pass else 'STOP'
    lines = [
        '# Noise8 BC-TPro raw-logit 精确工作点分析', '',
        '协议身份：`%s`；推理精度：`%s`。' % (
            profile,
            'FP32' if not grid.PROFILES[profile]['inference_amp'] else 'AMP',
        ), '',
        '最终门槛：**%s**；raw-logit 与概率网格方向%s。'
        % (final_status, '一致' if agreement else '不一致'), '',
        '这里的 `Pd@B1-Fa` 使用同 seed B1 在 raw logit=0 的整数虚警预算；'
        '`Fa@B1-Pd` 使用同一 B1 工作点的整数命中数。阈值发现覆盖全部目标峰值，'
        '计数来自同一批临时 float32 推理数组，因此不受 sigmoid=1 饱和影响。', '',
        '## 三随机种子 mean ± sample-SD', '',
        '| Variant | Pd@B1-Fa (%) | Fa@B1-Pd (×1e-5) | ΔPd (pp) | Fa relative reduction | Latency ratio |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for item in summaries:
        lines.append(
            '| {label} | {pd:.3f} ± {pd_sd:.3f} | {fa:.3f} ± {fa_sd:.3f} | '
            '{dpd:.3f} ± {dpd_sd:.3f} | {red:.2f}% ± {red_sd:.2f}% | '
            '{lat:.3f} ± {lat_sd:.3f} |'.format(
                label=item['variant_label'],
                pd=item['pd_at_fixed_fa_mean'] * 100,
                pd_sd=item['pd_at_fixed_fa_sd'] * 100,
                fa=item['fa_at_fixed_pd_mean'] * 1e5,
                fa_sd=item['fa_at_fixed_pd_sd'] * 1e5,
                dpd=item['delta_pd_at_fixed_fa_mean'] * 100,
                dpd_sd=item['delta_pd_at_fixed_fa_sd'] * 100,
                red=item['fa_at_fixed_pd_relative_reduction_mean'] * 100,
                red_sd=item['fa_at_fixed_pd_relative_reduction_sd'] * 100,
                lat=item['latency_ratio_mean'], lat_sd=item['latency_ratio_sd'],
            )
        )
    lines.extend(['', '## Raw-logit C2 gate', ''])
    lines.extend('- ' + detail for detail in exact_details)
    lines.extend(['', '## 概率网格对照', ''])
    lines.append('- 概率网格状态：**%s**。' % probability_status)
    lines.extend('- ' + detail for detail in probability_details)
    lines.extend(['', '## 重复计数检查', ''])
    lines.extend('- ' + detail for detail in repeat_details)
    lines.extend([
        '', '## 解释边界', '',
        '- 仅使用 Noise8 官方训练集合中的固定 64/16 internal split；官方 test 未访问。',
        '- 12 个 epoch-32 checkpoint 均为随机初始化训练；无预训练权重。',
        '- raw-logit 评测不替代论文 27 阈值 AUC，而是修正固定工作点的 sigmoid 饱和歧义。',
        '- 三个训练 seed 只支持稳定性筛查；若两种阈值域结论不一致，停止 C3，而不是选择更有利的结果。',
        '- 本流程只使用路径、清单、模型元数据和原始整数计数做语义核对。',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return final_status, agreement


def main(argv=None):
    args = parse_args(argv)
    experiment_root = args.experiment_root.expanduser().resolve()
    output_root = experiment_root / 'exact_logit_metrics'
    primary = {}
    checkpoint_cache = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            stem = exact_stem(variant, seed)
            primary[(variant, seed)] = load_exact_output(
                output_root / (stem + '.json'), output_root / (stem + '.npz'),
                variant, seed, 0, args.log_root, checkpoint_cache, args.profile,
            )
    for variant in VARIANTS:
        for seed in SEEDS:
            payload = primary[(variant, seed)]['payload']
            if variant == 'none':
                if payload.get('reference') is not None:
                    raise ValueError('Primary B1 must define, not consume, the reference.')
            else:
                reference = payload.get('reference') or {}
                expected_path = primary[('none', seed)]['path'].resolve()
                if Path(reference.get('path', '')).expanduser().resolve() != expected_path:
                    raise ValueError('%s uses the wrong same-seed B1 reference.' % payload['run_identity']['run_id'])
                b1_zero = primary[('none', seed)]['payload']['workpoint_at_logit_zero']
                expected_budget = {
                    field: b1_zero[field]
                    for field in ('false_pixels', 'true_targets', 'total_targets', 'pixels')
                }
                if reference.get('workpoint') != expected_budget:
                    raise ValueError('%s embeds the wrong B1 integer budget.' % payload['run_identity']['run_id'])
                if payload.get('reference_workpoint') != expected_budget:
                    raise ValueError('%s evaluates against the wrong B1 budget.' % payload['run_identity']['run_id'])

    repeat_details = []
    all_reproducible = True
    for variant in ('none', 'center_ring'):
        for seed in SEEDS:
            stem = exact_stem(variant, seed, repeat_index=1)
            repeated = load_exact_output(
                output_root / (stem + '.json'), output_root / (stem + '.npz'),
                variant, seed, 1, args.log_root, checkpoint_cache, args.profile,
            )
            repeat_reference = repeated['payload'].get('reference')
            if variant == 'none':
                if repeat_reference is not None:
                    raise ValueError('Repeated B1 must define, not consume, the reference.')
            else:
                expected_reference = primary[('none', seed)]['path'].resolve()
                if Path((repeat_reference or {}).get('path', '')).expanduser().resolve() != expected_reference:
                    raise ValueError('Repeated C2 uses the wrong same-seed B1 reference.')
            matched, differences = compare_repeat_counts(
                primary[(variant, seed)], repeated,
            )
            all_reproducible = all_reproducible and matched
            repeat_details.append(
                '%s seed%d: %s%s'
                % (
                    variant, seed, 'MATCH' if matched else 'MISMATCH',
                    '' if matched else ' (' + ', '.join(differences) + ')',
                )
            )

    probability_rows = load_probability_rows(
        experiment_root, args.log_root, args.profile,
    )
    probability_status, probability_details = grid.provisional_gate(probability_rows)
    rows = make_exact_rows(primary, probability_rows)
    exact_status, exact_details = c2_gate(rows, all_reproducible)
    summaries = summarize(rows)
    write_csv(experiment_root / 'noise8_exact_logit_results.csv', rows)
    write_csv(experiment_root / 'noise8_exact_logit_summary.csv', summaries)
    final_status, agreement = write_report(
        experiment_root / 'NOISE8_EXACT_LOGIT_ANALYSIS.md', summaries,
        exact_status, exact_details, probability_status,
        probability_details, repeat_details, args.profile,
    )
    summary_payload = {
        'profile': args.profile,
        'raw_logit_gate': exact_status,
        'probability_grid_gate': probability_status,
        'conclusions_agree': agreement,
        'final_status': final_status,
        'repeat_counts_match': all_reproducible,
        'primary_runs': 12,
        'repeat_runs': 6,
        'official_test_accessed': False,
    }
    exact.atomic_json_dump(
        summary_payload, experiment_root / 'noise8_exact_logit_gate.json',
    )
    print('Validated 12 primary and 6 repeated Noise8 exact-logit evaluations.')
    print('Raw-logit gate: %s' % exact_status)
    print('Probability-grid gate: %s' % probability_status)
    print('Final status: %s' % final_status)


if __name__ == '__main__':
    main()
