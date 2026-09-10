#!/usr/bin/env python3
"""Validate and summarize locked Noise8 official-test evaluations."""

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

from tools import analyze_bc_tpro_stage1 as common  # noqa: E402
from tools import validate_bc_tpro_noise8_final as protocol  # noqa: E402


PAPER_THRESHOLDS = common.PAPER_THRESHOLDS
EXPECTED_WINDOWS = 60
LOW_SNR_NAMES = frozenset({
    'Sequence92', 'Sequence47', 'Sequence56', 'Sequence59',
    'Sequence76', 'Sequence101', 'Sequence105', 'Sequence119',
})
HIGH_SNR_NAMES = frozenset({
    'Sequence85', 'Sequence86', 'Sequence87', 'Sequence88',
    'Sequence89', 'Sequence90', 'Sequence91', 'Sequence93',
    'Sequence94', 'Sequence95', 'Sequence96', 'Sequence97',
})


def fail(message):
    raise ValueError(message)


def _finite(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail('%s must be numeric.' % context)
    value = float(value)
    if not math.isfinite(value):
        fail('%s must be finite.' % context)
    return value


def _close(observed, expected, context, abs_tol=1e-12):
    observed = _finite(observed, context)
    if not math.isclose(
        observed, float(expected), rel_tol=1e-9, abs_tol=abs_tol,
    ):
        fail('%s mismatch: observed=%r expected=%r.'
             % (context, observed, expected))


def _count_array(value, shape, context, positive=False):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        fail('%s is not a numeric array: %s' % (context, error))
    if array.shape != shape:
        fail('%s shape mismatch: %r != %r.' % (context, array.shape, shape))
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        fail('%s must contain finite integer counts.' % context)
    if positive:
        if np.any(array <= 0):
            fail('%s must contain positive counts.' % context)
    elif np.any(array < 0):
        fail('%s must contain non-negative counts.' % context)
    return array


def validate_curve_counts(payload, expected_names, context):
    if payload.get('schema_version') != 2:
        fail('%s schema_version must be 2.' % context)
    counts = payload.get('curve_counts')
    if not isinstance(counts, dict):
        fail('%s curve_counts are missing.' % context)
    if counts.get('sequence_names') != list(expected_names):
        fail('%s sequence order differs from official test.txt.' % context)
    thresholds = np.asarray(counts.get('thresholds'), dtype=np.float64)
    if not np.array_equal(thresholds, PAPER_THRESHOLDS):
        fail('%s must contain only the paper 27-threshold grid.' % context)
    shape = (len(expected_names), len(PAPER_THRESHOLDS))
    false_counts = _count_array(
        counts.get('false_pixels_by_sequence'), shape,
        context + ' false-pixel counts',
    )
    true_counts = _count_array(
        counts.get('true_targets_by_sequence'), shape,
        context + ' true-target counts',
    )
    target_counts = _count_array(
        counts.get('total_targets_by_sequence'), shape,
        context + ' total-target counts',
    )
    pixels = _count_array(
        counts.get('pixel_count_by_sequence'), (len(expected_names),),
        context + ' pixel counts', positive=True,
    )
    if np.any(true_counts > target_counts):
        fail('%s true-target counts exceed totals.' % context)
    if not np.array_equal(
        target_counts,
        np.broadcast_to(target_counts[:, :1], target_counts.shape),
    ):
        fail('%s target totals vary by threshold.' % context)
    if np.any(np.diff(true_counts, axis=1) > 0):
        fail('%s true-target counts increase with threshold.' % context)
    if np.any(np.diff(false_counts, axis=1) > 0):
        fail('%s false-pixel counts increase with threshold.' % context)
    if np.any(false_counts > pixels[:, None]):
        fail('%s false-pixel counts exceed evaluated pixels.' % context)
    if target_counts[:, 0].sum() <= 0:
        fail('%s official test contains no target count.' % context)
    return {
        'thresholds': thresholds,
        'false': false_counts,
        'true': true_counts,
        'target': target_counts,
        'pixels': pixels,
    }


def _summary_for_indices(curve, indices):
    false_counts = curve['false'][indices].sum(axis=0)
    true_counts = curve['true'][indices].sum(axis=0)
    target_counts = curve['target'][indices].sum(axis=0)
    pixels = curve['pixels'][indices].sum()
    pd_curve = np.divide(
        true_counts, target_counts,
        out=np.zeros_like(true_counts, dtype=np.float64),
        where=target_counts != 0,
    )
    fa_curve = false_counts / pixels
    half = int(np.flatnonzero(PAPER_THRESHOLDS == 0.5)[0])
    return {
        'pd': float(pd_curve[half]),
        'pd_percent': float(pd_curve[half] * 100.0),
        'fa': float(fa_curve[half]),
        'fa_x1e5': float(fa_curve[half] * 1e5),
        'auc': float(abs(np.trapz(pd_curve, fa_curve))),
        'true_targets': int(true_counts[half]),
        'total_targets': int(target_counts[half]),
        'false_pixels': int(false_counts[half]),
        'pixel_count': int(pixels),
    }


def _validate_summary(observed, expected, context):
    if not isinstance(observed, dict):
        fail('%s summary is missing.' % context)
    for field, value in expected.items():
        _close(observed.get(field), value, context + '.' + field)


def _validate_eval_namespace(eval_log, metrics_path, job, log_root):
    records = common.parse_namespace_records(eval_log)
    matches = []
    for record in records:
        value = record['values'].get('metrics_json')
        if isinstance(value, str) and Path(value).expanduser().resolve() == metrics_path:
            matches.append(record['values'])
    if len(matches) != 1:
        fail('%s must contain exactly one matching evaluation Namespace.'
             % job['run_id'])
    namespace = matches[0]
    expected = {
        'amp': False,
        'batch_size': 1,
        'dataset': protocol.DATASET,
        'epoch': 32,
        'eval_chunk_rows': 32,
        'gpu': job['gpu'],
        'log_dir': job['log_dir'],
        'output_only': False,
        'prefetch_factor': 1,
        'seqlen': 40,
        'sequence_list': None,
        'sequence_start': 0,
        'sequence_stop': None,
        'split': 'test',
        'test_workers': 1,
        'threshold_eval': 0.5,
        'threshold_grid_step': 0.0,
    }
    for field, value in expected.items():
        if field not in namespace:
            fail('%s evaluation Namespace lacks %s.' % (job['run_id'], field))
        common._require_typed_equal(
            job['run_id'], 'evaluation Namespace ' + field,
            namespace[field], value,
        )
    common._require_path_equal(
        job['run_id'], 'evaluation Namespace datapath',
        namespace.get('datapath'), protocol.EXPECTED_DATA_ROOT,
    )
    common._require_path_equal(
        job['run_id'], 'evaluation Namespace logpath',
        namespace.get('logpath'), log_root.parent,
    )
    marker = 'Paper-aligned metrics saved to %s.' % metrics_path
    if eval_log.read_text(encoding='utf-8').count(marker) != 1:
        fail('%s completion marker is missing or duplicated.' % job['run_id'])


def validate_one_metric(payload, metrics_path, job, expected_names, log_root):
    context = job['run_id'] + ' official-test metric'
    metrics_path = Path(metrics_path).expanduser().resolve(strict=True)
    log_root = Path(log_root).expanduser().resolve(strict=True)
    if payload.get('dataset') != protocol.DATASET:
        fail('%s dataset mismatch.' % context)
    if payload.get('model') != protocol.MODEL:
        fail('%s model mismatch.' % context)
    if payload.get('checkpoint_epoch') != 32:
        fail('%s checkpoint epoch mismatch.' % context)
    if payload.get('sequence_length') != 40:
        fail('%s sequence length mismatch.' % context)
    if payload.get('sequence_count') != 20:
        fail('%s sequence count mismatch.' % context)
    if payload.get('evaluation_windows') != EXPECTED_WINDOWS:
        fail('%s evaluation-window count mismatch.' % context)
    if payload.get('inference_amp') is not False:
        fail('%s must use FP32 inference.' % context)
    if not np.array_equal(
        np.asarray(payload.get('paper_thresholds'), dtype=np.float64),
        PAPER_THRESHOLDS,
    ):
        fail('%s paper threshold metadata mismatch.' % context)
    _close(payload.get('operating_threshold'), 0.5,
           context + ' operating_threshold')

    expected_checkpoint = (
        log_root / job['log_dir'] / 'checkpoints' / 'epoch_32_model.pth'
    ).resolve()
    if Path(payload.get('checkpoint', '')).expanduser().resolve() != expected_checkpoint:
        fail('%s checkpoint path mismatch.' % context)
    training_log = (
        log_root / job['log_dir'] / 'logs' / (protocol.MODEL + '.txt')
    )
    checkpoint = protocol.validate_training_checkpoint(
        expected_checkpoint, training_log, job, log_root,
    )

    curve = validate_curve_counts(payload, expected_names, context)
    all_indices = np.arange(len(expected_names), dtype=np.int64)
    low_indices = np.asarray([
        index for index, name in enumerate(expected_names)
        if name in LOW_SNR_NAMES
    ], dtype=np.int64)
    high_indices = np.asarray([
        index for index, name in enumerate(expected_names)
        if name in HIGH_SNR_NAMES
    ], dtype=np.int64)
    if low_indices.size != 8 or high_indices.size != 12:
        fail('%s SNR group membership mismatch.' % context)
    if set(expected_names) != LOW_SNR_NAMES | HIGH_SNR_NAMES:
        fail('%s has an unknown SNR group.' % context)
    expected_all = _summary_for_indices(curve, all_indices)
    expected_low = _summary_for_indices(curve, low_indices)
    expected_high = _summary_for_indices(curve, high_indices)
    _validate_summary(payload.get('all'), expected_all, context + ' all')
    _validate_summary(payload.get('low_snr'), expected_low, context + ' low_snr')
    _validate_summary(payload.get('high_snr'), expected_high, context + ' high_snr')

    for field in (
        'elapsed_seconds', 'cuda_peak_allocated_gib',
        'cuda_peak_reserved_gib', 'parameters_m',
    ):
        if _finite(payload.get(field), context + ' ' + field) <= 0:
            fail('%s %s must be positive.' % (context, field))
    if payload['cuda_peak_reserved_gib'] < payload['cuda_peak_allocated_gib']:
        fail('%s reserved CUDA peak is below allocated peak.' % context)

    model_module = __import__(
        'networks.models.DeepPro-Plus_BCTPro', fromlist=['detector'],
    )
    reference = model_module.detector(
        1, 40, 40,
        structure_variant=job['structure_variant'],
        structure_bottleneck_channels=8,
        eval_chunk_rows=32,
    )
    expected_parameters = sum(
        parameter.numel() for parameter in reference.parameters()
    ) / 1e6
    _close(payload['parameters_m'], expected_parameters,
           context + ' parameter count')
    reference.load_state_dict(checkpoint['model_state_dict'], strict=True)

    eval_log = log_root / job['log_dir'] / 'eval_epoch-32.txt'
    _validate_eval_namespace(eval_log, metrics_path, job, log_root)
    return {
        'run_id': job['run_id'],
        'role': job['role'],
        'code': job['code'],
        'variant': job['structure_variant'],
        'seed': job['seed'],
        'gpu': job['gpu'],
        'pd_percent': expected_all['pd_percent'],
        'fa_x1e5': expected_all['fa_x1e5'],
        'auc': expected_all['auc'],
        'low_snr_pd_percent': expected_low['pd_percent'],
        'low_snr_fa_x1e5': expected_low['fa_x1e5'],
        'low_snr_auc': expected_low['auc'],
        'high_snr_pd_percent': expected_high['pd_percent'],
        'high_snr_fa_x1e5': expected_high['fa_x1e5'],
        'high_snr_auc': expected_high['auc'],
        'elapsed_seconds': float(payload['elapsed_seconds']),
        'cuda_peak_allocated_gib': float(payload['cuda_peak_allocated_gib']),
        'cuda_peak_reserved_gib': float(payload['cuda_peak_reserved_gib']),
        'parameters_m': float(payload['parameters_m']),
    }


def load_context(args):
    protocol.require_profile(args.profile)
    final_root = Path(args.final_root).expanduser().resolve()
    expected_root = protocol.EXPECTED_FINAL_ROOT
    if final_root != expected_root:
        fail('Final experiment root mismatch: %s.' % final_root)
    _payload, jobs = protocol.load_and_validate_frozen_plan(
        args.lock, args.stage1_root, args.log_root, args.data_root,
        final_root, args.profile,
    )
    _train_names, test_names = protocol.validate_official_split_metadata(
        args.data_root,
    )
    return jobs, test_names, final_root


def load_one(args, jobs, test_names, final_root):
    by_id = {job['run_id']: job for job in jobs}
    if args.run_id not in by_id:
        fail('Run is not present in the locked final plan: %s.' % args.run_id)
    job = by_id[args.run_id]
    metrics_path = (
        final_root / 'official_test_metrics'
        / (job['run_id'] + '__official_test.json')
    ).resolve()
    payload = json.loads(metrics_path.read_text(encoding='utf-8'))
    return validate_one_metric(
        payload, metrics_path, job, test_names,
        Path(args.log_root).expanduser().resolve(),
    )


def _mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size != 3 or not np.isfinite(array).all():
        fail('Every final summary requires three finite seed values.')
    return float(array.mean()), float(array.std(ddof=1))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['code'], row['variant'])].append(row)
    summaries = []
    numeric = (
        'pd_percent', 'fa_x1e5', 'auc',
        'low_snr_pd_percent', 'low_snr_fa_x1e5', 'low_snr_auc',
        'high_snr_pd_percent', 'high_snr_fa_x1e5', 'high_snr_auc',
        'elapsed_seconds', 'cuda_peak_allocated_gib', 'cuda_peak_reserved_gib',
        'parameters_m',
    )
    for (code, variant), group in grouped.items():
        if sorted(row['seed'] for row in group) != [47, 49, 51]:
            fail('%s final results do not contain all three seeds.' % code)
        item = {'code': code, 'variant': variant, 'n': 3}
        for field in numeric:
            item[field + '_mean'], item[field + '_sd'] = _mean_sd(
                [row[field] for row in group]
            )
        summaries.append(item)
    summaries.sort(key=lambda item: item['code'] != 'B1')
    return summaries


def paired_differences(rows):
    index = {(row['variant'], row['seed']): row for row in rows}
    variants = sorted({row['variant'] for row in rows} - {'none'})
    if not variants:
        return []
    if len(variants) != 1:
        fail('Final comparison must contain B1 and at most one locked candidate.')
    candidate = variants[0]
    result = []
    for seed in (47, 49, 51):
        baseline = index[('none', seed)]
        current = index[(candidate, seed)]
        result.append({
            'variant': candidate,
            'seed': seed,
            'delta_pd_percent': current['pd_percent'] - baseline['pd_percent'],
            'delta_fa_x1e5': current['fa_x1e5'] - baseline['fa_x1e5'],
            'delta_auc': current['auc'] - baseline['auc'],
        })
    return result


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summaries, differences):
    lines = [
        '# BC-TPro Noise8 最终 80/20 论文指标', '',
        '状态：锁定结构在全部官方 train 80 序列上从随机权重训练；所有训练完成后，'
        '固定 epoch 32 对官方 test 20 序列各评测一次。', '',
        '## Overall（mean ± sample SD，n=3）', '',
        '| Model | Pd@0.5 (%) ↑ | Fa@0.5 (×1e-5) ↓ | 27-threshold AUC ↑ | Time (s) | Params (M) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in summaries:
        lines.append(
            '| {code} `{variant}` | {pd:.3f} ± {pd_sd:.3f} | '
            '{fa:.4f} ± {fa_sd:.4f} | {auc:.6f} ± {auc_sd:.6f} | '
            '{time:.2f} ± {time_sd:.2f} | '
            '{params:.6f} ± {params_sd:.6f} |'.format(
                code=row['code'], variant=row['variant'],
                pd=row['pd_percent_mean'], pd_sd=row['pd_percent_sd'],
                fa=row['fa_x1e5_mean'], fa_sd=row['fa_x1e5_sd'],
                auc=row['auc_mean'], auc_sd=row['auc_sd'],
                time=row['elapsed_seconds_mean'],
                time_sd=row['elapsed_seconds_sd'],
                params=row['parameters_m_mean'],
                params_sd=row['parameters_m_sd'],
            )
        )
    lines.extend([
        '', '## Low-/High-SNR 分组', '',
        '| Model | Low Pd (%) | Low Fa (×1e-5) | Low AUC | High Pd (%) | High Fa (×1e-5) | High AUC |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in summaries:
        lines.append(
            '| {code} | {lpd:.3f} ± {lpd_sd:.3f} | {lfa:.4f} ± {lfa_sd:.4f} | '
            '{lauc:.6f} ± {lauc_sd:.6f} | {hpd:.3f} ± {hpd_sd:.3f} | '
            '{hfa:.4f} ± {hfa_sd:.4f} | {hauc:.6f} ± {hauc_sd:.6f} |'.format(
                code=row['code'],
                lpd=row['low_snr_pd_percent_mean'],
                lpd_sd=row['low_snr_pd_percent_sd'],
                lfa=row['low_snr_fa_x1e5_mean'],
                lfa_sd=row['low_snr_fa_x1e5_sd'],
                lauc=row['low_snr_auc_mean'],
                lauc_sd=row['low_snr_auc_sd'],
                hpd=row['high_snr_pd_percent_mean'],
                hpd_sd=row['high_snr_pd_percent_sd'],
                hfa=row['high_snr_fa_x1e5_mean'],
                hfa_sd=row['high_snr_fa_x1e5_sd'],
                hauc=row['high_snr_auc_mean'],
                hauc_sd=row['high_snr_auc_sd'],
            )
        )
    if differences:
        fields = ('delta_pd_percent', 'delta_fa_x1e5', 'delta_auc')
        values = {field: _mean_sd([row[field] for row in differences])
                  for field in fields}
        lines.extend([
            '', '## 锁定候选相对 B1 的同 seed 配对差', '',
            '- ΔPd：%.3f ± %.3f 个百分点。' % values['delta_pd_percent'],
            '- ΔFa：%.4f ± %.4f ×1e-5。' % values['delta_fa_x1e5'],
            '- ΔAUC：%.6f ± %.6f。' % values['delta_auc'],
        ])
    lines.extend([
        '', '## 解释边界', '',
        '- 论文 DeepPro-Plus HiNo 参考值为 Pd 76.23%、Fa 1.69×1e-5、AUC 0.9171；本地结果可并列表述，但环境与本地数据副本仍需披露。',
        '- 检测结果只报告 Pd@0.5、Fa@0.5 与官方 27 阈值 Pd-Fa AUC。',
        '- Noise8 图像与 mask 来自本地 Noise8 数据；Pd/Fa 所需质心标签由相邻 Clean 同名标注回退，需在方法中披露。',
        '- 三个训练 seed 描述初步训练波动；不能把小差异或区间重叠写成等效性证明。',
        '- 训练与评测身份由规范化路径、精确清单、参数、checkpoint 元数据和逐序列整数计数核对。',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('one', 'all'):
        command = subparsers.add_parser(name)
        command.add_argument('--data-root', type=Path, required=True)
        command.add_argument('--stage1-root', type=Path, required=True)
        command.add_argument('--log-root', type=Path, required=True)
        command.add_argument('--lock', type=Path, required=True)
        command.add_argument(
            '--profile', choices=(protocol.PROFILE,), required=True,
        )
        command.add_argument('--final-root', type=Path, required=True)
        if name == 'one':
            command.add_argument('--run-id', required=True)
        else:
            command.add_argument('--queue-root', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    jobs, test_names, final_root = load_context(args)
    if args.command == 'one':
        row = load_one(args, jobs, test_names, final_root)
        print(
            'VALID official-test metric: %s Pd=%.5f Fa=%.8e AUC=%.8f'
            % (row['run_id'], row['pd_percent'], row['fa_x1e5'] * 1e-5,
               row['auc'])
        )
        return

    queue_root = Path(args.queue_root).expanduser().resolve(strict=True)
    expected_queue = (
        Path(args.log_root).expanduser().resolve()
        / '_queues' / 'bc_tpro_final_noise8_2026-09-09'
    ).resolve()
    if queue_root != expected_queue:
        fail('Final queue root mismatch.')
    rows = []
    for job in jobs:
        receipt = queue_root / 'status' / (job['run_id'] + '.test.done')
        if not receipt.is_file():
            fail('Missing completed one-pass official-test receipt: %s.'
                 % job['run_id'])
        one_args = argparse.Namespace(**vars(args))
        one_args.run_id = job['run_id']
        rows.append(load_one(one_args, jobs, test_names, final_root))
    summaries = summarize(rows)
    differences = paired_differences(rows)
    _write_csv(final_root / 'final80_results.csv', rows)
    _write_csv(final_root / 'final80_summary.csv', summaries)
    _write_csv(final_root / 'final80_paired_differences.csv', differences)
    write_report(final_root / 'FINAL80_PAPER_RESULTS.md', summaries, differences)
    print('Validated %d one-pass official-test results.' % len(rows))
    print('Wrote %s' % (final_root / 'FINAL80_PAPER_RESULTS.md'))


if __name__ == '__main__':
    try:
        main()
    except (
        FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError,
    ) as error:
        raise SystemExit('INVALID final Noise8 result: %s' % error)
