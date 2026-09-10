#!/usr/bin/env python3
"""Locked paper-level analysis for dedicated Noise8 BC-TPro stage 1.

The command fails before opening any metric payload unless all 12 official
probability-metric payloads and their training/checkpoint provenance exist.
Candidate qualification and ranking use only threshold-0.5 Pd/Fa and the
official 27-threshold Pd-Fa AUC. Training curves are optimization diagnostics.
"""

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_bc_tpro_noise8_stage1 as grid_analysis  # noqa: E402


VARIANTS = tuple(grid_analysis.VARIANTS)
CANDIDATES = tuple(variant for variant in VARIANTS if variant != 'none')
SEEDS = grid_analysis.SEEDS
EXPECTED_EPOCHS = tuple(range(1, 33))
CURVE_PATTERNS = {
    'train_loss': re.compile(r'Training mean loss:\s*([^\s]+)'),
    'train_iou': re.compile(
        r'Training accuracy \(IoU\) of prediction:\s*([^\s]+)'
    ),
}
EPOCH_PATTERN = re.compile(r'\*{4}\s*Epoch\s+(\d+)/32\s*\*{4}')


class IncompleteEvidenceError(RuntimeError):
    """Raised before result payloads are opened when the lock is incomplete."""


class C3RequiredError(RuntimeError):
    """Raised when the preregistered C2 gate requires a C3 experiment."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--experiment-root', type=Path,
        default=(
            REPO_ROOT / 'experiments'
            / 'bc_tpro_stage1_noise8_2026-09-09'
        ),
    )
    parser.add_argument(
        '--log-root', type=Path,
        default=REPO_ROOT / 'log' / 'sem_seg',
    )
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument(
        '--profile', choices=tuple(grid_analysis.PROFILES),
        default=grid_analysis.MODERNIZED_PROFILE,
    )
    return parser.parse_args(argv)


def required_artifacts(
    experiment_root, log_root, profile=grid_analysis.MODERNIZED_PROFILE,
):
    experiment_root = Path(experiment_root).resolve()
    log_root = Path(log_root).resolve()
    metrics_root = experiment_root / 'metrics'
    artifacts = []
    for job in grid_analysis.expected_manifest_rows(profile):
        run_id = job['run_id']
        artifacts.extend([
            (
                'probability metric ' + run_id,
                metrics_root / (run_id + grid_analysis.METRICS_SUFFIX),
            ),
            (
                'training log ' + run_id,
                log_root / job['log_dir'] / 'logs'
                / (grid_analysis.MODEL + '.txt'),
            ),
            (
                'epoch-32 checkpoint ' + run_id,
                log_root / job['log_dir'] / 'checkpoints'
                / 'epoch_32_model.pth',
            ),
        ])
    return artifacts


def require_complete_artifacts(
    experiment_root, log_root, profile=grid_analysis.MODERNIZED_PROFILE,
):
    artifacts = required_artifacts(experiment_root, log_root, profile)
    missing = [(label, path) for label, path in artifacts if not path.is_file()]
    if missing:
        preview = '; '.join(
            '%s=%s' % (label, path) for label, path in missing[:8]
        )
        if len(missing) > 8:
            preview += '; ... and %d more' % (len(missing) - 8)
        raise IncompleteEvidenceError(
            'LOCKED_INCOMPLETE: refusing to read partial candidate results; '
            'missing %d required artifacts: %s' % (len(missing), preview)
        )
    return artifacts


def load_complete_evidence(
    experiment_root, log_root, profile=grid_analysis.MODERNIZED_PROFILE,
):
    """Open evidence only after an all-artifact existence barrier succeeds."""
    experiment_root = Path(experiment_root).resolve()
    log_root = Path(log_root).resolve()
    require_complete_artifacts(experiment_root, log_root, profile)

    probability_rows, probability_loaded = grid_analysis.load_rows(
        experiment_root, log_root, profile,
    )
    return probability_rows, probability_loaded


def require_c3_gate_resolved(probability_rows):
    """Refuse a C0-C2-only lock when official Pd/Fa/AUC authorize C3."""
    probability_status, _probability_details = (
        grid_analysis.provisional_gate(probability_rows)
    )
    if probability_status == 'PROVISIONAL_PASS':
        raise C3RequiredError(
            'C3_REQUIRED: the official threshold-0.5 Pd/Fa and 27-threshold '
            'Pd-Fa AUC gate passed. Train and evaluate preregistered C3 before '
            'creating a final candidate lock.'
        )
    return probability_status


def _indexed(rows, label):
    index = {}
    for row in rows:
        key = (row.get('variant'), row.get('seed'))
        if key in index:
            raise ValueError('%s has duplicate key %r.' % (label, key))
        index[key] = row
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if set(index) != expected:
        missing = sorted(expected - set(index))
        unexpected = sorted(set(index) - expected)
        raise ValueError(
            '%s must contain all 12 variant/seed rows; missing=%r unexpected=%r.'
            % (label, missing, unexpected)
        )
    return index


def evaluate_candidate_eligibility(
    probability_rows, parameter_by_run, identity_verified,
    profile=grid_analysis.MODERNIZED_PROFILE,
):
    """Apply the amended official Pd/Fa/AUC-only qualification conditions."""
    probability_index = _indexed(probability_rows, 'probability rows')
    expected_runs = {
        grid_analysis.expected_manifest_rows(profile)[index]['run_id']
        for index in range(12)
    }
    if set(parameter_by_run) != expected_runs:
        raise ValueError('Parameter table must cover the exact 12 run identities.')

    decisions = []
    for variant in CANDIDATES:
        probability_group = [
            probability_index[(variant, seed)] for seed in SEEDS
        ]
        parameters = [
            float(parameter_by_run[row['run_id']]) for row in probability_group
        ]
        if not all(math.isfinite(value) and value > 0 for value in parameters):
            raise ValueError('%s parameters must be finite and positive.' % variant)
        if len(set(parameters)) != 1:
            raise ValueError('%s parameter count differs across seeds.' % variant)

        mean_delta_pd = float(np.mean([
            row['delta_pd_at_0_5'] for row in probability_group
        ]))
        mean_fa_reduction = float(np.mean([
            row['fa_at_0_5_relative_reduction'] for row in probability_group
        ]))
        pareto_seed_count = sum(
            row['delta_pd_at_0_5'] >= 0
            and row['delta_fa_at_0_5'] <= 0
            for row in probability_group
        )
        mean_pd = float(np.mean([
            row['pd_at_0_5'] for row in probability_group
        ]))
        mean_fa = float(np.mean([
            row['fa_at_0_5'] for row in probability_group
        ]))
        mean_auc = float(np.mean([row['auc'] for row in probability_group]))
        mean_delta_auc = float(np.mean([
            row['delta_auc'] for row in probability_group
        ]))
        latency_values = [row['latency_ratio'] for row in probability_group]
        finite_primary = all(math.isfinite(value) for value in (
            mean_pd, mean_fa, mean_delta_pd, mean_fa_reduction,
            mean_auc, mean_delta_auc,
            *latency_values,
        ))
        checks = {
            'complete_finite_metrics': finite_primary,
            'mean_delta_pd_at_0_5_ge_minus_1pp': (
                finite_primary and mean_delta_pd >= -0.01
            ),
            'mean_fa_at_0_5_relative_reduction_gt_zero': (
                finite_primary and mean_fa_reduction > 0.0
            ),
            'at_least_two_joint_nonworse_seeds': pareto_seed_count >= 2,
            'mean_paper_auc_not_below_b1': (
                finite_primary and mean_delta_auc >= 0.0
            ),
            'every_seed_latency_le_1p3': (
                finite_primary
                and all(0.0 < value <= 1.30 for value in latency_values)
            ),
            'identity_verified': bool(identity_verified),
        }
        decisions.append({
            'variant': variant,
            'variant_label': grid_analysis.VARIANTS[variant][3],
            'qualified': all(checks.values()),
            'mean_pd_at_0_5': mean_pd,
            'mean_fa_at_0_5': mean_fa,
            'mean_paper_auc': mean_auc,
            'mean_delta_paper_auc': mean_delta_auc,
            'mean_fa_at_0_5_relative_reduction': mean_fa_reduction,
            'mean_delta_pd_at_0_5': mean_delta_pd,
            'joint_nonworse_seed_count': pareto_seed_count,
            'max_latency_ratio': max(latency_values),
            'parameters_m': parameters[0],
            **checks,
        })
    return decisions


def rank_qualified_candidates(decisions):
    by_variant = {item.get('variant'): item for item in decisions}
    if set(by_variant) != set(CANDIDATES) or len(by_variant) != len(decisions):
        raise ValueError('Decisions must contain exactly C0/C1/C2 once each.')
    eligible = [by_variant[variant] for variant in CANDIDATES
                if by_variant[variant]['qualified']]
    if not eligible:
        return {
            'status': 'B1_FALLBACK',
            'selected_variant': 'none',
            'qualified_ranking': [],
            'nondominated_candidates': [],
            'reason': 'No candidate passed all frozen qualification checks.',
        }

    def dominates(left, right):
        gains = (
            left['mean_pd_at_0_5'] - right['mean_pd_at_0_5'],
            right['mean_fa_at_0_5'] - left['mean_fa_at_0_5'],
            left['mean_paper_auc'] - right['mean_paper_auc'],
        )
        return all(value >= 0.0 for value in gains) and any(
            value > 0.0 for value in gains
        )

    nondominated = [
        item for item in eligible
        if not any(dominates(other, item) for other in eligible)
    ]
    remaining = [item for item in eligible if item not in nondominated]
    compatible_order = nondominated + remaining
    if len(nondominated) != 1:
        return {
            'status': 'UNRESOLVED_TRADEOFF',
            'selected_variant': None,
            # Kept for schema compatibility; this is registered order, not a
            # scalar metric ranking.
            'qualified_ranking': [item['variant'] for item in compatible_order],
            'nondominated_candidates': [
                item['variant'] for item in nondominated
            ],
            'reason': (
                'Multiple qualified candidates are Pareto-nondominated across '
                'mean Pd@0.5, mean Fa@0.5, and mean AUC27. No scalar weight or '
                'single-metric tie-break has been registered.'
            ),
        }
    return {
        'status': 'CANDIDATE_LOCKED',
        'selected_variant': nondominated[0]['variant'],
        'qualified_ranking': [item['variant'] for item in compatible_order],
        'nondominated_candidates': [nondominated[0]['variant']],
        'reason': (
            'The selected candidate is the only qualified Pareto-nondominated '
            'model across mean Pd@0.5, mean Fa@0.5, and mean AUC27.'
        ),
    }


def parse_training_curve(log_path, job):
    text = Path(log_path).read_text(encoding='utf-8')
    lines = text.splitlines()
    headers = []
    for index, line in enumerate(lines):
        match = EPOCH_PATTERN.search(line)
        if match:
            headers.append((index, int(match.group(1))))
    epochs = [epoch for _index, epoch in headers]
    if epochs != list(EXPECTED_EPOCHS):
        raise ValueError(
            '%s must contain exactly one ordered Epoch 1..32 block.' % log_path
        )
    rows = []
    for header_index, (start, epoch) in enumerate(headers):
        stop = headers[header_index + 1][0] if header_index + 1 < len(headers) else len(lines)
        block = '\n'.join(lines[start:stop])
        values = {}
        for field, pattern in CURVE_PATTERNS.items():
            matches = pattern.findall(block)
            if len(matches) != 1:
                raise ValueError(
                    '%s epoch %d must contain exactly one %s value.'
                    % (log_path, epoch, field)
                )
            try:
                value = float(matches[0])
            except ValueError as error:
                raise ValueError(
                    '%s epoch %d has invalid %s.' % (log_path, epoch, field)
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    '%s epoch %d has non-finite %s.'
                    % (log_path, epoch, field)
                )
            values[field] = value
        if values['train_loss'] < 0:
            raise ValueError('%s epoch %d has negative loss.' % (log_path, epoch))
        for field in ('train_iou',):
            if not 0.0 <= values[field] <= 1.0:
                raise ValueError(
                    '%s epoch %d %s lies outside [0,1].'
                    % (log_path, epoch, field)
                )
        rows.append({
            'run_id': job['run_id'],
            'variant': job['structure_variant'],
            'variant_label': grid_analysis.VARIANTS[
                job['structure_variant']
            ][3],
            'seed': int(job['seed']),
            'epoch': epoch,
            **values,
        })
    return rows


def load_training_curves(
    log_root, profile=grid_analysis.MODERNIZED_PROFILE,
):
    rows = []
    for job in grid_analysis.expected_manifest_rows(profile):
        log_path = (
            Path(log_root).resolve() / job['log_dir'] / 'logs'
            / (grid_analysis.MODEL + '.txt')
        )
        rows.extend(parse_training_curve(log_path, job))
    if len(rows) != 12 * 32:
        raise ValueError('Training curve table must contain exactly 384 rows.')
    return rows


def summarize_training_curves(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['variant'], row['epoch'])].append(row)
    summaries = []
    for variant in VARIANTS:
        for epoch in EXPECTED_EPOCHS:
            group = groups.get((variant, epoch), [])
            if sorted(row['seed'] for row in group) != list(SEEDS):
                raise ValueError(
                    '%s epoch %d must contain exactly seeds 47/49/51.'
                    % (variant, epoch)
                )
            item = {
                'variant': variant,
                'variant_label': grid_analysis.VARIANTS[variant][3],
                'epoch': epoch,
                'n_seeds': 3,
            }
            for field in CURVE_PATTERNS:
                values = np.asarray([row[field] for row in group], dtype=np.float64)
                item[field + '_mean'] = float(values.mean())
                item[field + '_sd'] = float(values.std(ddof=1))
            summaries.append(item)
    return summaries


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty table: %s' % path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=str(path.parent),
            prefix='.' + path.name + '.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_locked_candidate_payload(
    selection, decisions, experiment_root, log_root,
    profile=grid_analysis.MODERNIZED_PROFILE,
):
    if selection['status'] not in ('CANDIDATE_LOCKED', 'B1_FALLBACK'):
        raise ValueError(
            'A lock artifact requires a completed candidate or B1 fallback.'
        )
    selected_variant = selection['selected_variant']
    if selected_variant not in VARIANTS:
        raise ValueError('Locked variant is outside the registered variants.')
    recomputed = rank_qualified_candidates(decisions)
    for field in ('status', 'selected_variant', 'qualified_ranking'):
        if selection.get(field) != recomputed.get(field):
            raise ValueError(
                'Lock selection is inconsistent with frozen ranking: %s.'
                % field
            )
    if not all(
        item.get('identity_verified') is True
        and item.get('complete_finite_metrics') is True
        for item in decisions
    ):
        raise ValueError(
            'A lock artifact requires complete finite evidence and verified '
            'identity for every candidate.'
        )
    experiment_root = Path(experiment_root).resolve()
    log_root = Path(log_root).resolve()
    protocol_filename = (
        'OFFICIAL_METRIC_AMENDMENT_2026-09-10.md'
        if profile == grid_analysis.UPSTREAM_PROFILE
        else 'FINAL_EVALUATION_PLAN_2026-09-09.md'
    )
    ranking = list(selection['qualified_ranking'])
    rank_by_variant = {
        variant: rank for rank, variant in enumerate(ranking, start=1)
    }
    eligible = []
    for decision in decisions:
        if decision['qualified']:
            eligible.append({
                'rank': rank_by_variant[decision['variant']],
                'variant': decision['variant'],
                'mean_pd_at_0_5': decision['mean_pd_at_0_5'],
                'mean_fa_at_0_5': decision['mean_fa_at_0_5'],
                'mean_paper_auc': decision['mean_paper_auc'],
                'mean_fa_at_0_5_relative_reduction': decision[
                    'mean_fa_at_0_5_relative_reduction'
                ],
                'mean_delta_pd_at_0_5': decision[
                    'mean_delta_pd_at_0_5'
                ],
                'joint_nonworse_seed_count': decision[
                    'joint_nonworse_seed_count'
                ],
                'max_latency_ratio': decision['max_latency_ratio'],
                'parameters_m': decision['parameters_m'],
            })
    eligible.sort(key=lambda item: item['rank'])
    return {
        'schema_version': 2,
        'status': selection['status'],
        'selection_complete': True,
        'protocol_path': str(
            (
                experiment_root
                / protocol_filename
            ).resolve()
        ),
        'locked_model': {
            'role': 'B1' if selected_variant == 'none' else 'candidate',
            'model': grid_analysis.MODEL,
            'selected_variant': selected_variant,
        },
        'model': grid_analysis.MODEL,
        'selected_variant': selected_variant,
        'baseline': {
            'role': 'B1',
            'model': grid_analysis.MODEL,
            'selected_variant': 'none',
        },
        'seeds': list(SEEDS),
        'eligible_candidates': eligible,
        'candidate_qualification': decisions,
        'detection_metric_contract': grid_analysis.official_metric_contract(),
        'selection_rule': [
            'mean_paper_auc_descending',
            'mean_pd_at_0_5_descending',
            'mean_fa_at_0_5_ascending',
            'parameters_m_ascending',
        ],
        'selector_inputs': [
            {'role': label, 'path': str(path.resolve())}
            for label, path in required_artifacts(
                experiment_root, log_root, profile,
            )
        ],
        'selection_uses_single_seed': False,
        'official_test_accessed': False,
    }


def plot_training_curves(summary_rows, png_path, pdf_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = {
        'train_loss': 'Training Soft-IoU loss',
        'train_iou': 'Training pixel IoU',
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharex=True)
    for variant in VARIANTS:
        subset = [row for row in summary_rows if row['variant'] == variant]
        epochs = np.asarray([row['epoch'] for row in subset])
        for axis, field in zip(axes, CURVE_PATTERNS):
            means = np.asarray([row[field + '_mean'] for row in subset])
            sd = np.asarray([row[field + '_sd'] for row in subset])
            lower = means - sd
            upper = means + sd
            if field != 'train_loss':
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)
            axis.plot(
                epochs, means,
                label=grid_analysis.VARIANTS[variant][3], linewidth=1.6,
            )
            axis.fill_between(epochs, lower, upper, alpha=0.14)
            axis.set_title(labels[field])
            axis.set_xlabel('Epoch')
            axis.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel('Mean ± sample SD (3 seeds)')
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc='upper center', ncol=4,
        bbox_to_anchor=(0.5, 1.04), frameon=False,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches='tight')
    figure.savefig(pdf_path, bbox_inches='tight')
    plt.close(figure)


def write_report(
    path, decisions, selection, curve_summaries,
    profile=grid_analysis.MODERNIZED_PROFILE,
):
    lines = [
        '# Noise8 BC-TPro 论文候选锁定与收敛分析', '',
        '协议身份：`%s`。' % profile, '',
        '状态：12/12 官方概率指标均完整并通过严格身份检查；raw-logit 结果不是候选锁定输入。',
        '',
        '## 候选资格', '',
        '| Candidate | Qualified | Mean Pd@0.5 (%) | Mean Fa@0.5 (×1e-5) | Mean ΔPd (pp) | Mean Fa reduction | Joint non-worse seeds | Mean 27-threshold AUC | Mean ΔAUC | Max latency ratio | Params (M) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in decisions:
        lines.append(
            '| {label} | {qualified} | {pd:.3f} | {fa:.4f} | '
            '{dpd:.3f} | {fred:.2f}% | {joint}/3 | '
            '{auc:.6f} | {dauc:.6f} | {lat:.3f} | {params:.6f} |'.format(
                label=row['variant_label'],
                qualified='PASS' if row['qualified'] else 'FAIL',
                pd=row['mean_pd_at_0_5'] * 100,
                fa=row['mean_fa_at_0_5'] * 1e5,
                dpd=row['mean_delta_pd_at_0_5'] * 100,
                fred=row['mean_fa_at_0_5_relative_reduction'] * 100,
                joint=row['joint_nonworse_seed_count'],
                auc=row['mean_paper_auc'],
                dauc=row['mean_delta_paper_auc'],
                lat=row['max_latency_ratio'], params=row['parameters_m'],
            )
        )
        failed = [
            key for key, value in row.items()
            if key in {
                'complete_finite_metrics',
                'mean_delta_pd_at_0_5_ge_minus_1pp',
                'mean_fa_at_0_5_relative_reduction_gt_zero',
                'at_least_two_joint_nonworse_seeds',
                'mean_paper_auc_not_below_b1',
                'every_seed_latency_le_1p3', 'identity_verified',
            } and value is False
        ]
        if failed:
            lines.append('  - Failed `%s`.' % '`, `'.join(failed))
    lines.extend([
        '', '## 锁定结果', '',
        '- 状态：`%s`' % selection['status'],
        '- 锁定结构：`%s`' % selection['selected_variant'],
        '- 合格候选排序：%s' % (
            ' > '.join(selection['qualified_ranking'])
            if selection['qualified_ranking'] else 'none'
        ),
        '- 原因：%s' % selection['reason'],
        '', '排序严格依次使用三 seed 平均论文 27 阈值 AUC、平均 Pd@0.5（降序）、平均 Fa@0.5（升序）、参数量。',
        '', '## 训练收敛诊断', '',
        '| Variant | Epoch-1 loss/IoU | Epoch-32 loss/IoU |',
        '|---|---:|---:|',
    ])
    for variant in VARIANTS:
        subset = [row for row in curve_summaries if row['variant'] == variant]
        first, last = subset[0], subset[-1]
        lines.append(
            '| {label} | {l1:.4f}/{i1:.4f} | '
            '{l32:.4f}/{i32:.4f} |'.format(
                label=grid_analysis.VARIANTS[variant][3],
                l1=first['train_loss_mean'], i1=first['train_iou_mean'],
                l32=last['train_loss_mean'], i32=last['train_iou_mean'],
            )
        )
    lines.extend([
        '', '## 解释边界', '',
        '- 收敛曲线来自训练集聚合量，只用于优化稳定性诊断，不是泛化证据，也不参与选模。',
        '- 第一阶段固定 64/16 internal validation 只用于候选锁定；本工具不读取官方 test 图像或生成其结果。',
        '- 三个 seed 的 sample SD 描述有限训练波动，不构成等效性检验。',
        '- 唯一检测指标合同为 Pd@0.5、Fa@0.5 与论文 27 阈值 Pd-Fa AUC；raw-logit 和 dense-grid 结果只可作补充敏感性分析。',
        '- 本流程不生成或要求文件内容哈希。',
    ])
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    experiment_root = args.experiment_root.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()

    probability_rows, probability_loaded = load_complete_evidence(
        experiment_root, log_root, args.profile,
    )
    require_c3_gate_resolved(probability_rows)
    parameters = {
        entry['job']['run_id']: float(entry['payload']['parameters_m'])
        for entry in probability_loaded.values()
    }
    decisions = evaluate_candidate_eligibility(
        probability_rows, parameters, identity_verified=True,
        profile=args.profile,
    )
    selection = rank_qualified_candidates(decisions)
    if selection['status'] == 'UNRESOLVED_TRADEOFF':
        raise RuntimeError(selection['reason'])

    curve_rows = load_training_curves(log_root, args.profile)
    curve_summaries = summarize_training_curves(curve_rows)
    write_csv(experiment_root / 'noise8_candidate_qualification.csv', decisions)
    write_csv(experiment_root / 'noise8_training_curves.csv', curve_rows)
    write_csv(
        experiment_root / 'noise8_training_curves_summary.csv',
        curve_summaries,
    )
    lock_payload = build_locked_candidate_payload(
        selection, decisions, experiment_root, log_root, args.profile,
    )
    if not args.no_plot:
        plot_training_curves(
            curve_summaries,
            experiment_root / 'noise8_training_convergence.png',
            experiment_root / 'noise8_training_convergence.pdf',
        )
    write_report(
        experiment_root / 'NOISE8_PAPER_ANALYSIS.md',
        decisions, selection, curve_summaries, args.profile,
    )
    # This is the sole completion sentinel. Publish it only after all requested
    # tables, figures and the human-readable report were written successfully.
    atomic_json_dump(
        experiment_root / 'LOCKED_CANDIDATE.json', lock_payload,
    )
    print('Paper evidence gate: 12 official Pd/Fa/AUC payloads verified.')
    print('Candidate selection status: %s' % selection['status'])
    print('Selected variant: %s' % selection['selected_variant'])


if __name__ == '__main__':
    main()
