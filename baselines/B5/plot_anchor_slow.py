"""Plot actual validation anchor I versus target/predicted slow leads on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnostic_signals import (LEADS, boxcar_width, fresh_directory, pearson,
                                 read_rows, record_groups, slow_component, write_rows)


def identity(row):
    return (row['pair_id'], row['target_record_id'], int(row['start_sample_500hz']))


def load_anchor_dataset(config_path, task):
    # Same raw anchor and eligibility rules as HuaweiValidationDataset; no scales,
    # demographics, checkpoint, model, or GPU are needed for this diagnostic.
    from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
    from .config import load_config, resolve_path
    config = load_config(config_path)
    common = ECGDataConfig.from_yaml(resolve_path(config, 'common_config'))
    common.raw['paths']['data_root'] = str(resolve_path(config, 'huawei_data_root'))
    return JointAnchorDataset(common, task, 'validation')


def anchor_index(dataset):
    rows = [dataset._rows[i] for i in dataset._indices]
    lookup = {}
    for i, row in enumerate(rows):
        key = identity(row)
        if key in lookup:
            raise ValueError(f'Duplicate anchor identity: {key}')
        lookup[key] = (i, row)
    return lookup


def aligned_anchor(dataset, lookup, metadata, target, indices):
    anchors = []
    for i in indices:
        key = identity(metadata[i])
        if key not in lookup:
            raise ValueError(f'Evaluation window absent from validation cache: {key}')
        index, row = lookup[key]
        if metadata[i].get('subject_id') != row.get('subject_id'):
            raise ValueError(f'Anchor subject mismatch: {key}')
        sample = dataset[index]
        # Reject stale evaluation arrays or a different cache, even if IDs match.
        if not np.array_equal(sample.Y_12lead, target[i]):
            raise ValueError(f'Saved evaluation target disagrees with validation cache: {key}')
        anchor = np.asarray(sample.anchor_i_ecg, dtype=np.float64)
        if anchor.shape != (1, 5000) or not np.isfinite(anchor).all():
            raise ValueError(f'Invalid raw synchronous I: {key}')
        anchors.append(anchor[0])
    return np.concatenate(anchors)


def make_curves(anchor, target, prediction, width):
    values = np.asarray([anchor, target, prediction], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected three finite, aligned complete-record signals')
    # Stitch FIRST, then filter; never restart averaging at a 10-second boundary.
    slow = np.stack([slow_component(x, width) for x in values])
    return slow, slow - slow.mean(axis=1, keepdims=True)


def curve_metrics(slow):
    anchor, target, prediction = slow
    return {
        'anchor_target_slow_r_diagnostic': pearson(anchor, target),
        'anchor_prediction_slow_r_diagnostic': pearson(anchor, prediction),
        'prediction_target_slow_r_diagnostic': pearson(prediction, target),
        'anchor_target_slow_difference_r_diagnostic': pearson(np.diff(anchor), np.diff(target)),
        'anchor_slow_mean_uV': float(anchor.mean()),
        'target_slow_mean_uV': float(target.mean()),
        'prediction_slow_mean_uV': float(prediction.mean()),
        'anchor_slow_std_uV': float(anchor.std()),
        'target_slow_std_uV': float(target.std()),
        'prediction_slow_std_uV': float(prediction.std()),
        'prediction_target_slow_rmse_uV_diagnostic': float(np.sqrt(np.mean((prediction-target)**2))),
    }


def render_plot(path, curves, lead, pair_id, centered):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 50 Hz is sufficient for DISPLAY of a ~1-second averaged curve. Metrics and
    # NPZ exports always retain every original 500 Hz point.
    take = np.unique(np.r_[np.arange(0, curves.shape[1], 10), curves.shape[1]-1])
    time = take / 500.
    labels = ('Actual input I', f'Target {lead}', f'Prediction {lead}')
    colors = ('#25804d', '#222222', '#287bd1')
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True, layout='constrained')
    mode = 'each slow curve mean removed FOR DISPLAY ONLY' if centered else 'raw voltage, no centering or calibration'
    fig.suptitle(f'{pair_id} | {lead} | {mode}', fontsize=12)
    for ax, signal, label, color in zip(axes, curves, labels, colors):
        ax.plot(time, signal[take], color=color, linewidth=1.2, label=label)
        for boundary in np.arange(10., curves.shape[1]/500., 10.):
            ax.axvline(boundary, color='gray', linestyle=':', linewidth=.6, alpha=.5)
        ax.set_ylabel('Voltage (uV)')
        ax.legend(loc='upper right')
        ax.grid(alpha=.2)
        ax.set_xlim(time[0], time[-1])
        # Independent voltage axes keep weak I changes visible; never rescale data.
    axes[-1].set_xlabel('Time (s)')
    fig.supxlabel('Moving average of complete record; independent y axes; diagnostic only', fontsize=9)
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def run(evaluation, output, dataset, task='task2', leads=('V3',), pair_ids=(),
        max_records=3, width=501):
    folder = Path(evaluation).resolve() / task
    metadata = read_rows(folder / 'window_metadata.csv')
    groups = record_groups(metadata)
    if not groups:
        raise ValueError('Empty evaluation metadata')
    p = np.load(folder / 'prediction_uV.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(folder / 'target_uV.npy', mmap_mode='r', allow_pickle=False)
    if p.shape != y.shape or p.shape != (len(metadata), 12, 5000):
        raise ValueError('Expected aligned saved raw-uV arrays [windows,12,5000]')
    if not leads or len(set(leads)) != len(leads) or any(lead not in LEADS[1:] for lead in leads):
        raise ValueError('Select distinct missing target leads II through V6')
    if max_records < 1 or width < 3 or width % 2 != 1:
        raise ValueError('Invalid record count or odd moving-average width')
    if len(set(pair_ids)) != len(pair_ids) or any(key not in groups for key in pair_ids):
        raise ValueError('Requested pair IDs must be distinct and present in the evaluation')
    # Validate ALL evaluation identities/subjects before any automatic selection.
    lookup = anchor_index(dataset)
    for row in metadata:
        match = lookup.get(identity(row))
        if match is None or row.get('subject_id') != match[1].get('subject_id'):
            raise ValueError(f'Evaluation metadata does not match validation anchor: {identity(row)}')
    explicit_selection = bool(pair_ids)
    if not explicit_selection:
        ranking = []
        for key, indices in groups.items():
            errors = []
            for lead in leads:
                channel = LEADS.index(lead)
                a = np.concatenate([p[i, channel] for i in indices])
                b = np.concatenate([y[i, channel] for i in indices])
                delta = slow_component(a, width)-slow_component(b, width)
                errors.append(float(np.sqrt(np.mean((delta-delta.mean())**2))))
            ranking.append((float(np.mean(errors)), key))
        ranking.sort()
        positions = np.unique(np.linspace(0, len(ranking)-1, min(max_records, len(ranking))).round().astype(int))
        pair_ids = [ranking[i][1] for i in positions]
    # Read and verify selected actual anchors before creating any output.
    anchors = {key: aligned_anchor(dataset, lookup, metadata, y, groups[key]) for key in pair_ids}
    output = fresh_directory(Path(output))
    metrics, records = [], []
    for ordinal, key in enumerate(pair_ids):
        indices = groups[key]
        target = np.concatenate([y[i] for i in indices], axis=1)
        prediction = np.concatenate([p[i] for i in indices], axis=1)
        if not np.isfinite(target).all() or not np.isfinite(prediction).all():
            raise ValueError(f'Nonfinite saved record: {key}')
        entry = {'pair_id': key, 'target_record_id': metadata[indices[0]]['target_record_id'],
                 'windows': len(indices), 'leads': {}}
        for lead in leads:
            slow, centered = make_curves(anchors[key], target[LEADS.index(lead)], prediction[LEADS.index(lead)], width)
            prefix = f'record_{ordinal:02d}_{lead}_anchor_slow'
            render_plot(output / (prefix+'_raw.png'), slow, lead, key, False)
            render_plot(output / (prefix+'_centered.png'), centered, lead, key, True)
            np.savez_compressed(output / (prefix+'_curves.npz'), time_seconds=np.arange(slow.shape[1])/500.,
                                anchor_slow_uV=slow[0], target_slow_uV=slow[1], prediction_slow_uV=slow[2])
            row = {'pair_id': key, 'lead': lead, 'points': slow.shape[1], **curve_metrics(slow)}
            metrics.append(row)
            entry['leads'][lead] = {'filename_prefix': prefix}
        records.append(entry)
    write_rows(output / 'record_metrics.csv', metrics)
    manifest = {'training_started': False, 'model_inference_started': False, 'diagnostic_only': True,
                'evaluation': str(Path(evaluation).resolve()), 'task': task, 'fs_hz': 500,
                'slow_points': width, 'actual_slow_seconds': width/500., 'display_hz': 50,
                'anchor_source': 'JointAnchorDataset.anchor_i_ecg, raw uV; NOT prediction lead I',
                'anchor_contract': 'validation simulates test-visible synchronous I from same-window d12 I',
                'selected_cache_targets_match_evaluation': True,
                'selection': 'explicit pair IDs' if explicit_selection else 'evenly spaced centered-slow-error ranks',
                'metadata_sha256': hashlib.sha256(json.dumps([identity(r) for r in metadata]).encode()).hexdigest(),
                'records': records,
                'warning': 'Correlations are descriptive, not proof of predictability or official scores; averaging is not physiological baseline identification.'}
    (output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# B5 输入 I 与目标/预测慢曲线', '',
             '读取真实验证 anchor 与已有预测；没有训练、模型推理或目标校正。',
             '每条记录先按时间拼接，再对完整记录移动平均。raw 图保留电压，centered 图仅为显示分别去均值。',
             '三行共用时间轴，各行电压轴独立；虚线为原10秒窗口边界。NPZ保留500Hz完整慢曲线。', '',
             '| pair | 导联 | I与真实慢曲线r | 预测与真实慢曲线r |', '|---|---|---:|---:|']
    fmt = lambda value: 'undefined' if value is None else f'{value:.4f}'
    for row in metrics:
        lines.append(f"| {row['pair_id']} | {row['lead']} | {fmt(row['anchor_target_slow_r_diagnostic'])} | {fmt(row['prediction_target_slow_r_diagnostic'])} |")
    lines += ['', '低相关不等于无法预测，高相关不保证泛化；不要用此表代替正式完整记录评分。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path, help='Directory containing task1/task2 subdirectories')
    parser.add_argument('--task', choices=('task1', 'task2'), default='task2')
    parser.add_argument('--leads', nargs='+', choices=LEADS[1:], default=['V3'])
    parser.add_argument('--pair-id', action='append', default=[], help='Repeat to select the same scored pairs as existing plots')
    parser.add_argument('--max-records', type=int, default=3, help='Automatic representative rank selection only')
    parser.add_argument('--slow-seconds', type=float, default=1.)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    dataset = load_anchor_dataset(args.config, args.task)
    run(args.evaluation, args.output_dir, dataset, args.task, args.leads,
        args.pair_id, args.max_records, boxcar_width(args.slow_seconds))
    print(f'Done: {args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
