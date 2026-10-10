"""Analyze ALL saved validation windows on CPU; never runs or trains a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnostic_signals import (LEADS, aggregate_leads, boxcar_width, fresh_directory,
                                 read_rows, record_groups, record_metrics, write_rows)


def analyze_task(folder: Path, width: int, window_stats: list | None = None):
    metadata = read_rows(folder / 'window_metadata.csv')
    p = np.load(folder / 'prediction_uV.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(folder / 'target_uV.npy', mmap_mode='r', allow_pickle=False)
    if p.shape != y.shape or p.shape != (len(metadata), 12, 5000):
        raise ValueError(f'Invalid raw-uV array shape: {folder}')
    groups = record_groups(metadata)
    stats, digest = [], hashlib.sha256()
    for index in range(len(y)):
        if not np.isfinite(p[index]).all() or not np.isfinite(y[index]).all():
            raise ValueError(f'Nonfinite window {index}: {folder}')
        digest.update(np.ascontiguousarray(y[index], dtype=np.float64).tobytes())
    for key in sorted(groups):
        indices = groups[key]
        prediction = np.concatenate([p[i] for i in indices], axis=1)
        target = np.concatenate([y[i] for i in indices], axis=1)
        # Filter each complete scored record, never individual 10-second windows.
        for lead in range(1, 12):
            stats.append({'pair_id': key, 'target_record_id': metadata[indices[0]]['target_record_id'],
                          'subject_id': metadata[indices[0]].get('subject_id', ''), 'lead': LEADS[lead],
                          **record_metrics(prediction[lead], target[lead], width)})
            if window_stats is not None:
                for i in indices:
                    pm,ym=float(p[i,lead].mean(dtype=np.float64)),float(y[i,lead].mean(dtype=np.float64))
                    window_stats.append({'pair_id':key,'target_record_id':metadata[i]['target_record_id'],
                        'lead':LEADS[lead],'start_sample_500hz':metadata[i]['start_sample_500hz'],
                        'prediction_mean_uV':pm,'target_mean_uV':ym,'mean_error_uV':pm-ym})
    summary = aggregate_leads(stats)
    if any(r['raw_r'] is None for r in summary):
        raise ValueError('A lead has no defined raw correlation')
    raw_r = float(np.mean([r['raw_r'] for r in summary]))
    raw_rmse = float(np.mean([r['raw_rmse_uV'] for r in summary]))
    official = folder / 'overall_metrics.csv'
    if official.exists():
        row = read_rows(official)[0]
        if not np.isclose(raw_r, float(row['r_missing11']), rtol=0, atol=1e-7):
            raise ValueError(f'Recomputed raw r disagrees with saved official report: {folder}')
        if not np.isclose(raw_rmse, float(row['missing11_mean_rmse_uV']), rtol=1e-8, atol=1e-5):
            raise ValueError(f'Recomputed raw RMSE disagrees with saved official report: {folder}')
    keys = [(r.get('pair_id', ''), r['target_record_id'], r['start_sample_500hz']) for r in metadata]
    contract = {'windows': len(metadata), 'records': len(groups), 'raw_r_missing11': raw_r,
                'raw_missing11_mean_rmse_uV': raw_rmse, 'target_sha256_float64': digest.hexdigest(),
                'metadata_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                'official_report_checked': official.exists()}
    return stats, summary, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', required=True, help='Label=directory; repeat for models')
    parser.add_argument('--tasks', nargs='+', choices=('task1', 'task2', 'public'), default=['task1', 'task2'])
    parser.add_argument('--slow-seconds', type=float, default=1.)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    width = boxcar_width(args.slow_seconds)
    evaluations = {}
    for value in args.evaluation:
        label, separator, directory = value.partition('=')
        if not separator or not label or not directory or label in evaluations or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
            parser.error('Use unique simple labels, e.g. --evaluation E1=results/B5/E1_local_meta/best_evaluation')
        evaluations[label] = Path(directory).resolve()
    output = fresh_directory(args.output_dir)
    manifest = {'training_started': False, 'model_inference_started': False, 'diagnostic_only': True,
                'fs_hz': 500, 'boxcar_points': width, 'requested_slow_seconds': args.slow_seconds,
                'edge_policy': 'complete-record edge padding', 'evaluations': {},
                'cross_model_target_and_metadata_match': {},
                'note': 'Filtered metrics are diagnostic, never official scores. Slow/fast MSE have a cross term.'}
    task_summary, reference = [], {}
    for label, directory in evaluations.items():
        manifest['evaluations'][label] = {'directory': str(directory), 'tasks': {}}
        for task in args.tasks:
            print(f'Analyzing {label}/{task} (all complete records)', flush=True)
            windows=[]
            stats, leads, contract = analyze_task(directory / task, width, windows)
            if task in reference:
                match = all(contract[k] == reference[task][k] for k in ('metadata_sha256', 'target_sha256_float64'))
                manifest['cross_model_target_and_metadata_match'][f'{label}/{task}'] = match
                if not match:
                    raise ValueError('Different targets/window ordering across models; align data before comparison')
            else:
                reference[task] = contract
            manifest['evaluations'][label]['tasks'][task] = contract
            destination = output / label / task
            destination.mkdir(parents=True)
            write_rows(destination / 'record_lead_diagnostics.csv', stats)
            write_rows(destination / 'lead_diagnostics.csv', leads)
            write_rows(destination / 'window_mean_diagnostics.csv', windows)
            result = {'model': label, 'task': task, 'records': contract['records'], 'windows': contract['windows'],
                      'raw_r_missing11': contract['raw_r_missing11'],
                      'raw_missing11_mean_rmse_uV': contract['raw_missing11_mean_rmse_uV']}
            for component in ('slow', 'fast'):
                values = [r[component + '_r'] for r in leads if r[component + '_r'] is not None]
                result[component + '_r_missing11_diagnostic'] = float(np.mean(values)) if values else None
                result[component + '_defined_leads'] = len(values)
            chest = [r for r in leads if r['lead'].startswith('V')]
            result['chest_raw_rmse_uV'] = float(np.mean([r['raw_rmse_uV'] for r in chest]))
            result['chest_centered_rmse_uV_diagnostic'] = float(np.mean([r['centered_rmse_uV'] for r in chest]))
            chest_points = [r for r in stats if r['lead'].startswith('V')]
            result['chest_offset_mse_fraction'] = sum(r['offset_mse']*r['points'] for r in chest_points) / max(sum(r['raw_mse']*r['points'] for r in chest_points), 1e-30)
            boundary_count=sum(r['boundary_count'] for r in chest_points)
            boundary_mse=sum(r['window_mean_change_error_mse']*r['boundary_count'] for r in chest_points if r['boundary_count'])
            result['chest_window_mean_change_error_rmse_uV_diagnostic']=float(np.sqrt(boundary_mse/boundary_count)) if boundary_count else None
            result['chest_boundary_point_jump_error_mae_uV_diagnostic']=sum(r['point_jump_error_mae_uV']*r['boundary_count'] for r in chest_points if r['boundary_count'])/boundary_count if boundary_count else None
            task_summary.append(result)
    write_rows(output / 'task_summary.csv', task_summary)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# B5 全验证集快慢变化诊断', '', '只读取已有预测，未训练或运行模型。原始指标保持不变；滤波指标仅作诊断。', '',
             '| 模型 | 数据 | 原始r | 慢变化诊断r | 快变化诊断r | 胸导联RMSE μV | 常数偏移平方误差占比 |',
             '|---|---|---:|---:|---:|---:|---:|']
    fmt = lambda value: 'undefined' if value is None else f'{value:.4f}'
    for r in task_summary:
        lines.append(f"| {r['model']} | {r['task']} | {fmt(r['raw_r_missing11'])} | {fmt(r['slow_r_missing11_diagnostic'])} | {fmt(r['fast_r_missing11_diagnostic'])} | {r['chest_raw_rmse_uV']:.2f} | {100*r['chest_offset_mse_fraction']:.1f}% |")
    lines += ['', '慢/快变化通过每条完整评分记录的移动平均分离，两端edge padding。',
              '不能把诊断r当作正式分数；不能将去均值或真实目标慢曲线用于推理校正。',
              'slow MSE与fast MSE不直接相加，record表同时报告误差交叉项。',
              'lead表的r按记录平均、RMSE按点加权，与原始评估聚合一致。']
    lines += ['', '跨窗口指标见task_summary.csv和lead表；window_mean_diagnostics.csv列出每个10秒窗口的均值误差。',
              '跳变误差扣除真实目标自身的变化，不能把自然波峰直接当作拼接问题。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Done: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
