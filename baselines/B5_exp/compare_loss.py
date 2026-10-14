"""Audit paired C/M/F settings and compare all three final candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import sha, csv_read, csv_write, json_write
from .losses import LOSS_PROTOCOL


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare(paths, output):
    paths = [Path(path).resolve() for path in paths]
    configs = [read(path / 'resolved_config.json') for path in paths]
    ignore = {'group', 'mean_weight', 'residual_weight'}
    common = {k: v for k, v in configs[0].items() if k not in ignore}
    if [c['group'] for c in configs] != ['C', 'M', 'F']:
        raise ValueError('Expected C, M, F run directories in order')
    for config in configs:
        if config.get('experiment_protocol') != LOSS_PROTOCOL or {k: v for k, v in config.items() if k not in ignore} != common:
            raise ValueError('Runs differ in more than the single auxiliary loss')
        if config['mean_weight'] != (config['calibrated_weights']['mean'] if config['group'] == 'M' else 0.):
            raise ValueError('Unexpected mean loss setting')
        if config['residual_weight'] != (config['calibrated_weights']['residual'] if config['group'] == 'F' else 0.):
            raise ValueError('Unexpected residual loss setting')
    score_rows, wave_rows, history_rows, hashes = [], [], [], []
    initial_scores = []
    for config, path in zip(configs, paths):
        completed = read(path / 'completed.json')
        if not completed.get('completed') or completed['additional_epochs'] != config['epochs']:
            raise ValueError('Training is incomplete: ' + str(path))
        initial = read(path / 'initial_evaluation.json')
        initial_scores.append(initial['hybrid_score'])
        score_rows.append({'group': config['group'], 'candidate': 'initializer', 'epoch': 0, **initial['hybrid_score']})
        for candidate in ('best_slow_r', 'best_score', 'last'):
            folder = path / ('evaluation_' + candidate)
            info = read(folder / 'evaluation.json')
            if not info.get('completed') or info.get('group') != config['group'] or info.get('experiment_protocol') != LOSS_PROTOCOL:
                raise ValueError('Missing or wrong candidate evaluation')
            if info['probe_checkpoint_sha256'] != sha(path / (candidate + '.pt')) or info['prepared_sha256'] != config['prepared_sha256']:
                raise ValueError('Stale candidate evaluation')
            if candidate == 'last' and info['epoch'] != config['epochs']:
                raise ValueError('Last checkpoint did not complete budget')
            for score in info['scores']:
                if score['method'] == 'learned_slow_hybrid':
                    score_rows.append({'group': config['group'], 'candidate': candidate, 'epoch': info['epoch'],
                                       **{k: v for k, v in score.items() if k != 'method'}})
            wave_rows.extend({'group': config['group'], 'candidate': candidate, **r} for r in csv_read(folder / 'wave_summary.csv'))
            hashes.append({'group': config['group'], 'candidate': candidate, 'evaluation_sha256': sha(folder / 'evaluation.json')})
        history = [json.loads(line) for line in (path / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if [r['epoch'] for r in history] != list(range(config['epochs'] + 1)):
            raise ValueError('Missing or duplicated epoch history')
        for row in history:
            history_rows.append({'group': config['group'], 'epoch': row['epoch'],
                                 'loss': row.get('train', {}).get('loss'),
                                 'slow_r': row.get('validation', {}).get('slow_r_unique_quality_good_50Hz'),
                                 'total_score': row.get('validation', {}).get('hybrid_score', {}).get('competition_total_score')})
    fields = ('task1_r_missing11', 'task2_r_missing11', 'competition_total_score')
    for other in initial_scores[1:]:
        if not np.allclose([initial_scores[0][k] for k in fields], [other[k] for k in fields], atol=1e-6, rtol=0):
            raise ValueError('Common initializer generated different initial scores')
    for row in score_rows:
        control = next(r for r in score_rows if r['group'] == 'C' and r['candidate'] == row['candidate'])
        row['delta_total_vs_C_same_candidate'] = row['competition_total_score'] - control['competition_total_score']
        row['delta_total_vs_initializer'] = row['competition_total_score'] - initial_scores[0]['competition_total_score']
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use empty comparison output')
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / 'raw_scores.csv', score_rows)
    csv_write(output / 'wave_diagnostics.csv', wave_rows)
    csv_write(output / 'history.csv', history_rows)
    json_write(output / 'audit.json', {'completed': True, 'protocol': LOSS_PROTOCOL, 'common_configuration': common,
                                     'same_initial_weights': True, 'evaluations': hashes, 'target_used_at_inference': False})
    lines = ['# C/M/F慢曲线监督实验', '', '三组从同一S30权重出发，网络和训练预算一致。结果是固定B5快分量与新慢预测的组合，尚未接入FM。', '',
             '| 组别 | 候选 | 新增轮次 | Task1 r | Task2 r | 机器RMSE | 体脂秤RMSE | 总分/60 | 相对C | 相对初始化 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in score_rows:
        lines.append(f"| {row['group']} | {row['candidate']} | {row['epoch']} | {row['task1_r_missing11']:.6f} | "
                     f"{row['task2_r_missing11']:.6f} | {row['ecg_machine_d6_task2_missing_lead_mean_rmse_uV']:.2f} | "
                     f"{row['body_scale_d6_task2_missing_lead_mean_rmse_uV']:.2f} | {row['competition_total_score']:.3f} | "
                     f"{row['delta_total_vs_C_same_candidate']:+.3f} | {row['delta_total_vs_initializer']:+.3f} |")
    lines += ['', '先比较三组last的同等20轮结果，再比较同一选模规则的best。M/F超过C但低于共同初始化，只代表减轻退步。',
              '查看wave_diagnostics.csv中的正常导联/独有目标、V2/V3/V4局部图、快残差和边界误差；不能只看总分。',
              '正常导联诊断排除flagged导联；完整60分评分保留现有集合，两者不可混用。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(output / 'report.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--mean', required=True)
    parser.add_argument('--residual', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare([args.control, args.mean, args.residual], args.output)


if __name__ == '__main__':
    main()
