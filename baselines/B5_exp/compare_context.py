"""Compare existing matched C/S30 with S50, keeping differing epoch-0 scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import sha, csv_read, csv_write, json_write
from .context_audit import CONTEXT_PROTOCOL, CANDIDATES, read, canonical, verify_control_files
from .check_loss import implementation_hashes


def compare(s30, s50, output):
    paths = [Path(s30).resolve(), Path(s50).resolve()]
    configs = [read(path / 'resolved_config.json') for path in paths]
    if [c['context_seconds'] for c in configs] != [30, 50] or canonical(configs[0]) != canonical(configs[1]):
        raise ValueError('Expected matched original-loss S30/S50; changed more than input context')
    if configs[1].get('context_extension_protocol') != CONTEXT_PROTOCOL:
        raise ValueError('S50 compatibility preflight missing')
    preflight = read(paths[1] / 'context_preflight.json')
    proof = preflight['s50_compatibility']
    if sha(paths[1] / 'context_preflight.json') != configs[1]['calibration_sha256'] or not proof.get('s30_bitwise_equal'):
        raise ValueError('Invalid S50 preflight snapshot')
    if Path(proof['control_reference']['path']).resolve() != paths[0] or proof['control_reference'] != configs[1]['control_reference']:
        raise ValueError('S50 was paired with a different C control')
    verify_control_files(proof['control_reference'])
    if configs[1]['implementation_sha256'] != implementation_hashes():
        raise ValueError('Run comparison with the S50 implementation used in training')
    rows, wave_rows, history_rows, hashes = [], [], [], []
    baseline = None
    for path, config, tag in zip(paths, configs, ('S30', 'S50')):
        completed = read(path / 'completed.json')
        if not completed.get('completed') or completed['additional_epochs'] != config['epochs']:
            raise ValueError('Run is incomplete: ' + tag)
        initial = read(path / 'initial_evaluation.json')
        if baseline is not None and initial['baseline_score'] != baseline:
            raise ValueError('Different fixed B5 baseline scores')
        baseline = initial['baseline_score']
        rows.append({'context': tag, 'candidate': 'initializer', 'epoch': 0, **initial['hybrid_score']})
        for candidate in CANDIDATES:
            folder = path / ('evaluation_' + candidate)
            info = read(folder / 'evaluation.json')
            if not info.get('completed') or info['context_seconds'] != config['context_seconds'] or info.get('group') != 'C':
                raise ValueError('Wrong context/candidate evaluation')
            if info['probe_checkpoint_sha256'] != sha(path / (candidate + '.pt')) or info['prepared_sha256'] != config['prepared_sha256']:
                raise ValueError('Stale context evaluation')
            if info['B5_checkpoint_sha256'] != proof['control_reference']['B5_checkpoint_sha256']:
                raise ValueError('Fixed B5 model differs')
            if candidate == 'last' and info['epoch'] != config['epochs']:
                raise ValueError('Last checkpoint did not complete budget')
            score = next(row for row in info['scores'] if row['method'] == 'learned_slow_hybrid')
            rows.append({'context': tag, 'candidate': candidate, 'epoch': info['epoch'],
                         **{k:v for k,v in score.items() if k != 'method'}})
            wave_rows.extend({'context': tag, 'candidate': candidate, **row} for row in csv_read(folder / 'wave_summary.csv'))
            hashes.append({'context': tag, 'candidate': candidate, 'evaluation_sha256': sha(folder / 'evaluation.json')})
        history = [json.loads(line) for line in (path / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if [row['epoch'] for row in history] != list(range(config['epochs'] + 1)):
            raise ValueError('Missing or duplicated epoch history')
        for row in history:
            history_rows.append({'context': tag, 'epoch': row['epoch'], 'loss': row.get('train', {}).get('loss'),
                'slow_r': row.get('validation', {}).get('slow_r_unique_quality_good_50Hz'),
                'total_score': row.get('validation', {}).get('hybrid_score', {}).get('competition_total_score')})
    for row in rows:
        control = next(r for r in rows if r['context'] == 'S30' and r['candidate'] == row['candidate'])
        initial = next(r for r in rows if r['context'] == row['context'] and r['candidate'] == 'initializer')
        row['delta_total_vs_S30_same_candidate'] = row['competition_total_score'] - control['competition_total_score']
        row['delta_total_vs_own_initializer'] = row['competition_total_score'] - initial['competition_total_score']
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use empty comparison output')
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / 'raw_scores.csv', rows)
    csv_write(output / 'wave_diagnostics.csv', wave_rows)
    csv_write(output / 'history.csv', history_rows)
    json_write(output / 'audit.json', {'completed': True, 'protocol': CONTEXT_PROTOCOL,
        's30_compatibility_verified': True, 'same_initial_weights': configs[0]['initial_weights_sha256'],
        'control_reference': proof['control_reference'], 'evaluations': hashes,
        'target_used_at_inference': False, 'only_active_change': 'input_I_context_30_to_50_seconds'})
    lines = ['# S30与S50原CNN输入范围实验', '',
        '同一S30第50轮权重、原损失、seed42及新增20轮，只改变同步I可见范围。固定B5快分量与慢CNN预测组合，尚未接入FM。', '',
        '| 输入 | 候选 | 新增轮次 | Task1 r | Task2 r | 机器RMSE | 体脂秤RMSE | 总分/60 | 相对S30 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['context']} | {row['candidate']} | {row['epoch']} | {row['task1_r_missing11']:.6f} | "
            f"{row['task2_r_missing11']:.6f} | {row['ecg_machine_d6_task2_missing_lead_mean_rmse_uV']:.2f} | "
            f"{row['body_scale_d6_task2_missing_lead_mean_rmse_uV']:.2f} | {row['competition_total_score']:.3f} | "
            f"{row['delta_total_vs_S30_same_candidate']:+.3f} |")
    lines += ['', '先比较相同20轮last，再比较相同选模规则的best，同时看正常导联慢r、快细节、独有目标和窗口边界。',
        '两组权重初始化相同，但可见输入不同，因此第0轮输出和分数允许不同；不能据此判断权重未正确加载。',
        'S50比S30好支持当前CNN能利用更多I；收益有限不证明更远I没有信息。CNN名义感受野约40.9秒，并非每个输出都利用全部50秒。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(str(output / 'report.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s30', required=True)
    parser.add_argument('--s50', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.s30, args.s50, args.output)


if __name__ == '__main__':
    main()
