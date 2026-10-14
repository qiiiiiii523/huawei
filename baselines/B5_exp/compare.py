from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import csv_read, csv_write, json_write, sha


def compare(s10, s30, output):
    paths = [Path(s10), Path(s30)]
    infos = [json.loads((p / 'evaluation.json').read_text(encoding='utf-8')) for p in paths]
    configs = [json.loads((p.parent / 'resolved_config.json').read_text(encoding='utf-8')) for p in paths]
    if [i['context_seconds'] for i in infos] != [10, 30] or not all(i['completed'] for i in infos):
        raise ValueError('Expected completed S10/S30 evaluations in each run/evaluation_best directory')
    if infos[0]['prepared_sha256'] != infos[1]['prepared_sha256'] or infos[0]['B5_checkpoint_sha256'] != infos[1]['B5_checkpoint_sha256']:
        raise ValueError('Compared different data or original B5 checkpoints')
    ignore = {'context_seconds'}
    if {k: v for k, v in configs[0].items() if k not in ignore} != {k: v for k, v in configs[1].items() if k not in ignore}:
        raise ValueError('S10/S30 changed more than input context')
    rows = []
    for tag, info in zip(('S10', 'S30'), infos):
        for score in info['scores']:
            if tag == 'S30' and score['method'] == 'baseline':
                if score != infos[0]['scores'][0]:
                    raise ValueError('B5 baseline differs between experiments')
                continue
            rows.append({'experiment': 'B5' if score['method'] == 'baseline' else tag, **score})
    slow_rows = []
    for tag, p in zip(('S10', 'S30'), paths):
        slow_rows.extend({'experiment': tag, **r} for r in csv_read(p / 'slow_summary.csv'))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use empty comparison output')
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / 'raw_scores.csv', rows)
    csv_write(output / 'slow_curves.csv', slow_rows)
    lines = ['# B5-exp S10/S30能力实验', '',
             '两组网络、初始化、人口学、标签、患者划分、预算一致，只改变同步I输入范围。', '',
             '| 设置 | Task1 r | Task2 r | 机器胸导联RMSE μV | 体脂秤胸导联RMSE μV | 总分/60 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['task1_r_missing11']:.6f} | {r['task2_r_missing11']:.6f} | "
                     f"{r['ecg_machine_d6_task2_missing_lead_mean_rmse_uV']:.2f} | "
                     f"{r['body_scale_d6_task2_missing_lead_mean_rmse_uV']:.2f} | {r['competition_total_score']:.3f} |")
    lines += ['', 'S10/S30这里指固定B5快分量＋小网络预测慢曲线的诊断混合波形。没有用真实慢曲线作推理校正，尚未把网络加入FM。', '',
              '结合slow_curves.csv检查完整120秒慢r/RMSE及独有目标；不要只看训练loss或一张图。',
              'S30改善才支持增加当前记录I上下文；一次固定种子探索不足以证明稳定收益，必要时再做多种子确认。',
              '相邻I来自当前120秒记录，不是历史设备context。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_write(output / 'audit.json', {'completed': True, 'paired_settings_equal_except_context': True,
                                     'same_initial_weights': configs[0]['initial_weights_sha256'],
                                     'prepared_sha256': infos[0]['prepared_sha256'], 'target_used_at_inference': False,
                                     'evaluation_json_sha256': [sha(p / 'evaluation.json') for p in paths]})
    print(str(output / 'report.md'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s10', required=True)
    parser.add_argument('--s30', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.s10, args.s30, args.output)


if __name__ == '__main__':
    main()
