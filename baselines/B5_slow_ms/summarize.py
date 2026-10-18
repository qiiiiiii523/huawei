"""Read completed arms and independent evaluation scores; no model inference."""
import argparse
from pathlib import Path
from baselines.B5_best.common import read, write, csv_write
from . import GROUPS, PROTOCOL


def summarize(run):
    run = Path(run).resolve()
    study = read(run / 'shared/study.json')
    if study.get('protocol') != PROTOCOL:
        raise ValueError('Wrong residual experiment')
    rows = [{'group': 'initializer', 'checkpoint': 'source_best', 'epoch': 0,
             **{k: study['reference_score'][k] for k in ('task1_r_missing11', 'task2_r_missing11', 'main_correlation')}}]
    for group in GROUPS:
        done = read(run / group / 'completed.json')
        if done.get('completed') is not True:
            raise ValueError('Arm did not complete: ' + group)
        for checkpoint in ('best', 'last'):
            pointer = run / group / f'latest_{checkpoint}_evaluation.txt'
            evaluation = read(Path(pointer.read_text().strip()) / 'evaluation.json')
            if evaluation.get('completed') is not True or evaluation['group'] != group:
                raise ValueError('Missing matching independent evaluation')
            rows.append({'group': group, 'checkpoint': checkpoint, 'epoch': evaluation['epoch'],
                         **{k: evaluation['score'][k] for k in ('task1_r_missing11', 'task2_r_missing11', 'main_correlation')}})
    for row in rows:
        row['delta_vs_initializer'] = row['main_correlation'] - rows[0]['main_correlation']
    csv_write(run / 'summary.csv', rows)
    write(run / 'summary.json', {'completed': True, 'score_protocol': study['score_protocol'], 'rows': rows})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    for row in summarize(args.run):
        print(row)


if __name__ == '__main__':
    main()
