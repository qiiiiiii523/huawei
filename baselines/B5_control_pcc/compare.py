"""Compare A/B/C under identical control100 reliable-I validation."""
import argparse
from pathlib import Path
import torch
from baselines.B5_best.common import read, write, csv_write, sha
from . import PROTOCOL, GROUPS
from .common import load_study, multiscale_for_group


def compare(run):
    run = Path(run).resolve()
    study_path = run / 'shared/study.json'
    study, _ = load_study(study_path)
    initializations = [read(run / g / 'initialization.json') for g in GROUPS]
    if (initializations[0]['train_FM_cache_root'] == initializations[1]['train_FM_cache_root']
            or initializations[1]['train_FM_cache_root'] != initializations[2]['train_FM_cache_root']):
        raise ValueError('A/B must use different TRAIN FM caches; B/C must use the same one')
    for group, initialization in zip(GROUPS, initializations):
        if initialization.get('multiscale') != multiscale_for_group(study, group):
            raise ValueError('Incorrect group multiscale configuration')
    for field in ('initial_slow_state_sha256', 'fixed_FM_state_sha256', 'training',
                  'combo_PCC_weight', 'sampling', 'beta', 'cache_sha256', 'score_protocol'):
        if any(item[field] != initializations[0][field] for item in initializations[1:]):
            raise ValueError('Shared settings differ: ' + field)
    rows, scores = [], {}
    for label, group in zip('ABC', GROUPS):
        if not read(run / group / 'completed.json')['completed']:
            raise ValueError('All three arms must finish before comparing')
        for name in ('best', 'last'):
            cp = torch.load(run / group / (name + '.pt'), map_location='cpu', weights_only=True)
            score = cp['validation_score']
            if (cp['protocol'] != PROTOCOL or cp['group'] != group or cp['study_sha256'] != sha(study_path)
                    or cp.get('multiscale') != multiscale_for_group(study, group)
                    or score['score_protocol'] != study['score_protocol']):
                raise ValueError('Checkpoint/study/scoring scope mismatch')
            rows.append({'arm': label, 'group': group, 'checkpoint': name, 'epoch': cp['epoch'], **score,
                         'multiscale_weight': cp['multiscale']['weight'],
                         'delta_from_epoch0': score['main_correlation'] - study['reference_score']['main_correlation']})
            scores[(label, name)] = score['main_correlation']
    differences = {name + '_B_minus_A': scores[('B', name)] - scores[('A', name)] for name in ('best', 'last')}
    differences.update({name + '_C_minus_B': scores[('C', name)] - scores[('B', name)] for name in ('best', 'last')})
    csv_write(run / 'comparison.csv', rows)
    write(run / 'comparison.json', {'completed': True, 'validation_FM': 'control100_all_three_arms',
                                    'rows': rows, 'comparisons': differences})
    for row in rows:
        print(row, flush=True)
    print(differences, flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    compare(p.parse_args().run)
