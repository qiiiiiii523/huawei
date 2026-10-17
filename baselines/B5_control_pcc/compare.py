"""Compare both completed arms under the SAME control100 reliable-I validation."""
import argparse
from pathlib import Path
import torch
from baselines.B5_best.common import read, write, csv_write, sha
from . import PROTOCOL, GROUPS
from .common import load_study


def compare(run):
    run = Path(run).resolve()
    study_path = run / 'shared/study.json'
    study, _ = load_study(study_path)
    initializations = [read(run / g / 'initialization.json') for g in GROUPS]
    if initializations[0]['train_FM_cache_root'] == initializations[1]['train_FM_cache_root']:
        raise ValueError('Both arms unexpectedly used the same TRAIN FM cache')
    for field in ('initial_slow_state_sha256', 'fixed_FM_state_sha256', 'training',
                  'combo_PCC_weight', 'sampling', 'beta', 'cache_sha256'):
        if initializations[0][field] != initializations[1][field]:
            raise ValueError('Paired settings differ: ' + field)
    rows = []
    for group in GROUPS:
        if not read(run / group / 'completed.json')['completed']:
            raise ValueError('Both paired arms must finish before comparing')
        for name in ('best', 'last'):
            cp = torch.load(run / group / (name + '.pt'), map_location='cpu', weights_only=True)
            score = cp['validation_score']
            if (cp['protocol'] != PROTOCOL or cp['study_sha256'] != sha(study_path)
                    or score['score_protocol'] != study['score_protocol']):
                raise ValueError('Checkpoint/study/scoring scope mismatch')
            rows.append({'group': group, 'checkpoint': name, 'epoch': cp['epoch'], **score,
                         'delta_from_epoch0': score['main_correlation'] - study['reference_score']['main_correlation']})
    csv_write(run / 'comparison.csv', rows)
    write(run / 'comparison.json', {'completed': True, 'validation_FM': 'control100_both_arms', 'rows': rows,
          'best_adapt_minus_control': rows[2]['main_correlation'] - rows[0]['main_correlation']})
    for row in rows:
        print(row, flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    compare(p.parse_args().run)
