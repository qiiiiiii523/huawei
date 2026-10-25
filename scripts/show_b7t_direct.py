"""Read progress/results without loading checkpoints or requiring torch."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'results/B7-T-direct-bs16')
args = p.parse_args()
for preset in ('g6_19m', 'w144_g5_21m'):
    root = args.root/preset/'seed42'
    print(f'\n========== {preset} ==========')
    status = root/'pipeline_status.json'
    if not status.exists():
        print('No status yet; inspect launcher.log')
        continue
    value = json.loads(status.read_text())
    print('All complete:', value['success'])
    for name, state in value['jobs'].items():
        result = value['results'].get(name, {})
        score = result.get('best_r')
        print(name, state, 'r='+f'{score:.6f}' if score is not None else '')
        if 'error' in result:
            print('ERROR:', result['error'])
        log = root/'logs'/(name+'.log')
        if log.exists() and state != 'done':
            print('\n'.join(log.read_text(errors='replace').splitlines()[-3:]))
    launcher = root/'launcher.log'
    if launcher.exists():
        print('Controller:', '\n'.join(launcher.read_text(errors='replace').splitlines()[-2:]))
