"""Read saved round-one best validation results without rerunning inference."""
import argparse
import json
from pathlib import Path
from .round1 import VARIANTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='results', help='Contains round1_C0_seed42 through round1_C4_seed42')
    a = p.parse_args()
    rows = []
    for variant in VARIANTS:
        root = Path(a.root) / f'round1_{variant}_seed42'
        status_path = root / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        for stage in ('P', 'A'):
            result = {'variant': variant, 'stage': stage, 'status': status.get('jobs', {}).get(stage, 'not_started')}
            done_path = root / stage / 'completed.json'
            if done_path.exists():
                done = json.loads(done_path.read_text(encoding='utf-8'))
                report = root / stage / f"validation_{done['best_update']:07d}.json"
                metrics = json.loads(report.read_text(encoding='utf-8'))['metrics']
                weights = {'task1': .5, 'machine': .25, 'body': .25} if stage == 'A' else {
                    g: 1 / len(metrics['lead_groups']) for g in metrics['lead_groups']}
                result.update(best_update=done['best_update'], updates=done['update'], best_r=done['best_r'],
                              stopped_early=done.get('stopped_early', False), best_checkpoint=str(root/stage/'best_r.pt'),
                              leads={name: sum(weights[g] * metrics['lead_groups'][g][str(j)] for g in weights)
                                     for j, name in ((1, 'II'), (2, 'III'), (5, 'aVF'))})
            rows.append(result)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
