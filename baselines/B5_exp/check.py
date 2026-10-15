"""CPU preflight only: input/target/schema/initialization checks; no optimizer."""
import argparse
import json

import numpy as np
import torch

from .core import SlowCNN, Records, contexts, load_prepared, json_write
from .train import state_hash


def check(prepared, report):
    path, info = load_prepared(prepared)
    datasets = [Records(path / name) for name in ('train.npz', 'validation.npz')]
    for data in datasets:
        expected = {'anchor': (1, 6000), 'target': (11, 6000), 'mask': (11,),
                    'numeric': (3,), 'sex': (), 'field_mask': (4,), 'age_topcoded': (1,)}
        if set(data.arrays) != set(expected) or not len(data):
            raise ValueError('Incomplete prepared fields')
        for key, shape in expected.items():
            if data.arrays[key].shape != (len(data), *shape) or not np.isfinite(data.arrays[key]).all():
                raise ValueError('Invalid prepared field: ' + key)
        if not np.all(data.arrays['mask'].sum(axis=-1) > 0):
            raise ValueError('No usable missing target leads')
    torch.manual_seed(42)
    first = SlowCNN().eval()
    torch.manual_seed(42)
    second = SlowCNN().eval()
    hashes = [state_hash(m.state_dict()) for m in (first, second)]
    assert hashes[0] == hashes[1]
    sample = datasets[0][0]
    inputs = {key: value[None] for key, value in sample.items()
              if key in ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')}
    shapes = {}
    with torch.no_grad():
        for seconds in (10, 30, 50):
            # One center-window forward suffices; no labels enter the model.
            x = contexts(inputs['anchor'], seconds)[:1]
            output = first(x, **{k: v for k, v in inputs.items() if k != 'anchor'})
            if output.shape != (1, 11, seconds * 50) or not torch.isfinite(output).all():
                raise ValueError('Invalid CPU model forward')
            shapes[str(seconds)] = list(output.shape)
    result = {'ready': True, 'training_started': False, 'optimizer_updates': 0,
              'same_initial_weights': True, 'initial_weights_sha256': hashes[0],
              'parameters': sum(p.numel() for p in first.parameters()), 'cpu_forward_shapes': shapes,
              'train_records': len(datasets[0]), 'validation_unique_targets': len(datasets[1]),
              'base_commit': info['base_commit'], 'score_protocol': info['applied_score_protocol'],
              'target_means_removed': False, 'note': 'Functional preflight, not a performance result'}
    json_write(report, result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    check(args.prepared, args.report)


if __name__ == '__main__':
    main()
