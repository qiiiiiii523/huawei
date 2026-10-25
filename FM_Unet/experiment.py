"""Load and check slow-FM experiment plans; this is not a training launcher."""
import argparse
import json
from pathlib import Path

import torch

from .config import LossConfig, ModelConfig, read_json
from .fm_experiment import build


def load_experiment(path):
    config = read_json(path)
    if set(config) != {'stage', 'model', 'data', 'loss', 'training', 'validation'}:
        raise ValueError('Expected stage/model/data/loss/training/validation')
    model_config = ModelConfig.from_dict(config['model'])
    LossConfig(**config['loss'])
    data = config['data']
    expected = {'public10': (10, 10), 'public75': (5, 75), 'huawei120': (5, 120)}
    if config['stage'] not in expected:
        raise ValueError('Unknown stage')
    if (data['native_hz'], data['input_seconds']) != expected[config['stage']]:
        raise ValueError('Stage rate/window mismatch')
    if data['output_seconds'] != data['input_seconds'] or data['target'] != 'MA501':
        raise ValueError('Expected full real-record MA501 slow targets')
    training = config['training']
    for key in ('updates', 'batch_size', 'checkpoint_every'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f'Positive integer required: {key}')
    if training['optimizer'] != 'AdamW' or training['learning_rate'] <= 0 or training['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    if not 0 <= training['ema_decay'] < 1 or training['gradient_clip'] <= 0:
        raise ValueError('Invalid EMA/clip')
    validation = config['validation']
    if validation['mode'] != 'final_only' or validation['steps'] != 16 or validation['samples'] != 16:
        raise ValueError('Expected final-only Heun16/K16 validation')
    if validation['solver'] != 'heun':
        raise ValueError('Expected Heun solver')
    model = build(model_config, data['native_hz'], data['input_seconds'], data['output_seconds'])
    count = sum(p.numel() for p in model.parameters())
    if len(model_config.channels) != 7 or not 10_000_000 < count < 20_000_000:
        raise ValueError('Expected seven levels with 10M–20M parameters')
    return config, model


def build_optimizer(model, config):
    training = config['training']
    return torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                             weight_decay=training['weight_decay'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config, model = load_experiment(args.config)
    print(json.dumps({'stage': config['stage'], 'parameters': sum(p.numel() for p in model.parameters()),
                      'model': model.config.to_dict(), 'data': config['data'],
                      'training': config['training'], 'validation': config['validation'],
                      'status': 'plan_checked_only; no data loaded or training started'}, indent=2))


if __name__ == '__main__':
    main()
