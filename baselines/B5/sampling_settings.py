"""Shared deployment sampling profile; independent of checkpoint/training settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


DEFAULT_SAMPLING_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'b5_inference.yaml'
SAMPLING_KEYS = ('seed', 'solver', 'steps', 'samples')


def inference_sampling(path: str | Path = DEFAULT_SAMPLING_CONFIG,
                       overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import yaml
    with Path(path).open(encoding='utf-8-sig') as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict) or not isinstance(profile.get('sampling'), dict):
        raise ValueError('Sampling profile must contain a sampling mapping')
    settings = dict(profile['sampling'])
    if set(settings) != set(SAMPLING_KEYS):
        raise ValueError('Sampling profile requires exactly seed, solver, steps and samples')
    if overrides:
        if set(overrides) - set(SAMPLING_KEYS):
            raise ValueError('Unknown sampling override')
        settings.update({key: value for key, value in overrides.items() if value is not None})
    for key in ('seed', 'steps', 'samples'):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int):
            raise ValueError(f'Sampling {key} must be an integer')
    if not 0 <= settings['seed'] < 2**32:
        raise ValueError('Sampling seed must be in [0, 2**32)')
    if settings['steps'] < 1 or settings['samples'] < 1:
        raise ValueError('Sampling steps and samples must be positive')
    if settings['solver'] not in {'heun', 'euler'}:
        raise ValueError('Sampling solver must be heun or euler')
    return settings


def add_sampling_arguments(parser):
    parser.add_argument('--sampling-config', default=str(DEFAULT_SAMPLING_CONFIG),
                        help='Final inference profile; default configs/b5_inference.yaml (Heun16, K16)')
    parser.add_argument('--steps', type=int, help='Optional override of the final profile')
    parser.add_argument('--samples', type=int, help='Optional K override; average K independent ODE outputs')
    parser.add_argument('--solver', choices=('heun', 'euler'))
    parser.add_argument('--seed', type=int)


def sampling_from_arguments(args):
    return inference_sampling(args.sampling_config,
                              {key: getattr(args, key) for key in SAMPLING_KEYS})
