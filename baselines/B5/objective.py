"""Canonical loss settings and resume compatibility, without importing Torch."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def slow_settings(loss: dict[str, Any]) -> dict[str, Any]:
    weight = float(loss.get('slow_trend', 0.))
    seconds = float(loss.get('slow_trend_seconds', 1.))
    delta = float(loss.get('slow_trend_delta', 1.))
    leads = loss.get('slow_trend_leads', 'chest6')
    if not all(math.isfinite(v) for v in (weight, seconds, delta)) or weight < 0 or seconds <= 0 or delta <= 0:
        raise ValueError('Slow-trend weight must be finite/nonnegative; seconds/delta finite/positive')
    width = max(3, int(round(seconds * 500)))
    width += int(width % 2 == 0)
    if width > 5000:
        raise ValueError('Slow-trend window must fit the 5000-point training window')
    if leads not in ('chest6', 'missing11'):
        raise ValueError('slow_trend_leads must be chest6 or missing11')
    return {'weight': weight, 'seconds': seconds, 'width': width, 'delta': delta, 'leads': leads}


def loss_signature(loss: dict[str, Any]) -> dict[str, Any]:
    settings = slow_settings(loss)
    result = {key: float(loss[key]) for key in ('huber', 'pcc', 'anchor', 'physiology', 'huber_delta')}
    result['slow_trend'] = settings['weight']
    if settings['weight']:
        result.update({'slow_trend_width': settings['width'], 'slow_trend_delta': settings['delta'],
                       'slow_trend_leads': settings['leads']})
    return result


def validate_resume_loss(checkpoint: dict[str, Any], loss: dict[str, Any]) -> None:
    previous = checkpoint.get('loss_config', checkpoint.get('config', {}).get('loss'))
    if previous is None:
        raise ValueError('Checkpoint has no auditable loss configuration')
    if loss_signature(previous) != loss_signature(loss):
        raise ValueError('Exact resume changed loss objective; use --init-checkpoint and a new output directory')


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_pair_configs(control: dict[str, Any], experiment: dict[str, Any]) -> None:
    """The only substantive difference may be slow-trend weight and output identity."""
    for section in ('stage', 'model', 'training', 'validation', 'sampling'):
        if control[section] != experiment[section]:
            raise ValueError(f'Paired configurations differ in {section}')
    if control['stage'] != 'finetune' or int(control['training']['epochs']) != 20:
        raise ValueError('This paired experiment requires finetune stage and 20 epochs')
    if control['training'].get('validate_initial') is not True:
        raise ValueError('Both paired runs must validate/save the initializer before optimizer updates')
    from .config import resolve_path
    if resolve_path(control, 'output_dir') == resolve_path(experiment, 'output_dir'):
        raise ValueError('Paired outputs must be different directories')
    if set(control['paths']) != set(experiment['paths']):
        raise ValueError('Paired data-path fields differ')
    for key in control['paths']:
        if key != 'output_dir' and resolve_path(control, key) != resolve_path(experiment, key):
            raise ValueError(f'Paired data paths differ: {key}')
    a, b = dict(control['loss']), dict(experiment['loss'])
    ca, cb = slow_settings(a), slow_settings(b)
    if ca['weight'] != 0 or cb['weight'] <= 0:
        raise ValueError('Control requires slow_trend=0; experiment requires slow_trend>0')
    a.pop('slow_trend', None); b.pop('slow_trend', None)
    if a != b or {k:v for k,v in ca.items() if k!='weight'} != {k:v for k,v in cb.items() if k!='weight'}:
        raise ValueError('Only the slow-trend weight may differ in paired loss settings')
