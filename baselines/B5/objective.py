"""Mainline loss validation and legacy checkpoint resume compatibility."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def validate_mainline_loss(loss: dict[str, Any]) -> None:
    # Old successful checkpoints may explicitly contain slow_trend=0. Accept
    # that metadata, but never silently ignore an enabled experimental loss.
    weight = float(loss.get('slow_trend', 0.))
    if not math.isfinite(weight) or weight != 0:
        raise ValueError('The clean B5 branch uses the original loss; enabled slow-trend experiments remain on baseline/B5')


def loss_signature(loss: dict[str, Any]) -> dict[str, Any]:
    validate_mainline_loss(loss)
    result = {key: float(loss[key]) for key in ('huber', 'pcc', 'anchor', 'physiology', 'huber_delta')}
    result['slow_trend'] = 0.  # Canonical compatibility with historical checkpoints.
    return result


def validate_resume_loss(checkpoint: dict[str, Any], loss: dict[str, Any]) -> None:
    previous = checkpoint.get('loss_config', checkpoint.get('config', {}).get('loss'))
    if previous is None:
        raise ValueError('Checkpoint has no auditable loss configuration')
    if loss_signature(previous) != loss_signature(loss):
        raise ValueError('Exact resume changed loss objective; use --init-checkpoint and a new output directory')


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
