import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best import SAMPLING
from baselines.B5_best.common import read, sha, state_sha
from baselines.B5_best.bundle import load_bundle, bundle_slow_beta
from baselines.B5_public_slow.common import model_from_bundle, verify_entries
from baselines.B5_public_slow.core import objective as original_objective
from . import PROTOCOL, GROUPS
from .multiscale import multiscale_shape_loss


def code_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ('baselines/B5_p_all', 'baselines/B5_public_slow', 'baselines/B5_best', 'baselines/B5', 'ecg12gen'):
        for p in sorted((root / directory).glob('*.py')):
            digest.update(p.relative_to(root).as_posix().encode())
            digest.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def group_settings(study, group):
    if group not in GROUPS:
        raise ValueError('Unknown group')
    settings = copy.deepcopy(study['multiscale'])
    settings['enabled_bands'] = [0, 1, 2]
    settings['weight'] = settings['weight'] if group == 'all' else 0.
    return settings


def objective(prediction, batch, scales, study, group):
    loss, parts = original_objective(prediction, batch, scales, study['residual_weight'], study['combo_weight'])
    settings = group_settings(study, group)
    extra = prediction.sum() * 0.
    metrics = {}
    if settings['weight']:
        extra, metrics = multiscale_shape_loss(prediction, batch['slow_target'], batch['mask'][:, 1:], settings)
        loss = loss + settings['weight'] * extra
    return loss, {**parts, 'multiscale_loss': extra, 'weighted_multiscale_loss': settings['weight'] * extra, **metrics}


def load_study(path, deep=False):
    path = Path(path).resolve()
    s = read(path)
    if (s.get('protocol') != PROTOCOL or s.get('score_protocol') != SCORE_PROTOCOL or
            s.get('sampling') != SAMPLING or s.get('beta') != .75 or s.get('groups') != list(GROUPS) or
            s.get('code_sha256') != code_digest()):
        raise ValueError('Study protocol/sampling/code changed; use its frozen worktree')
    for name, expected in s['source_files'].items():
        if sha(name) != expected:
            raise ValueError('Read-only source changed: ' + name)
    components = path.parent / 'components.pt'
    if sha(components) != s['bundle_sha256']:
        raise ValueError('Shared P-best model changed')
    bundle = load_bundle(components)
    if (bundle['slow_state_sha256'] != s['initial_slow_state_sha256'] or bundle_slow_beta(bundle) != .75 or
            bundle.get('reference_score_protocol') != SCORE_PROTOCOL):
        raise ValueError('Wrong P-best slow weights or beta/reference protocol')
    cfg = s['multiscale']
    floors = np.asarray(cfg.get('std_floor'))
    if (cfg.get('protocol') != 'native50-full120-multiscale-shape-v1' or cfg.get('widths') != [251,751] or
            cfg.get('shape_huber_weight') != .1 or cfg.get('weight') != .1 or
            cfg.get('floor_source') != 'TRAIN-target-QC-only' or floors.shape != (3,11) or
            not np.isfinite(floors).all() or (floors <= 0).any()):
        raise ValueError('The original ALL20 scale definition/coefficient must be retained')
    item = s['huawei']
    if sha(item['manifest']) != item['sha256']:
        raise ValueError('Huawei manifest changed')
    manifest = read(item['manifest'])
    if deep:
        verify_entries(item['root'], manifest['train'] + manifest['validation'])
    return s, bundle, manifest


def load_checkpoint(path, study_path, device):
    s, bundle, _ = load_study(study_path)
    cp = torch.load(path, map_location='cpu', weights_only=True)
    if (cp.get('protocol') != PROTOCOL or cp.get('study_sha256') != sha(study_path) or
            cp.get('group') not in GROUPS or cp.get('FM_state_sha256') != bundle['fm_state_sha256'] or
            cp.get('multiscale') != group_settings(s, cp.get('group')) or
            cp.get('score_protocol') != SCORE_PROTOCOL or cp.get('sampling_config') != SAMPLING):
        raise ValueError('Checkpoint belongs to another paired experiment')
    model = model_from_bundle(bundle, device)
    model.load_state_dict(cp['model'], strict=True)
    return cp, s, bundle, model
