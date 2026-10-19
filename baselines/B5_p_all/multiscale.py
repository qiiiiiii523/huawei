"""Exact legacy ALL20 scale supervision and TRAIN-only normalization floors."""
import numpy as np
import torch
from torch.nn import functional as F

LEGACY_FUNCTIONS_SHA256 = '2d04cd1b74083fae52ed76e07915c2bf9858648cfa590e9559ea51b333e7556f'


def smooth(x, width=51):
    if width < 3 or width % 2 != 1 or width > x.shape[-1]:
        raise ValueError("Need an odd smoothing width in [3, record length]")
    radius = width // 2
    padded = torch.cat((x[..., :1].expand(*x.shape[:-1], radius), x,
                        x[..., -1:].expand(*x.shape[:-1], radius)), dim=-1)
    return F.avg_pool1d(padded, width, stride=1)


def scale_bands(x):
    """Approximate <5s, 5..15s and >15s changes, not ideal frequency bands.

    Input is the complete native-50Hz/120s slow curve, never individual 10s chunks.
    No mean is removed from deployed predictions or absolute-voltage labels.
    """
    five, fifteen = smooth(x, 251), smooth(x, 751)
    return (x - five, five - fifteen, fifteen)


def multiscale_shape_loss(prediction, target, mask, config):
    """Target-derived normalization and train-only floors are used only in the loss.

    Each eligible record/lead/band compares direction and relative amplitude.
    Ignore near-flat labels using frozen training-set thresholds, not validation QC.
    """
    if prediction.shape != target.shape or prediction.shape[1:] != (11, 6000):
        raise ValueError('Multiscale supervision requires [B,11,6000] full-120s slow curves')
    if mask.shape != prediction.shape[:2]:
        raise ValueError('Need per-record, per-lead target QC')
    floors = torch.as_tensor(config['std_floor'], device=prediction.device, dtype=prediction.dtype)
    if floors.shape != (3, 11) or not torch.isfinite(floors).all() or (floors <= 0).any():
        raise ValueError('Need positive frozen TRAIN-only [3,11] standard-deviation floors')
    amplitude_weight = float(config['shape_huber_weight'])
    if amplitude_weight < 0 or not torch.isfinite(torch.tensor(amplitude_weight)):
        raise ValueError('Shape Huber coefficient must be finite and nonnegative')
    enabled = config.get('enabled_bands', [0, 1, 2])
    if (not isinstance(enabled, (tuple, list)) or not enabled or
            any(type(i) is not int or i not in (0, 1, 2) for i in enabled) or len(set(enabled)) != len(enabled)):
        raise ValueError('Need unique enabled scale indices from 0,1,2')
    valid = mask.bool()
    p = torch.where(valid[..., None], prediction, 0.)
    y = torch.where(valid[..., None], target, 0.)
    losses, metrics = [], {}
    names = ('short_lt5s', 'middle_5to15s', 'long_gt15s')
    for index, (name, pb, yb) in enumerate(zip(names, scale_bands(p), scale_bands(y))):
        if index not in enabled:
            continue
        # Whole-record centering is ONLY part of the added shape loss.
        pc = pb - pb.mean(-1, keepdim=True)
        yc = yb - yb.mean(-1, keepdim=True)
        std = yc.square().mean(-1).sqrt().detach()
        eligible = valid & (std >= floors[index][None])
        normalizer = torch.maximum(std, floors[index][None])[..., None]
        a, b = pc / normalizer, yc / normalizer
        r = (a * b).mean(-1) / torch.sqrt(a.square().mean(-1) * b.square().mean(-1) + 1e-8)
        shape = F.huber_loss(a, b, reduction='none').mean(-1)
        count = eligible.sum().clamp_min(1)
        pcc = torch.where(eligible, 1 - r, 0.).sum() / count
        huber = torch.where(eligible, shape, 0.).sum() / count
        losses.append(pcc + amplitude_weight * huber)
        metrics[name + '_pcc'] = pcc
        metrics[name + '_shape_huber'] = huber
        metrics[name + '_eligible_fraction'] = eligible.sum() / valid.sum().clamp_min(1)
    return torch.stack(losses).mean(), metrics


def multiscale_settings(root, entries, weight):
    """Freeze quiet-label floors using TRAIN labels/QC only; validation is never read."""
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Need a positive finite multiscale weight')
    statistics = [[[] for _ in range(11)] for _ in range(3)]

    def average(x, width):
        padded = np.pad(x, ((0, 0), (width // 2, width // 2)), mode='edge')
        acc = np.concatenate((np.zeros((len(x), 1)), padded.cumsum(-1)), axis=-1)
        return (acc[:, width:] - acc[:, :-width]) / width

    for entry in entries:
        with np.load(root / entry['file'], allow_pickle=False) as record:
            y, mask = record['slow_target'].astype(np.float64), record['mask'].astype(bool)
            if y.shape != (11, 6000) or mask.shape != (12,) or not mask[0] or not np.isfinite(y).all():
                raise ValueError('Invalid TRAIN slow target for multiscale statistics')
            five, fifteen = average(y, 251), average(y, 751)
            for band, value in enumerate((y - five, five - fifteen, fifteen)):
                std = value.std(-1)
                for lead in range(11):
                    if mask[lead + 1]:
                        statistics[band][lead].append(float(std[lead]))
    medians = [[float(np.median(v)) if v else 0. for v in band] for band in statistics]
    floors = np.maximum(np.asarray(medians) * .1, 1e-4)
    return {'protocol': 'native50-full120-multiscale-shape-v1', 'weight': float(weight),
            'shape_huber_weight': .1, 'widths': [251, 751], 'floor_source': 'TRAIN-target-QC-only',
            'floor_rule': 'max(0.1 * TRAIN median band std per lead, 1e-4 normalized units)',
            'std_floor': floors.tolist(), 'train_median_std': medians,
            'train_eligible_label_counts': [[len(v) for v in band] for band in statistics],
            'bands': ['slow_minus_MA5s', 'MA5s_minus_MA15s', 'MA15s'],
            'centering': 'loss only, per band per complete120s; output and original losses unchanged'}
