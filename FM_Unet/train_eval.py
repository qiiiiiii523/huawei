"""Final free-generation validation; never uses targets as model conditions."""
from collections import defaultdict
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .targets import GENERATED, missing11, slow_record
from .train_data import load_record

FAST_STATE_SHA = '990e92e30f5255046c6aa73ab55f1ef4e22fc814a05f4fc92c14457bd6d35651'


class FrozenFast:
    def __init__(self, code, checkpoint, device):
        # Reuse the user's existing B5 code/weights without copying or updating either.
        sys.path.insert(0, str(Path(code).resolve()))
        from baselines.B5.model import B5UNet
        from baselines.B5.config import ModelConfig
        from baselines.B5.flow import keyed_noise, integrate
        self.keyed_noise, self.integrate = keyed_noise, integrate
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        fm = payload['fm']
        h = hashlib.sha256()
        for key, value in sorted(fm['ema'].items()):
            v = value.detach().cpu().contiguous()
            h.update(key.encode()); h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
        if h.hexdigest() != FAST_STATE_SHA:
            raise ValueError('Expected frozen FM_P_best45 EMA; refusing a different fast model')
        self.model = B5UNet(ModelConfig.from_dict(fm['model_config']))
        self.model.load_state_dict(fm['ema'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.scales = np.asarray(fm['scales']['d12'], np.float32)
        self.device = device

    @torch.no_grad()
    def predict(self, anchor, metadata, record_key):
        if anchor.shape != (1, 60000):
            raise ValueError('Fast validation needs full real 120 seconds')
        result = []
        for start in range(0, 60000, 5000):
            a = torch.as_tensor(anchor[:, start:start+5000]/self.scales[0], device=self.device)[None]
            fields = {k: torch.as_tensor(v, device=self.device)[None] for k, v in metadata.items()}
            condition = self.model.encode_conditions(a, fields['numeric'], fields['sex'],
                                                     fields['field_mask'], fields['age_topcoded'])
            total = torch.zeros(1, 11, 5000, device=self.device)
            for member in range(16):
                z = self.keyed_noise([record_key+':'+str(start)], 5000, 42, member//2, self.device)
                if member % 2:
                    z = -z
                total += self.integrate(self.model, condition, z, steps=16, solver='heun')
            result.append(total[0].cpu().numpy()/16 * self.scales[1:, None])
        full = np.concatenate(result, -1)
        return full.astype(np.float64) - slow_record(full)


@torch.no_grad()
def sample_window(model, anchor, metadata, scales, noise_std, key, validation):
    device = next(model.parameters()).device
    no = model.output_seconds * model.rate
    a = torch.as_tensor(anchor/scales[0], dtype=torch.float32, device=device)[None]
    fields = {k: torch.as_tensor(v, device=device)[None] for k, v in metadata.items()}
    seed = int.from_bytes(hashlib.sha256(('huawei-series1|42|'+key).encode()).digest()[:8], 'little') % (2**63-1)
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(validation['samples']//2, 7, no, generator=generator) * torch.as_tensor(noise_std)[None, :, None]
    starts = torch.stack((z, -z), 1).flatten(0, 1)
    total = torch.zeros(7, no, dtype=torch.float64, device=device)
    for offset in range(0, len(starts), validation['sample_batch_size']):
        state = starts[offset:offset+validation['sample_batch_size']].to(device)
        count = len(state)
        aa = a.expand(count, -1, -1)
        mm = {k: v.expand(count, *v.shape[1:]) for k, v in fields.items()}
        for step in range(validation['steps']):
            dt = 1/validation['steps']
            t = a.new_full((count,), step/validation['steps'])
            v = model(state, t, aa, mm)
            w = model(state+dt*v, t+dt, aa, mm)
            state = state + dt*.5*(v+w)
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite generated slow state')
        total += state.double().sum(0)
    native = (total/validation['samples']).cpu()
    dense = F.interpolate(native[None], size=anchor.shape[-1], mode='linear', align_corners=False)[0].numpy()
    return missing11(dense*np.asarray(scales)[list(GENERATED), None], anchor)


def pair_metrics(prediction, target, mask):
    result = {}
    for j in range(11):
        if not mask[j+1]:
            continue
        p, y = prediction[j].astype('float64'), target[j+1].astype('float64')
        pc, yc = p-p.mean(), y-y.mean()
        denominator = np.linalg.norm(pc)*np.linalg.norm(yc)
        result[str(j+1)] = {'r': float(np.dot(pc, yc)/denominator) if denominator > 0 else None,
                            'rmse_uV': float(np.sqrt(np.mean((p-y)**2)))}
    return result


def aggregate(rows, stage):
    values = defaultdict(list)
    for row in rows:
        for lead, metrics in row['leads'].items():
            if metrics['r'] is not None:
                values[(row['group'], row['target_record_id'], lead)].append(metrics)
    grouped = defaultdict(lambda: defaultdict(list))
    for (group, target, lead), pairs in values.items():
        grouped[group][lead].append({k: float(np.mean([p[k] for p in pairs])) for k in ('r', 'rmse_uV')})
    if not grouped:
        raise ValueError('No defined QC-eligible validation correlation')
    summaries = {}
    for group, leads in grouped.items():
        if set(leads) != {str(j) for j in range(1, 12)}:
            raise ValueError(f'Incomplete eligible leads in validation group {group}')
        summaries[group] = {k: float(np.mean([np.mean([m[k] for m in records]) for records in leads.values()]))
                            for k in ('r', 'rmse_uV')}
    if stage == 'huawei120':
        if set(summaries) != {'task1', 'machine', 'body'}:
            raise ValueError('Huawei validation must include task1/machine/body')
        summary = {k: .5*summaries['task1'][k]+.25*summaries['machine'][k]+.25*summaries['body'][k]
                   for k in ('r', 'rmse_uV')}
    else:
        summary = {k: float(np.mean([g[k] for g in summaries.values()])) for k in ('r', 'rmse_uV')}
    return dict(**summary, groups=summaries, pairs=len(rows),
                score_scope='fast_plus_slow_full120_missing11' if stage == 'huawei120' else 'slow_only_real_window_missing11')


def evaluate(model, manifest, config, scales, noise_std, *, fast=None, fast_cache=None):
    model.eval()
    rows, width = [], model.input_seconds*500
    eligible = [r for r in manifest['records'] if r['split'] == 'validation' and r['input_reliable']]
    for i, row in enumerate(eligible, 1):
        anchor, target, mask, metadata = load_record(row)
        if not mask[0] or anchor.shape[-1] < width:
            continue
        if config['stage'] == 'huawei120' and anchor.shape[-1] != width:
            raise ValueError('Huawei validation needs exact real 120 seconds')
        # Public records use a deterministic central real window; no synthetic padding.
        start = (anchor.shape[-1]-width)//2
        reference_slow = slow_record(np.where(mask[:, None], target, 0.))[:, start:start+width] if config['stage'] != 'huawei120' else None
        anchor, target = anchor[:, start:start+width], target[:, start:start+width]
        slow = sample_window(model, anchor, metadata, scales, noise_std, row['record_id'], config['validation'])
        if config['stage'] == 'huawei120':
            if fast is None:
                raise ValueError('Huawei final validation requires the pinned frozen fast branch')
            fingerprint = hashlib.sha256(anchor.tobytes()+json_bytes(metadata)+row['record_id'].encode()+FAST_STATE_SHA.encode()).hexdigest()
            path = Path(fast_cache) / (fingerprint+'.npy')
            if path.exists():
                quick = np.load(path, allow_pickle=False)
            else:
                quick = fast.predict(anchor, metadata, row['record_id'])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, quick, allow_pickle=False)
            prediction, reference = quick+slow, target
        else:
            prediction, reference = slow, reference_slow
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite evaluation prediction')
        rows.append(dict(record_id=row['record_id'], target_record_id=row['target_record_id'],
                         group=row['group'], leads=pair_metrics(prediction, reference, mask)))
        if i == 1 or i % 100 == 0 or i == len(eligible):
            print(f'[final-validation] scored={len(rows)}/{len(eligible)}', flush=True)
    return aggregate(rows, config['stage']), rows


def json_bytes(metadata):
    return b''.join(k.encode()+np.asarray(v).tobytes() for k, v in sorted(metadata.items()))
