"""Task1 C1 pseudo-pair reader and quality-weighted loss."""
from pathlib import Path
import csv
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from ecg12gen.dataset import UnifiedECGDataset
from ecg12gen.losses import (
    masked_huber_loss, masked_pcc_loss,
    observed_consistency_loss, physiology_constraint_loss,
)


class B0PseudoDataset(Dataset):
    def __init__(self, config, preprocessor, directory):
        self.preprocessor = preprocessor
        directory = Path(directory)
        self.metadata_path = directory / 'task1_rpeak_train_window_metadata.csv'
        with self.metadata_path.open(encoding='utf-8-sig', newline='') as f:
            self.rows = list(csv.DictReader(f))
        self.rows.sort(key=lambda r: int(r['array_index']))
        self.inputs = np.load(directory / 'task1_rpeak_train_input.npy', mmap_mode='r', allow_pickle=False)
        self.targets = np.load(directory / 'task1_rpeak_train_target.npy', mmap_mode='r', allow_pickle=False)
        n = len(self.rows)
        if n == 0 or self.inputs.shape != (n, 1, 5000) or self.targets.shape != (n, 12, 5000):
            raise ValueError('Pseudo arrays must match nonempty metadata: [N,1,5000], [N,12,5000]')
        if [int(r['array_index']) for r in self.rows] != list(range(n)):
            raise ValueError('Pseudo array_index must be unique and contiguous')
        if len({r['source_window_id'] for r in self.rows}) != n:
            raise ValueError('Duplicate pseudo source windows')
        # Public reader enforces subject split, paired/usable status and training policy.
        original = UnifiedECGDataset(config, task_id='task1', split='train')
        wanted = {r['source_window_id'] for r in self.rows}
        source = {}
        for i in range(len(original)):
            sample = original[i]
            wid = str(sample.meta['window_id'])
            if wid in wanted:
                if wid in source:
                    raise ValueError('Ambiguous original window ID')
                source[wid] = i
        self.devices = []
        for i, row in enumerate(self.rows):
            if (row['split'] != 'train' or row['accepted'].lower() != 'true'
                or row['physical_sync'].lower() != 'false'
                or row['alignment_mode'] != 'pseudo_rpeak_monotonic_warp'):
                raise ValueError(f'Invalid pseudo supervision metadata at row {i}')
            q = float(row['alignment_quality_score'])
            if not np.isfinite(q) or not 0 < q <= 1:
                raise ValueError('Quality must be finite and in (0,1]')
            if row['source_window_id'] not in source:
                raise ValueError('Pseudo source is not an eligible original training window')
            sample = original[source[row['source_window_id']]]
            if str(sample.meta['subject_id']) != row['subject_id']:
                raise ValueError('Pseudo subject does not match original train subject')
            if not np.array_equal(self.inputs[i], sample.X_ecg):
                raise ValueError('Pseudo input differs from its original watch window')
            if not np.isfinite(self.inputs[i]).all() or not np.isfinite(self.targets[i]).all():
                raise ValueError('Non-finite pseudo waveform')
            self.devices.append(str(sample.meta['device_type']))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        transformed = self.preprocessor.transform_window(
            self.inputs[index], source_type=self.devices[index])
        target = self.preprocessor.transform_d12_target(self.targets[index]).model_signal
        observed = np.zeros_like(target)
        observed[:1] = (transformed.model_signal * transformed.scale_uV[:, None]
                        / self.preprocessor.scale_uV_by_source['d12'][:1, None])
        mask = np.zeros(12, dtype=bool)
        mask[0] = True
        return {
            'inputs': torch.from_numpy(np.array(transformed.model_signal, dtype=np.float32, copy=True)),
            'target': torch.from_numpy(np.array(target, dtype=np.float32, copy=True)),
            'observed_input': torch.from_numpy(np.array(observed, dtype=np.float32, copy=True)),
            'lead_mask': torch.from_numpy(mask),
            'quality': np.float32(row['alignment_quality_score']),
            'alignment_mode': row['alignment_mode'], 'physical_sync': False,
            'split': 'train', 'subject_id': row['subject_id'],
        }


class B0PseudoLoss:
    """Sum(q_i * per-window component_i) / sum(q_i), then profile weights.

    Applies quality weighting to all four components. No raw weak-pair target
    is accepted: the caller must supply accepted, warped pseudo training data.
    """
    def __init__(self, config_path):
        with Path(config_path).open(encoding='utf-8') as f:
            self.weights = yaml.safe_load(f)['rpeak_pseudo']
        if self.weights['quality_weight'] != 'alignment_quality_score':
            raise ValueError('Unsupported pseudo quality definition')
        for key in ('huber_missing', 'pcc_missing', 'physiology', 'observed_consistency'):
            w = float(self.weights[key])
            if not np.isfinite(w) or w < 0:
                raise ValueError('Invalid pseudo loss weight')

    def __call__(self, prediction, batch, scale):
        if (not all(s == 'train' for s in batch['split'])
            or not all(a == 'pseudo_rpeak_monotonic_warp' for a in batch['alignment_mode'])
            or batch['physical_sync'].any()):
            raise ValueError('Pseudo loss requires train-only pseudo alignment, not physical sync')
        q = torch.as_tensor(batch['quality'], dtype=prediction.dtype, device=prediction.device).detach()
        if q.shape != (len(prediction),) or not torch.isfinite(q).all() or (q <= 0).any() or (q > 1).any():
            raise ValueError('Invalid per-window quality weights')
        target = batch['target'].detach()
        mask = batch['lead_mask']
        if (prediction.shape != target.shape or prediction.ndim != 3
            or prediction.shape[1] != 12 or mask.shape != prediction.shape[:2]
            or mask.dtype != torch.bool or not mask[:, 0].all() or mask[:, 1:].any()):
            raise ValueError('Expected Task1 pseudo prediction/target and I-only mask')
        terms = {k: [] for k in ('huber_missing', 'pcc_missing', 'physiology', 'observed_consistency')}
        for i in range(len(prediction)):
            p, t, m = prediction[i:i+1], target[i:i+1], mask[i:i+1]
            terms['huber_missing'].append(masked_huber_loss(p, t, ~m))
            terms['pcc_missing'].append(masked_pcc_loss(p, t, ~m))
            terms['physiology'].append(physiology_constraint_loss(p * scale[None, :, None] / 1000.0))
            terms['observed_consistency'].append(observed_consistency_loss(
                p, batch['observed_input'][i:i+1].detach(), m))
        components = {k: (torch.stack(v) * q).sum() / q.sum() for k, v in terms.items()}
        total = sum(float(self.weights[k]) * v for k, v in components.items())
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite pseudo loss')
        logs = {k: float(v.detach()) for k, v in components.items()}
        logs['total'] = float(total.detach())
        return total, logs
