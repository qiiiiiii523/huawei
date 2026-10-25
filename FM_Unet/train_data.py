"""Read immutable B7 raw-uV caches and prepare full-record MA501 slow targets."""
from collections import defaultdict, OrderedDict
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .fm_experiment import restrict50
from .targets import GENERATED, slow_record

META = ('numeric', 'sex', 'field_mask', 'age_topcoded')


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if value.get('protocol') != 'B7-raw-uV-records-v1' or value.get('sample_rate') != 500:
        raise ValueError('Need prepared B7 raw-uV 500Hz manifest')
    if not value.get('records'):
        raise ValueError('Empty manifest')
    seen, subjects, targets = set(), {}, {}
    for row in value['records']:
        if row['record_id'] in seen or row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Duplicate record or invalid split')
        seen.add(row['record_id'])
        subject = row['subject_id']
        owner = (subject, row['split'])
        if subjects.setdefault(subject, row['split']) != row['split']:
            raise ValueError('Patient leakage across splits')
        if targets.setdefault(row['target_record_id'], owner) != owner:
            raise ValueError('Target leakage across patients/splits')
        if type(row['input_reliable']) is not bool:
            raise ValueError('Expected boolean input_reliable')
        p = Path(row['path'])
        row['resolved_path'] = str(p if p.is_absolute() else path.parent / p)
    return value


def empty_metadata():
    return dict(numeric=np.zeros(3, np.float32), sex=np.array(2, np.int64),
                field_mask=np.zeros(4, bool), age_topcoded=np.zeros(1, np.float32))


def load_record(row):
    with np.load(row['resolved_path'], allow_pickle=False) as z:
        a, y, mask = z['anchor_uV'].astype('float32'), z['target_uV'].astype('float32'), z['lead_mask']
        present = set(META) & set(z.files)
        if present and present != set(META):
            raise ValueError('Incomplete metadata')
        metadata = {k: z[k].copy() for k in META} if present else (
            {k: np.asarray(row['metadata'][k]) for k in META} if 'metadata' in row else empty_metadata())
    if y.ndim != 2 or y.shape[0] != 12 or a.shape != (1, y.shape[-1]):
        raise ValueError('Expected anchor[1,T], target[12,T]')
    if mask.shape != (12,) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Bad QC mask')
    mask = mask.astype(bool)
    if not np.isfinite(a).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Nonfinite observed/supervised data')
    return a, y, mask, metadata


def native_target(target, mask, scales, rate):
    # Mask BEFORE filtering: invalid/NaN targets must never contaminate valid leads.
    safe = np.where(mask[:, None], target, 0.)
    slow = slow_record(safe[list(GENERATED)]) / np.asarray(scales)[list(GENERATED), None]
    length = slow.shape[-1] // 10 * 10
    fifty = slow[:, :length].reshape(7, -1, 10).mean(-1).astype('float32')
    with torch.no_grad():
        return restrict50(torch.from_numpy(fifty)[None], rate)[0].numpy()


class SlowDataset:
    def __init__(self, manifest, config, scales, cache_dir):
        self.rows = []
        self.memory = OrderedDict()
        self.rate = config['data']['native_hz']
        self.seconds = config['data']['input_seconds']
        self.native_width = self.seconds * self.rate
        self.raw_width = self.seconds * 500
        self.scales = scales
        self.groups = defaultdict(lambda: defaultdict(list))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        candidates = [r for r in manifest['records'] if r['split'] == 'train' and r['input_reliable']]
        self.record_hashes = {}
        for number, row in enumerate(candidates, 1):
            # SHA binds cache contents and exact resume, without decoding the full ECG.
            digest = file_sha(row['resolved_path'])
            self.record_hashes[row['record_id']] = digest
            if row.get('real_samples', self.raw_width) < self.raw_width:
                continue
            with np.load(row['resolved_path'], allow_pickle=False) as z:
                mask = z['lead_mask']
                if mask.shape != (12,) or not np.isin(mask, [0, 1]).all():
                    raise ValueError('Bad prepared quality mask')
                if not mask[0] or not mask[list(GENERATED)].any():
                    continue
                if 'real_samples' not in row and z['anchor_uV'].shape[-1] < self.raw_width:
                    continue
            self.groups[row['source']][row['subject_id']].append(len(self.rows))
            self.rows.append(row)
            if number % 2000 == 0:
                print(f'[preflight] scanned={number}/{len(candidates)} usable={len(self.rows)}', flush=True)
        if not self.rows:
            raise ValueError(f'No reliable real training records >= {self.seconds}s')
        print(f'[preflight] complete usable={len(self.rows)} patients={sum(len(v) for v in self.groups.values())}', flush=True)

    def entry(self, index):
        if index in self.memory:
            self.memory.move_to_end(index)
            return self.memory[index]
        row = self.rows[index]
        key = hashlib.sha256(json.dumps([self.record_hashes[row['record_id']], list(self.scales),
                                       self.rate, 'MA501-edge-FIR-v1',
                                       {k: np.asarray(v).tolist() for k, v in row.get('metadata', {}).items()}], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / (key+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                item = {k: z[k].copy() for k in z.files}
        else:
            a, y, mask, metadata = load_record(row)
            item = dict(anchor=(a/self.scales[0]).astype('float32'),
                        target=native_target(y, mask, self.scales, self.rate),
                        lead_mask=mask[list(GENERATED)], **metadata)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, **item)
            temporary.replace(path)
        self.memory[index] = item
        if len(self.memory) > 512:
            self.memory.popitem(last=False)
        return item

    def indices(self, count):
        sources = sorted(self.groups)
        selected = []
        for _ in range(count):
            patients = self.groups[sources[np.random.randint(len(sources))]]
            names = sorted(patients)
            records = patients[names[np.random.randint(len(names))]]
            selected.append(records[np.random.randint(len(records))])
        return selected

    def __getitem__(self, index):
        item = self.entry(index)
        maximum = min(item['target'].shape[-1] - self.native_width,
                      (item['anchor'].shape[-1] - self.raw_width) // (500 // self.rate))
        start = int(np.random.randint(maximum + 1))
        raw = start * (500 // self.rate)
        return {'anchor': torch.from_numpy(item['anchor'][:, raw:raw+self.raw_width].copy()),
                'target': torch.from_numpy(item['target'][:, start:start+self.native_width].copy()),
                'lead_mask': torch.from_numpy(item['lead_mask'].copy()),
                'metadata': {k: torch.as_tensor(item[k].copy()) for k in META}}
