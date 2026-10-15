"""Existing B5 supervision plus target-free, frozen slow-condition lookup."""
from __future__ import annotations

import numpy as np
import torch

from . import INPUTS
from .slow import slow_window


def group_records(rows):
    groups={}
    for i,row in enumerate(rows): groups.setdefault(row['target_record_id'],[]).append(i)
    for key,indices in groups.items():
        indices.sort(key=lambda i:int(rows[i]['start_sample_500hz']))
        if [int(rows[i]['start_sample_500hz']) for i in indices]!=list(range(0,60000,5000)):
            raise ValueError('Requires 12 chronological windows for '+key)
        if len({rows[i]['subject_id'] for i in indices})!=1: raise ValueError('Record crosses subjects')
    return groups


def verify_inputs(base,record_metadata,arrays):
    """Match current test-visible I/demographics to the already audited prepared inputs."""
    grouped=group_records(base.rows)
    if set(grouped)!={r['target_record_id'] for r in record_metadata}:
        raise ValueError('Prepared/current record population changed')
    for i,meta in enumerate(record_metadata):
        rows=[base[j] for j in grouped[meta['target_record_id']]]
        anchor=np.concatenate([row['anchor'] for row in rows],axis=-1)
        native=anchor.reshape(1,6000,10).mean(-1)
        if not np.allclose(native,arrays['anchor'][i],atol=2e-6,rtol=2e-6):
            raise ValueError('Current I differs from prepared I: '+meta['target_record_id'])
        for key in INPUTS[1:]:
            if any(not np.array_equal(row[key],arrays[key][i]) for row in rows):
                raise ValueError('Demographics changed: '+meta['target_record_id']+'/'+key)


class ConditionedDataset:
    def __init__(self,base,native,metadata):
        self.base,self.native=base,native
        self.lookup={r['target_record_id']:i for i,r in enumerate(metadata)}
        self.rows=base.rows
        for row in self.rows:
            if row['target_record_id'] not in self.lookup: raise ValueError('Missing slow input record')
        if native.shape!=(len(metadata),11,6000): raise ValueError('Cached slow array shape mismatch')

    def __len__(self): return len(self.base)

    def __getitem__(self,index):
        row=dict(self.base[index])
        meta=self.rows[index]
        row['slow']=slow_window(self.native[self.lookup[meta['target_record_id']]],int(meta['start_sample_500hz']))
        return row


def collate(rows):
    result={key:torch.from_numpy(np.stack([np.asarray(row[key]) for row in rows]).copy()) for key in (*INPUTS,'slow')}
    for key in ('target','quality_mask'):
        if key in rows[0]: result[key]=torch.from_numpy(np.stack([row[key] for row in rows]).copy())
    result['keys']=[row['key'] for row in rows]
    result['evaluation_metadata']=[row['evaluation_metadata'] for row in rows]
    return result


def to_device(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
