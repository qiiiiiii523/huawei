"""Read-only M1 adapters over main's strict and joint-anchor datasets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import SupervisionMode
from .dataset import ECGDataConfig, JointAnchorDataset
from .d12_pretrain import StrictD12PretrainDataset
from .preprocessing import ECGPreprocessor, PreprocessingConfig

@dataclass(frozen=True)
class M1Item:
    anchor_model: torch.Tensor
    context_model: torch.Tensor | None
    target_model: torch.Tensor
    context_source_type: str | None
    context_lead_mask: torch.Tensor | None
    raw_target_uV: torch.Tensor
    raw_anchor_uV: torch.Tensor
    target_quality_mask: torch.Tensor
    meta: dict[str,Any]
    context_available: torch.Tensor | None = None

class M1PreparedDataset(Dataset[M1Item]):
    def __init__(self, samples:list[Any], preprocessor:ECGPreprocessor, stage:str, source_type:str|None=None):
        if stage not in {'P0_anchor_only','P1_joint_anchor'}: raise ValueError('invalid M1 stage')
        self.samples=samples; self.preprocessor=preprocessor; self.stage=stage; self.source_type=source_type
        if stage=='P0_anchor_only' and source_type is not None: raise ValueError('P0 has no context source')
        if stage=='P1_joint_anchor' and source_type is None: raise ValueError('P1 needs one context source')
        for sample in samples:
            sample.validate()
            if stage=='P0_anchor_only' and sample.split=='train' and sample.supervision_mode!=SupervisionMode.D12_I_PRETRAIN.value: raise ValueError('P0 train data must use d12_i_pretrain')
            if stage=='P1_joint_anchor' and sample.supervision_mode!=SupervisionMode.JOINT_ANCHOR_ADAPTATION.value: raise ValueError('P1 must use joint_anchor_adaptation')
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        sample=self.samples[index]
        target_raw=np.asarray(sample.Y_12lead,dtype=np.float32)
        target=self.preprocessor.transform_d12_target(target_raw).model_signal
        anchor_raw=target_raw[:1].copy(); anchor=self.preprocessor.transform_window(anchor_raw,'ecg_machine_i').model_signal
        context_model=None; context_mask=None; source=None; available=None
        if self.stage=='P1_joint_anchor':
            source=sample.context_source_type
            if source!=self.source_type: raise ValueError('context source routing mismatch')
            context_model=self.preprocessor.transform_window(np.asarray(sample.context_ecg,dtype=np.float32),source,sample.context_record_baseline_uV).model_signal
            time_mask=np.ones_like(context_model,dtype=bool) if sample.context_time_mask is None else np.asarray(sample.context_time_mask,dtype=bool)
            quality=np.asarray(sample.input_quality_mask,dtype=bool)
            # The original encoder has no masked temporal pooling. Incomplete
            # visible context is disabled; all validation windows are retained.
            available=torch.tensor(bool(time_mask.all() and quality.all()))
            context_model=np.where(time_mask & quality[:,None],context_model,0.).astype(np.float32)
            context_mask=np.asarray(sample.context_lead_mask,dtype=bool)
        target_quality_mask=np.asarray(sample.target_quality_mask,dtype=bool)
        if target_quality_mask.shape!=(12,): raise ValueError('M1 target quality mask must have 12 leads')
        return M1Item(torch.from_numpy(anchor), None if context_model is None else torch.from_numpy(context_model), torch.from_numpy(target), source, None if context_mask is None else torch.from_numpy(context_mask), torch.from_numpy(target_raw.copy()), torch.from_numpy(anchor_raw), torch.from_numpy(target_quality_mask), dict(sample.meta), available)

def m1_collate(items:list[M1Item])->dict[str,Any]:
    if not items: raise ValueError('empty M1 batch')
    context=[item.context_model for item in items]
    return {'anchor_model':torch.stack([x.anchor_model for x in items]),'context_model':None if context[0] is None else torch.stack(context), 'target_model':torch.stack([x.target_model for x in items]), 'context_source_type':items[0].context_source_type, 'context_lead_mask':None if items[0].context_lead_mask is None else torch.stack([x.context_lead_mask for x in items]), 'raw_target_uV':torch.stack([x.raw_target_uV for x in items]), 'raw_anchor_uV':torch.stack([x.raw_anchor_uV for x in items]), 'target_quality_mask':torch.stack([x.target_quality_mask for x in items]), 'meta':[x.meta for x in items], 'context_available':None if items[0].context_available is None else torch.stack([x.context_available for x in items])}

def _m1_joint_anchor_samples(config:ECGDataConfig, task_id:str, split:str, body_scale_variant:str='A_raw_window'):
    dataset=JointAnchorDataset(config,task_id,split,body_scale_variant)
    samples=list(dataset)
    for index,sample in enumerate(samples):
        row=dataset._rows[dataset._indices[index]]
        sample.meta.update({k:row[k] for k in ('pair_id','target_record_id','subject_id','start_sample_500hz')})
        if row.get('expected_window_count'): sample.meta['expected_window_count']=row['expected_window_count']
    if not samples:
        raise ValueError('M1 joint-anchor dataset is empty')
    return samples

def strict_m1_train(config):
    strict=StrictD12PretrainDataset(config,SupervisionMode.D12_I_PRETRAIN.value)
    with config.path('subject_split_csv').open(encoding='utf-8-sig',newline='') as handle:
        splits={r['subject_id']:r['split'] for r in csv.DictReader(handle)}
    if any(splits.get(r['subject_id'])!='train' for r in strict.rows):
        raise ValueError('M1 strict train index crosses subject split')
    return strict

def build_m1_datasets(config_path:str|Path,task_id:str,stage:str,source_type:str|None,preprocessor:ECGPreprocessor,body_scale_variant:str='A_raw_window'):
    config=ECGDataConfig.from_yaml(config_path)
    if stage=='P0_anchor_only':
        train_samples=list(strict_m1_train(config)); validation_samples=_m1_joint_anchor_samples(config,task_id,'validation',body_scale_variant)
        return M1PreparedDataset(train_samples,preprocessor,stage), M1PreparedDataset(validation_samples,preprocessor,stage)
    if task_id=='task1' and source_type!='watch_ecg': raise ValueError('task1 P1 context must be watch_ecg')
    if task_id=='task2' and source_type not in {'body_scale_d6','ecg_machine_d6'}: raise ValueError('task2 P1 needs one d6 source')
    train=[x for x in _m1_joint_anchor_samples(config,task_id,'train',body_scale_variant) if x.context_source_type==source_type]
    validation=[x for x in _m1_joint_anchor_samples(config,task_id,'validation',body_scale_variant) if x.context_source_type==source_type]
    if not train or not validation: raise ValueError(f'no samples for context source {source_type}')
    return M1PreparedDataset(train,preprocessor,stage,source_type), M1PreparedDataset(validation,preprocessor,stage,source_type)

def fit_m1_preprocessor(config_path:str|Path,task_id:str,stage:str,source_type:str|None,body_scale_variant:str='A_raw_window',scales_path:str|Path|None=None)->ECGPreprocessor:
    config=ECGDataConfig.from_yaml(config_path)
    pc=PreprocessingConfig.from_yaml(config.path('preprocessing_config'))
    if scales_path:
        frozen=ECGPreprocessor.load(pc,scales_path)
        scales={k:v.copy() for k,v in frozen.scale_uV_by_source.items()}
    else:
        strict=strict_m1_train(config); ranges=[]
        for row in strict.rows:
            raw=np.asarray(strict.targets[row['source_task_id']][int(row['source_array_index'])])
            if raw.shape!=(12,5000) or not np.isfinite(raw).all(): raise ValueError('Invalid M1 scale-fit waveform')
            ranges.append(np.percentile(raw,95,axis=-1)-np.percentile(raw,5,axis=-1))
        scale=np.maximum(np.median(np.stack(ranges),axis=0),pc.minimum_scale_uV).astype(np.float32)
        scales={'d12':scale,'ecg_machine_i':scale[:1].copy()}
    if stage=='P1_joint_anchor':
        if task_id=='task1': source_type='watch_ecg'
        if source_type not in {'watch_ecg','body_scale_d6','ecg_machine_d6'}: raise ValueError('invalid context source')
        if source_type not in scales:
            samples=[x for x in _m1_joint_anchor_samples(config,task_id,'train',body_scale_variant) if x.context_source_type==source_type]
            if not samples: raise ValueError('no train context samples for scale fitting')
            ranges=[np.percentile(x.context_ecg,95,axis=-1)-np.percentile(x.context_ecg,5,axis=-1) for x in samples]
            scales[source_type]=np.maximum(np.median(np.stack(ranges),axis=0),pc.minimum_scale_uV).astype(np.float32)
    elif stage!='P0_anchor_only': raise ValueError('invalid M1 stage')
    return ECGPreprocessor(pc,scales)
