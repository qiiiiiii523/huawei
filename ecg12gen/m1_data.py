"""Read-only M1 adapters over main's strict and joint-anchor datasets."""
from __future__ import annotations
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import JointAnchorSample, SupervisionMode, WINDOW_SAMPLES, canonical_lead_mask
from .dataset import ECGDataConfig
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
    meta: dict[str,Any]

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
        context_model=None; context_mask=None; source=None
        if self.stage=='P1_joint_anchor':
            source=sample.context_source_type
            if source!=self.source_type: raise ValueError('context source routing mismatch')
            context_model=self.preprocessor.transform_window(np.asarray(sample.context_ecg,dtype=np.float32),source).model_signal
            context_mask=np.asarray(sample.context_lead_mask,dtype=bool)
        return M1Item(torch.from_numpy(anchor), None if context_model is None else torch.from_numpy(context_model), torch.from_numpy(target), source, None if context_mask is None else torch.from_numpy(context_mask), torch.from_numpy(target_raw.copy()), torch.from_numpy(anchor_raw), dict(sample.meta))

def m1_collate(items:list[M1Item])->dict[str,Any]:
    if not items: raise ValueError('empty M1 batch')
    context=[item.context_model for item in items]
    return {'anchor_model':torch.stack([x.anchor_model for x in items]),'context_model':None if context[0] is None else torch.stack(context), 'target_model':torch.stack([x.target_model for x in items]), 'context_source_type':items[0].context_source_type, 'context_lead_mask':None if items[0].context_lead_mask is None else torch.stack([x.context_lead_mask for x in items]), 'raw_target_uV':torch.stack([x.raw_target_uV for x in items]), 'raw_anchor_uV':torch.stack([x.raw_anchor_uV for x in items]), 'meta':[x.meta for x in items]}

def _read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def _m1_joint_anchor_samples(config:ECGDataConfig, task_id:str, split:str, body_scale_variant:str='A_raw_window'):
    if task_id not in {'task1','task2'} or split not in {'train','validation'}:
        raise ValueError('invalid M1 joint-anchor task or split')
    task_dir=config.path(f'{task_id}_output')
    targets=np.load(task_dir/f'{task_id}_{split}_target.npy',mmap_mode='r')
    if targets.ndim!=3 or targets.shape[1:]!=(12,WINDOW_SAMPLES):
        raise ValueError(f'invalid M1 target shape: {targets.shape}')
    subject_split={row['subject_id']:row['split'] for row in _read_csv(config.path('subject_split_csv'))}
    pairs={row['pair_id']:row for row in _read_csv(config.path(f'{task_id}_pair_manifest_csv'))}
    if task_id=='task2' and body_scale_variant=='B_detrend_0p2Hz_then_window':
        inputs=np.load(config.path('task2_body_scale_b_train_input' if split=='train' else 'task2_body_scale_b_validation_input'),mmap_mode='r')
        rows=[row for row in _read_csv(config.path('task2_body_scale_b_metadata')) if row['split']==split]
        indexed=[(row,int(row['local_array_index']),int(row['canonical_array_index'])) for row in rows]
    else:
        inputs=np.load(task_dir/f'{task_id}_{split}_input.npy',mmap_mode='r')
        rows=[row for row in _read_csv(task_dir/f'{task_id}_window_metadata.csv') if row['split']==split]
        indexed=[(row,int(row['array_index']),int(row['array_index'])) for row in rows]
    expected_channels=1 if task_id=='task1' else 6
    if inputs.ndim!=3 or inputs.shape[1:]!=(expected_channels,WINDOW_SAMPLES):
        raise ValueError(f'invalid M1 input shape: {inputs.shape}')
    allowed={'watch_ecg'} if task_id=='task1' else {'ecg_machine_d6','body_scale_d6'}
    samples=[]
    for row,input_index,target_index in indexed:
        pair=pairs.get(row['pair_id'])
        source_type=str(row.get('input_type') or (pair or {}).get('input_type') or '')
        if pair is None or source_type not in allowed:
            continue
        if subject_split.get(row['subject_id'])!=split:
            raise ValueError('invalid subject split')
        if pair.get('pair_status')!='paired' or pair.get('input_quality_status')!='usable' or pair.get('target_quality_status')!='usable':
            continue
        if input_index>=len(inputs) or target_index>=len(targets):
            raise ValueError('M1 joint-anchor array index out of bounds')
        context=np.asarray(inputs[input_index],dtype=np.float32)
        target=np.asarray(targets[target_index],dtype=np.float32)
        context_mask=np.ones(6 if task_id=='task2' else 1,dtype=bool)
        subject_id,window_id=str(row['subject_id']),str(row['window_id'])
        target_record_id=str(row.get('target_record_id') or pair.get('target_record_id') or window_id)
        meta={'subject_id':subject_id,'window_id':window_id,'pair_id':row['pair_id'],'device_type':source_type,'target_record_id':target_record_id,'input_processing_variant':body_scale_variant,'anchor_construction':'simulated_from_target_i_for_test_available_input','anchor_target_record_id':target_record_id,'anchor_window_id':window_id,'context_target_relation':'same_subject_cross_time','anchor_target_relation':'same_record_same_window','pointwise_loss_allowed':True,'context_target_pointwise_loss':False}
        sample=JointAnchorSample(context_ecg=context,context_source_type=source_type,anchor_i_ecg=target[:1].copy(),anchor_source_type='ecg_machine_i',Y_12lead=target,anchor_lead_mask=canonical_lead_mask(1),context_lead_mask=context_mask,task_id=task_id,split=split,subject_id=subject_id,pair_id=row['pair_id'],target_record_id=target_record_id,window_id=window_id,meta=meta,input_type=source_type)
        sample.validate()
        samples.append(sample)
    if not samples:
        raise ValueError('M1 joint-anchor dataset is empty')
    return samples

def _raw_train_targets(config:ECGDataConfig,task_id:str): return np.load(config.path(f'{task_id}_output')/f'{task_id}_train_target.npy',mmap_mode='r')

def build_m1_datasets(config_path:str|Path,task_id:str,stage:str,source_type:str|None,preprocessor:ECGPreprocessor,body_scale_variant:str='A_raw_window'):
    config=ECGDataConfig.from_yaml(config_path)
    if stage=='P0_anchor_only':
        train_samples=list(StrictD12PretrainDataset(config,SupervisionMode.D12_I_PRETRAIN.value)); validation_samples=_m1_joint_anchor_samples(config,task_id,'validation',body_scale_variant)
        return M1PreparedDataset(train_samples,preprocessor,stage), M1PreparedDataset(validation_samples,preprocessor,stage)
    if task_id=='task1' and source_type!='watch_ecg': raise ValueError('task1 P1 context must be watch_ecg')
    if task_id=='task2' and source_type not in {'body_scale_d6','ecg_machine_d6'}: raise ValueError('task2 P1 needs one d6 source')
    train=[x for x in _m1_joint_anchor_samples(config,task_id,'train',body_scale_variant) if x.context_source_type==source_type]
    validation=[x for x in _m1_joint_anchor_samples(config,task_id,'validation',body_scale_variant) if x.context_source_type==source_type]
    if not train or not validation: raise ValueError(f'no samples for context source {source_type}')
    return M1PreparedDataset(train,preprocessor,stage,source_type), M1PreparedDataset(validation,preprocessor,stage,source_type)

def fit_m1_preprocessor(config_path:str|Path,task_id:str,stage:str,source_type:str|None,body_scale_variant:str='A_raw_window')->ECGPreprocessor:
    config=ECGDataConfig.from_yaml(config_path); pc=PreprocessingConfig.from_yaml(config.repository_root/'configs'/'preprocessing.yaml')
    signals={'d12':np.asarray(_raw_train_targets(config,task_id),dtype=np.float32)}
    if stage=='P1_joint_anchor':
        if task_id=='task1':
            source_type='watch_ecg'
        if source_type not in {'watch_ecg','body_scale_d6','ecg_machine_d6'}: raise ValueError('invalid context source')
        samples=[x for x in _m1_joint_anchor_samples(config,task_id,'train',body_scale_variant) if x.context_source_type==source_type]
        if not samples: raise ValueError('no train context samples for scale fitting')
        signals[source_type]=np.stack([np.asarray(x.context_ecg,dtype=np.float32) for x in samples])
    return ECGPreprocessor.fit(pc,signals)
