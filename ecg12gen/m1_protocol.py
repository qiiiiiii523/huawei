"""Reject checkpoints trained using the old centered-target protocol."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .preprocessing import ECGPreprocessor, PreprocessingConfig

def protocol_metadata():
    return {'m1_run_protocol':'M1-main-v3',
            'preprocessing_protocol':'v3-raw-target-record-context',
            'evaluation_aggregation':'record_macro_after_chronological_window_stitch'}

def check_checkpoint_protocol(checkpoint):
    for key,value in protocol_metadata().items():
        if checkpoint.get(key)!=value:
            raise ValueError(f'Incompatible M1 checkpoint protocol: {key}; retrain the legacy centered-target model')

def checkpoint_preprocessor(checkpoint, config, scales_path):
    check_checkpoint_protocol(checkpoint)
    pre=ECGPreprocessor.load(PreprocessingConfig.from_yaml(config.path('preprocessing_config')),scales_path)
    expected=np.asarray(checkpoint.get('target_d12_scale_uV',[]),dtype=np.float32)
    if expected.shape!=(12,) or not np.array_equal(expected,pre.scale_uV_by_source['d12']):
        raise ValueError('M1 checkpoint and frozen target scales differ')
    source=checkpoint.get('context_source_type')
    if checkpoint.get('stage')=='P1_joint_anchor':
        stored=checkpoint.get('context_scale_uV')
        if stored is None or source not in pre.scale_uV_by_source or not np.array_equal(np.asarray(stored,dtype=np.float32),pre.scale_uV_by_source[source]):
            raise ValueError('M1 checkpoint and frozen context scales differ')
    return pre
