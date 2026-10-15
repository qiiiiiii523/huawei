from __future__ import annotations

import numpy as np
import torch
from baselines.B5.config import ModelConfig
from baselines.B5.checkpoint import scales_digest
from . import PROTOCOL,SCORE_PROTOCOL,GROUPS
from .common import state_hash
from .model import make_model


def load(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('protocol')!=PROTOCOL or value.get('score_protocol')!=SCORE_PROTOCOL:
        raise ValueError('Use a B5-S50 run checkpoint, not the FM or SlowCNN initializer')
    if value.get('group') not in GROUPS: raise ValueError('Unknown matched group')
    expected='none' if value['group']=='control' else value['group']
    if value.get('slow_variant')!=expected or value['config'].get('slow_variant')!=expected:
        raise ValueError('Checkpoint slow variant/group mismatch')
    config=ModelConfig.from_dict(value['model_config'])
    if value['base_architecture_hash']!=config.fingerprint or state_hash(value['model'])!=value['model_ema_sha256']:
        raise ValueError('FM architecture/EMA checksum mismatch')
    if scales_digest(value['scales'])!=value['scales_sha256']: raise ValueError('Scale checksum mismatch')
    d12=np.asarray(value['scales']['d12'],np.float32)
    if d12.shape!=(12,) or not np.isfinite(d12).all() or np.any(d12<=0) or not np.array_equal(d12[:1],value['scales']['ecg_machine_i']):
        raise ValueError('Invalid shared I/target scales')
    if (state_hash(value['slow_state'])!=value['slow_state_sha256'] or
            value['config'].get('slow_state_sha256')!=value['slow_state_sha256']): raise ValueError('Frozen SlowCNN checksum mismatch')
    return value


def model_from(value,device):
    base={k:v for k,v in value['model'].items() if not k.startswith('slow_adapter.')}
    model=make_model(ModelConfig.from_dict(value['model_config']),value['group'],base)
    model.load_state_dict(value['model'],strict=True)
    return model.to(device).eval()
