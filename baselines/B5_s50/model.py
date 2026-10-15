"""The only FM addition is a zero-projected 11-lead slow condition at the bottleneck."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from baselines.B5.model import B5UNet
from baselines.B5.config import ModelConfig
from . import GROUPS


class B5S50UNet(B5UNet):
    def __init__(self, config=ModelConfig()):
        super().__init__(config)
        self.slow_adapter = nn.Sequential(nn.Conv1d(11,32,3,padding=1), nn.SiLU(),
                                          nn.Conv1d(32,config.channels[2],1))
        nn.init.zeros_(self.slow_adapter[-1].weight)
        nn.init.zeros_(self.slow_adapter[-1].bias)

    def encode_conditions(self, anchor, numeric, sex, field_mask, age_topcoded, slow):
        if slow.shape != (len(anchor),11,anchor.shape[-1]) or not torch.isfinite(slow).all():
            raise ValueError('Slow condition must be finite [B,11,T], in the FM target scale')
        condition = super().encode_conditions(anchor,numeric,sex,field_mask,age_topcoded)
        reduced = F.interpolate(slow,size=condition.anchor_features[-1].shape[-1],mode='linear',align_corners=False)
        condition.slow_bottleneck = self.slow_adapter(reduced)
        return condition

    def forward(self,state,time,anchor,numeric,sex,field_mask,age_topcoded,slow):
        condition = self.encode_conditions(anchor,numeric,sex,field_mask,age_topcoded,slow)
        return self.velocity(state,time,condition),condition.anchor_prediction


class BoundModel:
    """Explicit per-batch binding for the unchanged original FM loss/ODE sampler."""
    def __init__(self, model, slow=None):
        self.model,self.slow=model,slow
        if isinstance(model,B5S50UNet) and slow is None:
            raise ValueError('B5-S50 requires a predicted slow condition')

    @property
    def training(self): return self.model.training

    def __call__(self,*args):
        return self.model(*args,self.slow) if isinstance(self.model,B5S50UNet) else self.model(*args)

    def encode_conditions(self,*args):
        return self.model.encode_conditions(*args,self.slow) if isinstance(self.model,B5S50UNet) else self.model.encode_conditions(*args)

    def velocity(self,*args): return self.model.velocity(*args)


def make_model(config,group,source_ema):
    if group not in GROUPS: raise ValueError('Group must be control, slow or slow_f')
    model = B5UNet(config) if group=='control' else B5S50UNet(config)
    incompatible = model.load_state_dict(source_ema,strict=False)
    allowed = [key for key in model.state_dict() if key.startswith('slow_adapter.')]
    if sorted(incompatible.missing_keys)!=sorted(allowed) or incompatible.unexpected_keys:
        raise ValueError('Initializer differs from the original B5 model; only slow_adapter may be new')
    return model
