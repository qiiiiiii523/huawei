"""Frozen SlowCNN definition copied from B5-exp-S50 3d503ce; original state keys retained."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_slow(checkpoint,device,variant='slow'):
    if checkpoint.get('protocol')!='B5-slow-probe-v1' or checkpoint['config'].get('context_seconds')!=50:
        raise ValueError('Requires an S50 SlowCNN checkpoint, not an FM model or S30 model')
    expected={'slow':'C','slow_f':'F'}
    if variant not in expected or checkpoint['config'].get('group')!=expected[variant] or checkpoint['config'].get('model_channels',48)!=48:
        raise ValueError('Slow source/group mismatch: slow requires C, slow_f requires F, both 48 channels')
    model=SlowCNN()
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).requires_grad_(False).eval()


def validate_slow_pair(checkpoints):
    """F was applied to the pretrained slow CNN, never to the new FM loss."""
    a,b=(checkpoints[key]['config'] for key in ('slow','slow_f'))
    matched=('prepared_sha256','context_seconds','model_channels','init_checkpoint_sha256',
        'initial_weights_sha256','seed','epochs','batch_records','validate_every',
        'learning_rate','weight_decay','huber_weight','pcc_weight')
    for key in matched:
        if key not in a or key not in b or a[key]!=b[key]:
            raise ValueError('S50/S50-F pretraining pair differs: '+key)
    if a['context_seconds']!=50 or a['epochs']!=20:
        raise ValueError('Requires the matched 50s/20-epoch slow experiment')
    if a.get('group')!='C' or b.get('group')!='F': raise ValueError('Expected original-loss C and residual-supervised F')
    if float(a.get('mean_weight',0))!=0 or float(b.get('mean_weight',0))!=0:
        raise ValueError('This comparison does not include the M mean loss')
    if float(a.get('residual_weight',0))!=0 or not np.isfinite(float(b.get('residual_weight',0))) or float(b.get('residual_weight',0))<=0:
        raise ValueError('Requires no F in S50 and a positive fixed F weight in S50-F')


@torch.no_grad()
def predict_native(model,inputs):
    """Only I and demographics. Full normalized 120s I at 50Hz -> [B,11,6000]."""
    if model.training: raise ValueError('The frozen S50 network must stay in eval mode')
    anchor=inputs['anchor']
    if anchor.shape[1:]!=(1,6000) or not torch.isfinite(anchor).all():
        raise ValueError('S50 needs full 120-second I at 50Hz')
    padded=torch.cat((anchor[...,:1].expand(-1,-1,1000),anchor,anchor[...,-1:].expand(-1,-1,1000)),dim=-1)
    windows=padded.unfold(-1,2500,500).permute(0,2,1,3).reshape(-1,1,2500)
    meta={k:inputs[k].repeat_interleave(12,dim=0) for k in ('numeric','sex','field_mask','age_topcoded')}
    output=model(windows,**meta)[...,1000:1500]
    result=output.reshape(len(anchor),12,11,500).permute(0,2,1,3).reshape(-1,11,6000)
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite predicted slow condition')
    return result


def slow_window(native,start,length=5000):
    if native.shape!=(11,6000) or start<0 or start+length>60000:
        raise ValueError('Invalid complete-record slow curve/window')
    # Interpolate using full-record block centers, never independently per window.
    centers=np.arange(6000)*10+4.5
    return np.stack([np.interp(np.arange(start,start+length),centers,row) for row in native]).astype(np.float32)

class SameConv(nn.Module):
    def __init__(self, incoming, outgoing, dilation=1):
        super().__init__()
        self.radius = 2 * dilation
        self.conv = nn.Conv1d(incoming, outgoing, 5, dilation=dilation)

    def forward(self, x):
        # Explicit repeats have deterministic backward; no reflection across patients.
        r = self.radius
        return self.conv(torch.cat((x[..., :1].expand(-1, -1, r), x,
                                    x[..., -1:].expand(-1, -1, r)), dim=-1))


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(channels)  # Channel-only, not per-window time centering.
        self.one, self.two = SameConv(channels, channels, dilation), SameConv(channels, channels, dilation)
        self.film = nn.Linear(32, 2 * channels)

    def forward(self, x, metadata):
        a, b = self.film(metadata).chunk(2, dim=-1)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = h * (1 + a[..., None]) + b[..., None]
        return x + self.two(F.silu(self.one(F.silu(h))))


class SlowCNN(nn.Module):
    """Same 48-channel network for all contexts; nominal RF 40.9s at 50Hz."""
    def __init__(self, channels=48):
        super().__init__()
        self.sex = nn.Embedding(3, 4)
        self.metadata = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32))
        self.input = SameConv(1, channels)
        self.blocks = nn.ModuleList(Block(channels, d) for d in DILATIONS)
        self.output = nn.Conv1d(channels, 11, 1)

    def forward(self, anchor, numeric, sex, field_mask, age_topcoded):
        if anchor.ndim != 3 or anchor.shape[1] != 1:
            raise ValueError('Model accepts only I plus demographics, never target curves')
        v = torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), dim=-1)
        meta = self.metadata(v)
        x = self.input(anchor)
        for block in self.blocks:
            x = block(x, meta)
        return self.output(F.silu(x))
