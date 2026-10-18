"""Frozen S50 CNN; original B5-exp-S50 state keys and full-record inference retained."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_slow(checkpoint,device,variant='slow'):
    if checkpoint.get('protocol') == 'B5-slow-ms-residual-v1':
        if variant != 'slow_f':
            raise ValueError('Residual slow predictor preserves the F variant')
        from baselines.B5_slow_ms.model import load_residual
        return load_residual(checkpoint, device)
    if checkpoint.get('protocol')!='B5-slow-probe-v1' or checkpoint['config'].get('context_seconds')!=50:
        raise ValueError('Requires an S50 SlowCNN checkpoint, not an FM model or S30 model')
    expected={'slow':'C','slow_f':'F'}
    if variant not in expected or checkpoint['config'].get('group')!=expected[variant] or checkpoint['config'].get('model_channels',48)!=48:
        raise ValueError('Slow source/group mismatch: slow requires C, slow_f requires F, both 48 channels')
    model=SlowCNN()
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).requires_grad_(False).eval()


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
