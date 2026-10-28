"""Small offline residual CNN: measured I + frozen seven predictions -> 1s offsets."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GENERATED = (1, 6, 7, 8, 9, 10, 11)
GROUP_WEIGHTS = {'task1': .5, 'machine': .25, 'body': .25}
LEADS = ('II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(.1),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(.1))

    def forward(self, x):
        return x + self.layers(x)


class MeanCorrector(nn.Module):
    """31,687 parameters. Symmetric context; no real-time/causal claim."""
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Conv1d(9, 32, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(32, 32, 5, padding=2), nn.GELU())
        self.temporal = nn.Sequential(*(ResidualBlock(32, d) for d in (1, 2, 4, 8)))
        self.head = nn.Conv1d(32, 7, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != 9 or inputs.shape[-1] % 50:
            raise ValueError('Expected normalized inputs[B,9,T*50] at50Hz')
        local = F.avg_pool1d(self.local(inputs), 50, 50)
        return self.head(self.temporal(local))


def offsets11(delta7_uV):
    """I is fixed, hence all limb offsets follow the single predicted II offset."""
    ii = delta7_uV[:, :1]
    return torch.cat((ii, ii, -.5*ii, -.5*ii, ii, delta7_uV[:, 1:]), dim=1)


def compose11(anchor, generated):
    ii = generated[:1]
    return np.concatenate((ii, ii-anchor, -(anchor+ii)/2, anchor-ii/2,
                           ii-anchor/2, generated[1:]), axis=0)


def make_features(anchor, generated, scales):
    """Target-free features identical to the original frozen-C5 experiment."""
    if anchor.shape != (1,60000) or generated.shape != (7,60000):
        raise ValueError('Expected original full120s anchor and generated leads')
    if not np.isfinite(anchor).all() or not np.isfinite(generated).all():
        raise ValueError('Nonfinite inference inputs')
    scales=np.asarray(scales,dtype=np.float64)
    if scales.shape!=(12,) or not np.isfinite(scales).all() or (scales<=0).any():
        raise ValueError('Twelve positive fixed training scales required')
    visible=np.concatenate((anchor,generated)).astype(np.float64)
    visible50=visible.reshape(8,6000,10).mean(-1)/scales[[0,*GENERATED],None]
    position=(np.arange(6000,dtype=np.float64)*10+4.5)/30000-1
    return np.concatenate((visible50,position[None])).astype(np.float32)


def apply_correction(model, anchor, generated, scales):
    """Return full12 output and7x120 offsets; no target or QC label input."""
    if model.training:
        raise ValueError('Correction inference requires eval mode')
    features=make_features(anchor,generated,scales)
    device=next(model.parameters()).device
    with torch.no_grad():
        delta=model(torch.from_numpy(features)[None].to(device))[0].cpu().numpy()
    delta=delta.astype(np.float64)*np.asarray(scales)[list(GENERATED),None]
    corrected=generated.astype(np.float64)+np.repeat(delta,500,axis=-1)
    output=np.concatenate((anchor.astype(np.float64),compose11(anchor.astype(np.float64),corrected)))
    return output.astype(np.float32),delta.astype(np.float32)


def make_example(anchor, generated, target, mask, scales):
    """Targets are labels/statistics only. Input computation never uses a target."""
    if anchor.shape != (1, 60000) or generated.shape != (7, 60000):
        raise ValueError('Every record must be a real120s record at500Hz')
    if target.shape != (12, 60000) or mask.shape != (12,):
        raise ValueError('Target/QC shape mismatch')
    if not np.isfinite(anchor).all() or not np.isfinite(generated).all():
        raise ValueError('Nonfinite inference inputs')
    # Keep absolute voltage. No record/window demeaning or per-target scaling.
    features = make_features(anchor,generated,scales)
    p = compose11(anchor.astype(np.float64), generated.astype(np.float64))
    valid = mask[1:].astype(bool)
    y = np.where(valid[:, None], target[1:], 0.).astype(np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite eligible training targets')
    pb, yb = p.reshape(11, 120, 500), y.reshape(11, 120, 500)
    pm, ym = pb.mean(-1), yb.mean(-1)
    pc, yc = pb-pm[..., None], yb-ym[..., None]
    return dict(features=features, pm=pm, ym=ym, pss=(pc*pc).sum(-1),
                yss=(yc*yc).sum(-1), cross=(pc*yc).sum(-1), mask=valid)


def statistics_metrics(pm, ym, pss, yss, cross, delta11):
    """Exact full-record r/RMSE after piecewise-constant1s corrections.

    All arguments are tensors[B,11,120]. Within-block centered moments are
    invariant to a constant correction, so no waveform approximation is used.
    """
    qm = pm + delta11
    qc, yc = qm-qm.mean(-1, keepdim=True), ym-ym.mean(-1, keepdim=True)
    qp = pss.sum(-1) + 500*(qc*qc).sum(-1)
    yy = yss.sum(-1) + 500*(yc*yc).sum(-1)
    xy = cross.sum(-1) + 500*(qc*yc).sum(-1)
    # Keep excluded/constant targets from creating NaN gradients at sqrt(0).
    denominator = (qp*yy).clamp_min(1e-30).sqrt()
    r = xy / denominator
    r = torch.where((qp > 0) & (yy > 0), r, torch.full_like(r, float('nan')))
    mse = ((pss+yss-2*cross).sum(-1) + 500*((qm-ym)**2).sum(-1))/60000
    return r, mse.clamp_min(0).sqrt()


def learning_loss(batch, delta7_normalized, scales):
    scale7 = scales[list(GENERATED)][None, :, None]
    delta = offsets11(delta7_normalized*scale7)
    residual = (batch['pm'] + delta - batch['ym'])/scales[None, 1:, None]
    huber = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none').mean(-1)
    r, _ = statistics_metrics(*(batch[k] for k in ('pm','ym','pss','yss','cross')), delta)
    mask = batch['mask'] & torch.isfinite(r)
    # Match group/lead/physical-target weights, not the number of short windows.
    weights = batch['weights']*mask
    return ((huber + .1*(1-torch.nan_to_num(r)))*weights).sum()/weights.sum().clamp_min(1e-12)
