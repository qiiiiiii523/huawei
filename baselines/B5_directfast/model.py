"""A deterministic use of the same U-Net, plus full-record differentiable H."""
import numpy as np
import torch
from torch.nn import functional as F

from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.B5_best.combine import slow_curve
from . import FIELDS, GROUPS


class DirectFastNet(B5UNet):
    def forward(self, anchor, numeric, sex, field_mask, age_topcoded):
        condition = self.encode_conditions(anchor, numeric, sex, field_mask, age_topcoded)
        return self.velocity(anchor.new_zeros(len(anchor), 11, anchor.shape[-1]),
                             anchor.new_zeros(len(anchor)), condition)


def visible_windows(anchor_uV, metadata, scale_i, device):
    raw = np.asarray(anchor_uV, np.float32)
    if raw.shape != (1, 60000) or not np.isfinite(raw).all():
        raise ValueError("Need finite visible I [1,60000], raw uV at 500Hz")
    fields = {k: torch.from_numpy(np.repeat(np.asarray(metadata[k])[None], 12, axis=0).copy()).to(device)
              for k in FIELDS}
    fields["anchor"] = torch.from_numpy((raw.reshape(1, 12, 5000).transpose(1, 0, 2)
                                         / scale_i).copy()).to(device)
    return fields


def build_model(group, model_config, state, device):
    if group not in GROUPS:
        raise ValueError("Only F1/D1 are trainable")
    cls = DirectFastNet if group == "D1" else B5UNet
    model = cls(ModelConfig.from_dict(model_config))
    model.load_state_dict(state, strict=True)
    if group == "D1":
        model.anchor_head.requires_grad_(False)  # original frozen head is used for deployment I
    return model.to(device)


def highpass(value):
    """H exactly once, preserving gradients across real window boundaries."""
    if value.ndim != 3 or value.shape[-1] < 501:
        raise ValueError("H expects [B,C,T] with at least 501 samples")
    # Explicit repeats avoid replication-pad CUDA backward nondeterminism.
    # Average pooling also avoids CUDA floating cumsum's deterministic-mode
    # restriction. Float64 mean differs from NumPy cumsum only by roundoff.
    x = value.double()
    padded = torch.cat((x[..., :1].expand(-1, -1, 250), x,
                        x[..., -1:].expand(-1, -1, 250)), dim=-1)
    return (x - F.avg_pool1d(padded, 501, stride=1)).to(value.dtype)


def assemble(q_uV, slow_uV, original_i_uV):
    if q_uV.shape != (11, 60000) or slow_uV.shape != q_uV.shape or original_i_uV.shape != (1, 60000):
        raise ValueError("Assembly needs full-record Q, predicted P and original I head")
    result = np.concatenate((original_i_uV.astype(np.float64),
                             q_uV.astype(np.float64) - slow_curve(q_uV) + .75 * slow_uV), axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite assembled output")
    return result.astype(np.float32)
