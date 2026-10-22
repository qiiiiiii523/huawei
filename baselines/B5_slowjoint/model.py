"""Keep the original S50 crop and G120 computation; unfreeze both networks."""
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from baselines.B5_best.slow import SlowCNN
from baselines.B5_slow120 import FIELDS
from baselines.B5_slow120.model import SlowCorrection, inputs as global_inputs


def local_native(model, anchor, fields, window_batch=2, checkpoint_blocks=False):
    """Differentiable equivalent of B5_best.slow.predict_native, no window centering."""
    if anchor.shape[1:] != (1, 6000) or window_batch < 1:
        raise ValueError("Need normalized full120 I [B,1,6000] at 50Hz")
    padded = torch.cat((anchor[..., :1].expand(-1, -1, 1000), anchor,
                        anchor[..., -1:].expand(-1, -1, 1000)), -1)
    windows = padded.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, 1, 2500)
    meta = {k: fields[k].repeat_interleave(12, 0) for k in FIELDS}
    values = []
    for start in range(0, len(windows), window_batch):
        w = windows[start:start + window_batch]
        f = {k: v[start:start + window_batch] for k, v in meta.items()}
        v = torch.cat((f["numeric"], model.sex(f["sex"]), f["field_mask"].float(), f["age_topcoded"]), -1)
        context = model.metadata(v)
        h = model.input(w)
        for block in model.blocks:
            h = (checkpoint(block, h, context, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled()
                 else block(h, context))
        values.append(model.output(torch.nn.functional.silu(h))[..., 1000:1500])
    return torch.cat(values).reshape(len(anchor), 12, 11, 500).permute(0, 2, 1, 3).reshape(-1, 11, 6000)


class JointSlow(nn.Module):
    def __init__(self, global_config, window_batch=2, checkpoint_blocks=True):
        super().__init__()
        self.local = SlowCNN()
        self.global_model = SlowCorrection("G120", **global_config)
        self.window_batch, self.checkpoint_blocks = window_batch, checkpoint_blocks

    def forward(self, anchor, global_i, local_i, numeric, sex, field_mask, age_topcoded):
        fields = dict(numeric=numeric, sex=sex, field_mask=field_mask, age_topcoded=age_topcoded)
        p = local_native(self.local, anchor, fields, self.window_batch, self.checkpoint_blocks)
        g = self.global_model(global_i, local_i, **fields)
        return p + g


def visible_batch(data, device, scale_i):
    """Only observed I/demographics are allowed into either network."""
    if "anchor" in data:
        anchor = np.asarray(data["anchor"], np.float32)
    else:
        anchor = np.asarray(data["anchor_uV"], np.float32).reshape(1, 6000, 10).mean(-1) / scale_i
    fields = {k: np.asarray(data[k]) for k in ("global_i", "local_i", *FIELDS)}
    return {k: torch.from_numpy(np.asarray(v)[None].copy()).to(device) for k, v in {"anchor": anchor, **fields}.items()}
