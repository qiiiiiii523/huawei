"""Keep the original S50 crop and G120 computation; unfreeze both networks."""
import math
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from baselines.B5_best.slow import SlowCNN
from baselines.B5_slow120 import FIELDS
from baselines.B5_slow120.model import SlowCorrection, inputs as global_inputs, tokens_to_native


FUSION_MODES = ("add", "mid")


class GlobalFiLM(nn.Module):
    """Channel/time conditioning, initialized as an EXACT identity on local features."""
    def __init__(self, incoming, local_channels=48):
        super().__init__()
        self.norm = nn.LayerNorm(incoming)  # channel-only, no time/voltage centering
        self.hidden = nn.Conv1d(incoming, incoming, 1)
        self.output = nn.Conv1d(incoming, 2 * local_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, local, global_features):
        z = self.norm(global_features.transpose(1, 2)).transpose(1, 2)
        gamma, beta = self.output(torch.nn.functional.silu(self.hidden(z))).chunk(2, dim=1)
        # Bound multiplicative changes initially; voltage shifts remain learnable.
        return local * (1 + .1 * gamma.tanh()) + beta


def global_forward(model, global_i, fields):
    """Original G120 output plus its last pre-output features; no new attention pass."""
    if global_i.shape[1:] != (1, 600):
        raise ValueError("G120 requires full120s I at 5Hz")
    meta = torch.cat((fields["numeric"], model.sex(fields["sex"]),
                      fields["field_mask"].float(), fields["age_topcoded"]), -1)
    x = model.input(global_i.transpose(1, 2)) + model.metadata(meta)[:, None]
    position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
    frequency = torch.exp(torch.arange(0, model.channels, 2, device=x.device, dtype=x.dtype)
                          * (-math.log(10000.) / model.channels))
    angle = position * frequency
    x = x + torch.stack((angle.sin(), angle.cos()), -1).reshape(x.shape[1], model.channels)[None]
    for block in model.blocks:
        x = block(x)
    correction = tokens_to_native(model.output(x).transpose(1, 2))
    features = tokens_to_native(x.transpose(1, 2))
    return correction, features


def local_native(model, anchor, fields, window_batch=2, checkpoint_blocks=False,
                 fusion=None, global_features=None, fusion_after=6):
    """Differentiable equivalent of B5_best.slow.predict_native, no window centering."""
    if anchor.shape[1:] != (1, 6000) or window_batch < 1:
        raise ValueError("Need normalized full120 I [B,1,6000] at 50Hz")
    padded = torch.cat((anchor[..., :1].expand(-1, -1, 1000), anchor,
                        anchor[..., -1:].expand(-1, -1, 1000)), -1)
    windows = padded.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, 1, 2500)
    meta = {k: fields[k].repeat_interleave(12, 0) for k in FIELDS}
    feature_windows = None
    if fusion is not None:
        if global_features is None or global_features.shape[0] != len(anchor) or global_features.shape[-1] != 6000:
            raise ValueError("Need aligned full-record global features")
        if not 1 <= fusion_after < len(model.blocks):
            raise ValueError("Fusion must leave local processing blocks after the injection")
        gf = torch.cat((global_features[..., :1].expand(-1, -1, 1000), global_features,
                        global_features[..., -1:].expand(-1, -1, 1000)), -1)
        feature_windows = gf.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, gf.shape[1], 2500)
    values = []
    for start in range(0, len(windows), window_batch):
        w = windows[start:start + window_batch]
        f = {k: v[start:start + window_batch] for k, v in meta.items()}
        v = torch.cat((f["numeric"], model.sex(f["sex"]), f["field_mask"].float(), f["age_topcoded"]), -1)
        context = model.metadata(v)
        h = model.input(w)
        for index, block in enumerate(model.blocks, 1):
            h = (checkpoint(block, h, context, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled()
                 else block(h, context))
            if fusion is not None and index == fusion_after:
                gf = feature_windows[start:start + window_batch]
                h = (checkpoint(fusion, h, gf, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled()
                     else fusion(h, gf))
        values.append(model.output(torch.nn.functional.silu(h))[..., 1000:1500])
    return torch.cat(values).reshape(len(anchor), 12, 11, 500).permute(0, 2, 1, 3).reshape(-1, 11, 6000)


class JointSlow(nn.Module):
    def __init__(self, global_config, window_batch=2, checkpoint_blocks=True, fusion_mode="add"):
        super().__init__()
        self.local = SlowCNN()
        self.global_model = SlowCorrection("G120", **global_config)
        self.window_batch, self.checkpoint_blocks = window_batch, checkpoint_blocks
        if fusion_mode not in FUSION_MODES:
            raise ValueError("Use add (matched joint control) or mid (feature modulation)")
        self.fusion_mode = fusion_mode
        # A local fixed construction seed avoids consuming extra training RNG in MID.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            self.fusion = GlobalFiLM(global_config.get("channels", 32)) if fusion_mode == "mid" else None

    def forward(self, anchor, global_i, local_i, numeric, sex, field_mask, age_topcoded):
        fields = dict(numeric=numeric, sex=sex, field_mask=field_mask, age_topcoded=age_topcoded)
        if self.fusion is None:
            p = local_native(self.local, anchor, fields, self.window_batch, self.checkpoint_blocks)
            g = self.global_model(global_i, local_i, **fields)
        else:
            g, features = global_forward(self.global_model, global_i, fields)
            p = local_native(self.local, anchor, fields, self.window_batch, self.checkpoint_blocks,
                             self.fusion, features, fusion_after=6)
        return p + g


def visible_batch(data, device, scale_i):
    """Only observed I/demographics are allowed into either network."""
    if "anchor" in data:
        anchor = np.asarray(data["anchor"], np.float32)
    else:
        anchor = (np.asarray(data["anchor_uV"], np.float32).reshape(1, 6000, 10).mean(-1)
                  / np.float32(scale_i)).astype(np.float32, copy=False)
    fields = {k: np.asarray(data[k]) for k in ("global_i", "local_i", *FIELDS)}
    return {k: torch.from_numpy(np.asarray(v)[None].copy()).to(device) for k, v in {"anchor": anchor, **fields}.items()}
