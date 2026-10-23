"""S50 local adaptation; optional lead/II conditions; immutable G120."""
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from baselines.B5_best.slow import SlowCNN, Block
from baselines.B5_slow120 import FIELDS
from baselines.B5_slow120.model import SlowCorrection, inputs as global_inputs


def local_native(model, anchor, fields, window_batch=2, checkpoint_blocks=False,
                 enable_decoder=True, ii_native=None):
    """Differentiable equivalent of B5_best.slow.predict_native, no window centering."""
    if anchor.shape[1:] != (1, 6000) or window_batch < 1:
        raise ValueError("Need normalized full120 I [B,1,6000] at 50Hz")
    padded = torch.cat((anchor[..., :1].expand(-1, -1, 1000), anchor,
                        anchor[..., -1:].expand(-1, -1, 1000)), -1)
    windows = padded.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, 1, 2500)
    meta = {k: fields[k].repeat_interleave(12, 0) for k in FIELDS}
    ii_windows = None
    if ii_native is not None:
        if ii_native.shape != (len(anchor), 1, 6000) or ii_native.requires_grad:
            raise ValueError("II condition must be detached full120 normalized II")
        padded_ii = torch.cat((ii_native[..., :1].expand(-1, -1, 1000), ii_native,
                               ii_native[..., -1:].expand(-1, -1, 1000)), -1)
        ii_windows = padded_ii.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, 1, 2500)
    values = []
    for start in range(0, len(windows), window_batch):
        w = windows[start:start + window_batch]
        f = {k: v[start:start + window_batch] for k, v in meta.items()}
        v = torch.cat((f["numeric"], model.sex(f["sex"]), f["field_mask"].float(), f["age_topcoded"]), -1)
        context = model.metadata(v)
        h = model.input(w)
        shared = None
        for index, block in enumerate(model.blocks):
            h = (checkpoint(block, h, context, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled()
                 else block(h, context))
            if index == 5:
                shared = h
        output = model.output(torch.nn.functional.silu(h))
        if enable_decoder and getattr(model, "lead_enabled", False):
            ii = ii_windows[start:start + window_batch] if ii_windows is not None else None
            delta = model.lead_decoder(shared, context, ii)
            output = torch.cat((output[:, :5], output[:, 5:] + delta), dim=1)
        values.append(output[..., 1000:1500])
    return torch.cat(values).reshape(len(anchor), 12, 11, 500).permute(0, 2, 1, 3).reshape(-1, 11, 6000)


class LeadDecoder(nn.Module):
    """Six conditions, shared nonlinear temporal weights; zero residual at start."""
    def __init__(self, channels=48):
        super().__init__()
        self.embedding = nn.Embedding(6, 16)
        self.condition = nn.Sequential(nn.Linear(channels + 32 + 16, 32), nn.SiLU(), nn.Linear(32, 32))
        self.blocks = nn.ModuleList([Block(channels, 64), Block(channels, 128)])
        self.output = nn.Conv1d(channels, 1, 1)
        self.ii_encoder = nn.Sequential(nn.Conv1d(1, channels, 1), nn.SiLU(),
                                        nn.Conv1d(channels, channels, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, shared, metadata, ii=None):
        summary = shared.mean(-1)
        if ii is not None:
            if ii.requires_grad:
                raise ValueError("II condition must use stop-gradient")
            shared = shared + self.ii_encoder(ii)
        values = []
        # Checkpoint each lead to limit memory without changing temporal crops.
        def decode(h, condition):
            for block in self.blocks:
                h = block(h, condition)
            return self.output(torch.nn.functional.silu(h))
        for j in range(6):
            e = self.embedding.weight[j].expand(len(shared), -1)
            condition = self.condition(torch.cat((summary, metadata, e), -1))
            values.append(checkpoint(decode, shared, condition, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else decode(shared, condition))
        return torch.cat(values, dim=1)


class JointSlow(nn.Module):
    def __init__(self, global_config, window_batch=2, checkpoint_blocks=True):
        super().__init__()
        self.local = SlowCNN()
        self.local.lead_decoder = LeadDecoder()
        self.set_group("B")
        self.global_model = SlowCorrection("G120", **global_config)
        self.window_batch, self.checkpoint_blocks = window_batch, checkpoint_blocks
        self.global_model.requires_grad_(False).eval()

    def set_group(self, group):
        if group not in ("A", "B", "C"):
            raise ValueError("Expected A, B or C")
        self.group = group
        self.local.lead_enabled = group != "A"
        self.local.lead_decoder.requires_grad_(group != "A")
        self.local.lead_decoder.ii_encoder.requires_grad_(group == "C")
        return self

    def train(self, mode=True):
        super().train(mode)
        self.global_model.eval()
        return self

    def forward(self, anchor, global_i, local_i, numeric, sex, field_mask, age_topcoded):
        fields = dict(numeric=numeric, sex=sex, field_mask=field_mask, age_topcoded=age_topcoded)
        with torch.no_grad():
            g = self.global_model(global_i, local_i, **fields)
        ii = None
        if self.group == "C":
            with torch.no_grad():
                base_p = local_native(self.local, anchor, fields, self.window_batch,
                                      enable_decoder=False)
                ii = (base_p[:, :1] + g[:, :1]).detach()
        p = local_native(self.local, anchor, fields, self.window_batch, self.checkpoint_blocks,
                         ii_native=ii)
        return p + g


def visible_batch(data, device, scale_i):
    """Only observed I/demographics are allowed into either network."""
    if "anchor" in data:
        anchor = np.asarray(data["anchor"], np.float32)
    else:
        anchor = np.asarray(data["anchor_uV"], np.float32).reshape(1, 6000, 10).mean(-1) / scale_i
    fields = {k: np.asarray(data[k]) for k in ("global_i", "local_i", *FIELDS)}
    return {k: torch.from_numpy(np.asarray(v)[None].copy()).to(device) for k, v in {"anchor": anchor, **fields}.items()}
