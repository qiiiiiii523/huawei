"""Frozen multirate sources, zero-output fusion, or original-window P adaptation."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.B5_slow120 import FIELDS
from baselines.B5_slowrate import RATES
from baselines.B5_best.slow import Block, predict_native
from . import GROUPS

INPUT_FIELDS = (*("signal_" + g for g in RATES), "anchor50", *FIELDS)


def inputs(batch):
    return {k: batch[k] for k in INPUT_FIELDS}


def array_batch(data, device):
    return {k: torch.from_numpy(np.asarray(data[k])[None].copy()).to(device) for k in INPUT_FIELDS}


def rate_forward(model, batch):
    return model(global_i=batch["signal_" + model.group], **{k: batch[k] for k in FIELDS})


class FixedEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        if not models:
            raise ValueError("Need at least one source")
        self.models = nn.ModuleList(models).requires_grad_(False)

    def forward(self, **batch):
        if set(batch) != set(INPUT_FIELDS):
            raise TypeError("Only visible I representations and demographics are model inputs")
        return torch.stack([rate_forward(m, batch) for m in self.models]).mean(0)


class PDelta(nn.Module):
    """Train only P's last blocks/head; keep 50s windows and central10s stitching."""
    def __init__(self, original, tail_blocks=2, context_seconds=50, extra_dilation=None):
        super().__init__()
        if tail_blocks not in (1, 2, 4):
            raise ValueError("tail_blocks must be 1, 2 or 4; full P retraining is a separate experiment")
        self.p = deepcopy(original).requires_grad_(False)
        self.cut = len(self.p.blocks) - tail_blocks
        self.reference_blocks = deepcopy(self.p.blocks[self.cut:]).requires_grad_(False)
        self.reference_head = deepcopy(self.p.output).requires_grad_(False)
        self.context_seconds = context_seconds
        if context_seconds not in (50, 60, 80, 100, 120):
            raise ValueError("Use 50/60/80/100/120 seconds for P")
        self.original50 = deepcopy(original).requires_grad_(False).eval() if context_seconds != 50 else None
        self.extra = nn.ModuleList()
        if extra_dilation is not None:
            block = Block(self.p.output.in_channels, extra_dilation)
            nn.init.zeros_(block.two.conv.weight)
            nn.init.zeros_(block.two.conv.bias)
            self.extra.append(block)
        for block in self.p.blocks[self.cut:]:
            block.requires_grad_(True)
        self.p.output.requires_grad_(True)

    @property
    def receptive_field_samples(self):
        layers = [self.p.input] + [conv for b in [*self.p.blocks, *self.extra] for conv in (b.one, b.two)]
        return 1 + sum((layer.conv.kernel_size[0] - 1) * layer.conv.dilation[0] for layer in layers)

    def forward(self, anchor50, initial=False, **meta):
        if anchor50.shape[1:] != (1, 6000):
            raise ValueError("P requires complete normalized 50Hz I")
        width = self.context_seconds * 50
        margin = (width - 500) // 2
        padded = torch.cat((anchor50[..., :1].expand(-1, -1, margin), anchor50,
                            anchor50[..., -1:].expand(-1, -1, margin)), -1)
        windows = padded.unfold(-1, width, 500).permute(0, 2, 1, 3).reshape(-1, 1, width)
        with torch.no_grad():
            repeated = {k: meta[k].repeat_interleave(12, 0) for k in FIELDS}
            vector = torch.cat((repeated["numeric"], self.p.sex(repeated["sex"]),
                                repeated["field_mask"].float(), repeated["age_topcoded"]), -1)
            encoded = self.p.metadata(vector)
            features = self.p.input(windows)
            for block in self.p.blocks[:self.cut]:
                features = block(features, encoded)
            reference = features
            for block in self.reference_blocks:
                reference = block(reference, encoded)
            reference = self.reference_head(F.silu(reference))[..., margin:margin + 500]
        updated = reference
        if not initial:
            updated = features
            for block in self.p.blocks[self.cut:]:
                updated = block(updated, encoded)
            for block in self.extra:
                updated = block(updated, encoded)
            updated = self.p.output(F.silu(updated))[..., margin:margin + 500]
        def stitch(value):
            return value.reshape(len(anchor50), 12, 11, 500).permute(0, 2, 1, 3).reshape(-1, 11, 6000)
        old = stitch(reference) if self.original50 is None else predict_native(
            self.original50.eval(), {"anchor": anchor50, **meta})
        return stitch(updated) - old


class Followup(nn.Module):
    def __init__(self, group, sources, reference_group="R5", original_p=None, tail_blocks=2,
                 p_context_seconds=100, p_extra_dilation=256):
        super().__init__()
        if group not in GROUPS or set(sources) != set(RATES) or reference_group not in RATES:
            raise ValueError("Invalid follow-up sources/group")
        self.group, self.reference_group = group, reference_group
        self.sources = nn.ModuleDict(sources).requires_grad_(False)
        if group == "CONT":
            self.continuation = deepcopy(self.sources[reference_group]).requires_grad_(True)
        elif group in ("FUSION", "FUSION_SAME"):
            self.fusion = nn.Sequential(nn.Conv1d(33, 16, 1), nn.SiLU(), nn.Conv1d(16, 11, 1))
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)
        else:
            if original_p is None:
                raise ValueError("PTAIL requires the pinned original P-S50")
            self.p_delta = PDelta(original_p, tail_blocks, p_context_seconds if group.startswith("PCTX") else 50,
                                  p_extra_dilation if group == "PCTX_RF" else None)

    def train(self, mode=True):
        super().train(mode)
        self.sources.eval()
        if hasattr(self, "p_delta"):
            # No dropout/BN in P. Eval preserves its original execution contract;
            # gradients still flow through the explicitly trainable tail.
            self.p_delta.eval()
        return self

    @torch.no_grad()
    def reference_delta(self, **batch):
        delta = rate_forward(self.sources[self.reference_group], batch)
        if self.group.startswith("PCTX"):
            delta = delta + self.p_delta(batch["anchor50"], initial=True, **{k: batch[k] for k in FIELDS})
        return delta

    def p_change(self, **batch):
        return self.p_delta(batch["anchor50"], **{k: batch[k] for k in FIELDS})

    def components(self, **batch):
        if set(batch) != set(INPUT_FIELDS):
            raise TypeError("Only visible I representations and demographics are model inputs")
        if self.group == "CONT":
            return rate_forward(self.continuation, batch), None
        # PCTX actually replaces P's execution, so use the original global
        # correction here, rather than adding the initial PCTX change twice.
        with torch.no_grad():
            reference = rate_forward(self.sources[self.reference_group], batch)
        if self.group in ("PTAIL", "PCTX", "PCTX_RF"):
            change = self.p_change(**batch)
            return reference + change, change
        with torch.no_grad():
            signals = ([reference] * 3 if self.group == "FUSION_SAME" else
                       [rate_forward(self.sources[g], batch) for g in RATES])
        return reference + self.fusion(torch.cat(signals, 1)), None

    def forward(self, **batch):
        return self.components(**batch)[0]
