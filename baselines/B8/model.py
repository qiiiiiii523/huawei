"""M1's 10-second, time-only axial regressor; no flow state or U-Net."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, cin, cout, kernel=5, stride=1, dilation=1):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(nn.Conv1d(cin, cout, kernel, stride, padding, dilation=dilation),
                                 nn.GroupNorm(math.gcd(cout, 8), cout), nn.GELU())
    def forward(self, x): return self.net(x)


class LocalResidual(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.a = ConvNormAct(channels, channels, 5, dilation=dilation)
        self.b = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                               nn.GroupNorm(math.gcd(channels, 8), channels), nn.Dropout(dropout))
    def forward(self, x): return F.gelu(x + self.b(self.a(x)))


class M1Encoder(nn.Module):
    """Fixed M1 scales: 5000 -> 1250 -> 250 time tokens."""
    def __init__(self, dropout):
        super().__init__()
        self.stem = ConvNormAct(1, 64, 7)
        self.f0 = nn.Sequential(LocalResidual(64, 1, dropout), LocalResidual(64, 2, dropout))
        self.to_f1 = ConvNormAct(64, 128, 7, stride=4)
        self.f1 = nn.Sequential(LocalResidual(128, 1, dropout), LocalResidual(128, 2, dropout))
        self.to_f2 = ConvNormAct(128, 256, 5, stride=5)
        self.f2 = nn.Sequential(LocalResidual(256, 1, dropout), LocalResidual(256, 2, dropout))
    def forward(self, anchor):
        if anchor.ndim != 3 or anchor.shape[1:] != (1, 5000):
            raise ValueError("B8 requires a normalized [B,1,5000] target-time I anchor")
        f0 = self.f0(self.stem(anchor)); f1 = self.f1(self.to_f1(f0)); f2 = self.f2(self.to_f2(f1))
        if f0.shape[-1] != 5000 or f1.shape[-1] != 1250 or f2.shape[-1] != 250:
            raise RuntimeError("M1 fixed temporal scales violated")
        return f0, f1, f2


class TimeOnlyBlock(nn.Module):
    """Bidirectional attention is solely over time; leads never attend to leads."""
    def __init__(self, d_model=128, heads=4, ffn=256, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, d_model), nn.Dropout(dropout))
    def forward(self, grid):
        b, leads, steps, dim = grid.shape
        x = grid.reshape(b * leads, steps, dim)
        q = self.norm(x)
        value, _ = self.attention(q, q, q, need_weights=False, is_causal=False)
        x = x + value
        return (x + self.ffn(self.ffn_norm(x))).reshape(b, leads, steps, dim)


class PerLead(nn.Module):
    @staticmethod
    def apply(x, layer):
        b, leads, channels, length = x.shape
        return layer(x.reshape(b * leads, channels, length)).reshape(b, leads, -1, length)


class Decoder(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.coarse = nn.Linear(d_model, 128)
        self.refine1 = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.GroupNorm(8, 128), nn.GELU(), nn.Dropout(dropout))
        self.skip1_signal, self.skip1_lead = nn.Conv1d(128, 128, 1), nn.Linear(d_model, 128)
        self.fuse1 = nn.Sequential(nn.Conv1d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.refine0 = nn.Sequential(nn.Conv1d(128, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU(), nn.Dropout(dropout))
        self.skip0_signal, self.skip0_lead = nn.Conv1d(64, 64, 1), nn.Linear(d_model, 64)
        self.fuse0 = nn.Sequential(nn.Conv1d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.head, self.bias, self.per_lead = nn.Conv1d(64, 1, 3, padding=1), nn.Linear(d_model, 1), PerLead()
    def forward(self, grid, f0, f1, lead_embedding):
        b = grid.shape[0]
        leads = grid.shape[1]
        x = self.coarse(grid).permute(0, 1, 3, 2).reshape(b * leads, 128, 250)
        x = F.interpolate(x, size=1250, mode="linear", align_corners=False).reshape(b, leads, 128, 1250)
        x = self.per_lead.apply(x, self.refine1)
        skip1 = self.skip1_signal(f1)[:, None] + self.skip1_lead(lead_embedding)[None, :, :, None]
        x = self.per_lead.apply(torch.cat((x, skip1), 2), self.fuse1)
        x = F.interpolate(x.reshape(b * leads, 128, 1250), size=5000, mode="linear", align_corners=False).reshape(b, leads, 128, 5000)
        x = self.per_lead.apply(x, self.refine0)
        skip0 = self.skip0_signal(f0)[:, None] + self.skip0_lead(lead_embedding)[None, :, :, None]
        x = self.per_lead.apply(torch.cat((x, skip0), 2), self.fuse0)
        return self.per_lead.apply(x, self.head).squeeze(2) + self.bias(lead_embedding).view(1, 12, 1)


class B8M1TimeRegressor(nn.Module):
    """Direct normalized B7-seven-lead regression from synchronous visible I."""
    def __init__(self, d_model=128, blocks=4, heads=4, ffn=256, dropout=.1):
        super().__init__()
        if d_model % heads or blocks < 1: raise ValueError("invalid time-attention dimensions")
        self.architecture = {"d_model": d_model, "blocks": blocks, "heads": heads, "ffn": ffn, "dropout": dropout,
                             "attention_axes": "time_only", "time_tokens": 250, "window_samples": 5000}
        self.encoder = M1Encoder(dropout)
        self.anchor_projection = nn.Linear(256, d_model)
        self.time_position = nn.Parameter(torch.randn(1, 250, d_model) * .02)
        # Keep B7's exact generated output space: II and V1--V6.
        self.lead_embedding = nn.Parameter(torch.randn(7, d_model) * .02)
        self.blocks = nn.ModuleList(TimeOnlyBlock(d_model, heads, ffn, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(d_model, dropout)
    def forward(self, anchor):
        f0, f1, f2 = self.encoder(anchor)
        tokens = self.anchor_projection(f2.transpose(1, 2)) + self.time_position
        grid = tokens[:, None] + self.lead_embedding[None, :, None]
        for block in self.blocks: grid = block(grid)
        return self.decoder(self.norm(grid), f0, f1, self.lead_embedding)
