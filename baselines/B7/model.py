"""B7-T: raw-I, multiscale 1-D Transformer velocity field; no CNN backbone."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class TimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(dimension // 2) / max(1, dimension // 2 - 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.SiLU(), nn.Linear(dimension * 4, dimension))

    def forward(self, time):
        phase = time[:, None].float() * 1000 * self.frequencies[None]
        return self.mlp(torch.cat((phase.sin(), phase.cos()), dim=-1))


class MetadataEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.sex_embedding = nn.Embedding(3, 8)
        self.mlp = nn.Sequential(nn.Linear(16, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, reference, metadata=None):
        b = len(reference)
        if not metadata:
            numeric = reference.new_zeros(b, 3)
            valid = torch.zeros(b, 4, dtype=torch.bool, device=reference.device)
            sex = torch.full((b,), 2, dtype=torch.long, device=reference.device)
            topcoded = reference.new_zeros(b, 1)
        else:
            numeric = metadata["numeric"].to(device=reference.device, dtype=reference.dtype)
            sex = metadata["sex"].to(device=reference.device)
            valid = metadata["field_mask"].to(device=reference.device, dtype=torch.bool)
            topcoded = metadata["age_topcoded"].to(device=reference.device, dtype=reference.dtype)
        if numeric.shape != (b, 3) or valid.shape != (b, 4) or sex.shape != (b,) or topcoded.shape != (b, 1):
            raise ValueError("Metadata shape mismatch")
        numeric = torch.where(valid[:, [0, 2, 3]], numeric, torch.zeros_like(numeric))
        sex = torch.where(valid[:, 1], sex.long(), torch.full_like(sex.long(), 2))
        topcoded = torch.where(valid[:, :1], topcoded, torch.zeros_like(topcoded))
        if not torch.isfinite(numeric).all() or not torch.isfinite(topcoded).all() or torch.any((sex < 0) | (sex > 2)):
            raise ValueError("Invalid observed demographics")
        return self.mlp(torch.cat((numeric, self.sex_embedding(sex), valid.to(numeric.dtype), topcoded), dim=-1))


def temporal_positions(length, dimension, reference):
    """Length-independent sinusoidal positions, shared by 10s and 40s runs."""
    frequencies = torch.exp(-math.log(10000.) * torch.arange(
        dimension // 2, device=reference.device, dtype=torch.float32) / max(1, dimension // 2 - 1))
    phase = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None] * frequencies[None]
    return torch.cat((phase.sin(), phase.cos()), -1).to(reference.dtype)[None]


class PatchEmbedding(nn.Module):
    """Jointly flatten every raw sample of I+state; never pool/encode I separately."""
    def __init__(self, patch_size, dimension):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(8 * patch_size, dimension)

    def forward(self, waveform):
        pad = (-waveform.shape[-1]) % self.patch_size
        if pad:
            waveform = F.pad(waveform, (0, pad), mode="replicate")
        b, _, length = waveform.shape
        patches = waveform.transpose(1, 2).reshape(b, length // self.patch_size, 8 * self.patch_size)
        return self.projection(patches)


class SelfAttention(nn.Module):
    def __init__(self, dimension, heads, window_size=None):
        super().__init__()
        self.heads = heads
        self.scale = (dimension // heads) ** -.5
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.projection = nn.Linear(dimension, dimension)
        self.window_size = window_size
        if window_size is not None:
            self.relative_bias = nn.Parameter(torch.zeros(2 * window_size - 1, heads))
            positions = torch.arange(window_size)
            self.register_buffer("relative_index", positions[:, None] - positions[None] + window_size - 1,
                                 persistent=False)
            nn.init.trunc_normal_(self.relative_bias, std=.02)

    def forward(self, tokens, valid):
        b, n, dimension = tokens.shape
        q, k, v = self.qkv(tokens).reshape(b, n, 3, self.heads, dimension // self.heads).permute(2, 0, 3, 1, 4)
        logits = (q * self.scale) @ k.transpose(-2, -1)
        if self.window_size is not None:
            if n != self.window_size:
                raise ValueError("Local attention expects one padded window")
            logits = logits + self.relative_bias[self.relative_index].permute(2, 0, 1)[None]
        logits = logits.masked_fill(~valid[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = logits.float().softmax(-1).to(logits.dtype)
        # Explicitly zero keys, including the all-padding-window case.
        weights = weights.masked_fill(~valid[:, None, None, :], 0.)
        values = (weights @ v).transpose(1, 2).reshape(b, n, dimension)
        return self.projection(values).masked_fill(~valid[:, :, None], 0.)


class TemporalBlock(nn.Module):
    def __init__(self, dimension, heads, condition_dim, mlp_ratio, window_size=None, shift=0):
        super().__init__()
        self.window_size, self.shift = window_size, shift
        self.norm1 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, dimension * 4))
        self.attention = SelfAttention(dimension, heads, window_size)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * mlp_ratio), nn.GELU(),
                                 nn.Linear(dimension * mlp_ratio, dimension))

    def attend(self, tokens, valid):
        if self.window_size is None:
            return self.attention(tokens, valid)
        b, n, dimension = tokens.shape
        width, left = self.window_size, self.shift
        right = (-(n + left)) % width
        # Non-cyclic shifted windows: the ECG end never wraps around to its start.
        padded = F.pad(tokens, (0, 0, left, right))
        mask = F.pad(valid, (left, right), value=False)
        windows = padded.reshape(-1, width, dimension)
        attended = self.attention(windows, mask.reshape(-1, width))
        return attended.reshape(b, n + left + right, dimension)[:, left:left + n]

    def forward(self, tokens, valid, condition):
        shift_a, scale_a, shift_m, scale_m = self.modulation(condition).chunk(4, -1)
        normalized = self.norm1(tokens) * (1 + scale_a[:, None]) + shift_a[:, None]
        tokens = tokens + self.attend(normalized.masked_fill(~valid[:, :, None], 0.), valid)
        normalized = self.norm2(tokens) * (1 + scale_m[:, None]) + shift_m[:, None]
        tokens = tokens + self.mlp(normalized)
        return tokens.masked_fill(~valid[:, :, None], 0.)


class PatchMerging(nn.Module):
    def __init__(self, dimension, output_dimension):
        super().__init__()
        self.norm = nn.LayerNorm(dimension * 2)
        self.projection = nn.Linear(dimension * 2, output_dimension)

    def forward(self, tokens, valid):
        tokens = tokens.masked_fill(~valid[:, :, None], 0.)
        if tokens.shape[1] % 2:
            tokens, valid = F.pad(tokens, (0, 0, 0, 1)), F.pad(valid, (0, 1), value=False)
        b, n, dimension = tokens.shape
        merged_valid = valid[:, 0::2] | valid[:, 1::2]
        merged = self.projection(self.norm(tokens.reshape(b, n // 2, dimension * 2)))
        return merged.masked_fill(~merged_valid[:, :, None], 0.), merged_valid


class PatchExpanding(nn.Module):
    def __init__(self, dimension, output_dimension):
        super().__init__()
        self.projection = nn.Linear(dimension, output_dimension * 2)
        self.fusion = nn.Linear(output_dimension * 2, output_dimension)

    def forward(self, tokens, skip, valid):
        expanded = self.projection(tokens).reshape(len(tokens), -1, skip.shape[-1])[:, :skip.shape[1]]
        return self.fusion(torch.cat((expanded, skip), -1)).masked_fill(~valid[:, :, None], 0.)


class B7Transformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c0, c1, c2 = self.config.embed_dims
        self.time_embedding = TimeEmbedding(self.config.time_dim)
        self.metadata_encoder = MetadataEncoder(self.config.metadata_dim)
        self.patch_embed = PatchEmbedding(self.config.patch_size, c0)

        def blocks(level, global_attention=False):
            return nn.ModuleList([TemporalBlock(
                self.config.embed_dims[level], self.config.heads[level],
                self.config.time_dim + self.config.metadata_dim, self.config.mlp_ratio,
                None if global_attention else self.config.window_size,
                0 if global_attention or i % 2 == 0 else self.config.window_size // 2)
                for i in range(self.config.depths[level])])

        self.encode0, self.encode1 = blocks(0), blocks(1)
        self.down1, self.down2 = PatchMerging(c0, c1), PatchMerging(c1, c2)
        self.bottleneck = blocks(2, global_attention=True)
        self.up1, self.up0 = PatchExpanding(c2, c1), PatchExpanding(c1, c0)
        self.decode1, self.decode0 = blocks(1), blocks(0)
        self.output_norm = nn.LayerNorm(c0)
        self.output = nn.Linear(c0, 7 * self.config.patch_size)

    @staticmethod
    def _blocks(blocks, tokens, valid, condition):
        for block in blocks:
            tokens = block(tokens, valid, condition)
        return tokens

    def forward(self, state, time, anchor, metadata=None):
        if state.ndim != 3 or state.shape[1] != 7 or state.shape[-1] < 8:
            raise ValueError("State must be [B,7,T], T>=8")
        if anchor.shape != (len(state), 1, state.shape[-1]) or time.shape != (len(state),):
            raise ValueError("I must be [B,1,T] and time [B]")
        if not torch.isfinite(state).all() or not torch.isfinite(anchor).all() or not torch.isfinite(time).all():
            raise ValueError("Nonfinite inputs")
        if torch.any((time < 0) | (time > 1)):
            raise ValueError("Flow time must be in [0,1]")
        condition = torch.cat((self.time_embedding(time), self.metadata_encoder(anchor, metadata)), -1)
        tokens = self.patch_embed(torch.cat((anchor, state), dim=1))
        tokens = tokens + temporal_positions(tokens.shape[1], tokens.shape[2], tokens)
        valid0 = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        s0 = self._blocks(self.encode0, tokens, valid0, condition)
        tokens, valid1 = self.down1(s0, valid0)
        s1 = self._blocks(self.encode1, tokens, valid1, condition)
        tokens, valid2 = self.down2(s1, valid1)
        tokens = self._blocks(self.bottleneck, tokens, valid2, condition)
        tokens = self._blocks(self.decode1, self.up1(tokens, s1, valid1), valid1, condition)
        tokens = self._blocks(self.decode0, self.up0(tokens, s0, valid0), valid0, condition)
        patches = self.output(self.output_norm(tokens))
        waveform = patches.reshape(len(state), -1, 7).transpose(1, 2)
        return waveform[:, :, :state.shape[-1]].contiguous()
