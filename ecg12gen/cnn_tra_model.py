"""Lead-I to (II, V1..V6) CNN/Transformer/FPN backbone derived from B3.

Input and output are scaled waveforms, each with 5000 samples (10 seconds).
The 120-second record is handled as twelve consecutive windows by the trainer.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


OUTPUT_LEADS = ("II", "V1", "V2", "V3", "V4", "V5", "V6")
WINDOW_SAMPLES = 5000
CHANNELS = (64, 128, 256)
TOKENS = 250
MODEL_DIM = 64


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.local = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.long = nn.Conv1d(channels, channels, 15, padding=7 * dilation,
                              dilation=dilation, groups=channels)
        self.mix = nn.Conv1d(2 * channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.norm(self.mix(torch.cat(
            (self.local(x), self.long(x)), dim=1))))


class AnchorCNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, 7, padding=3),
                                  nn.GroupNorm(8, 64), nn.GELU())
        self.f0_block = nn.Sequential(MultiScaleResidualBlock(64, 1),
                                      MultiScaleResidualBlock(64, 2))
        self.down1 = nn.Sequential(nn.Conv1d(64, 128, 7, stride=4, padding=3),
                                   nn.GroupNorm(8, 128), nn.GELU())
        self.f1_block = nn.Sequential(MultiScaleResidualBlock(128, 1),
                                      MultiScaleResidualBlock(128, 2))
        self.down2 = nn.Sequential(nn.Conv1d(128, 256, 7, stride=5, padding=3),
                                   nn.GroupNorm(16, 256), nn.GELU())
        self.f2_block = nn.Sequential(MultiScaleResidualBlock(256, 1),
                                      MultiScaleResidualBlock(256, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (1, WINDOW_SAMPLES):
            raise ValueError("I input must be [batch, 1, 5000]")
        f0 = self.f0_block(self.stem(x))
        f1 = self.f1_block(self.down1(f0))
        f2 = self.f2_block(self.down2(f1))
        if tuple(f.shape[-1] for f in (f0, f1, f2)) != (5000, 1250, TOKENS):
            raise RuntimeError("CNN pyramid length mismatch")
        return f0, f1, f2


class SevenLeadFPNDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lead_embedding = nn.Parameter(torch.zeros(7, MODEL_DIM))
        self.lead_norm = nn.LayerNorm(MODEL_DIM)
        self.lead_mlp = nn.Sequential(nn.Linear(MODEL_DIM, 128), nn.GELU(),
                                      nn.Linear(128, MODEL_DIM))
        self.to_1250 = nn.Conv1d(MODEL_DIM, 128, 1)
        self.skip_1250 = nn.Conv1d(128, 128, 1)
        self.refine_1250 = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2),
                                         nn.GroupNorm(8, 128), nn.GELU())
        self.to_5000 = nn.Conv1d(128, 64, 1)
        self.skip_5000 = nn.Conv1d(64, 64, 1)
        self.refine_5000 = nn.Sequential(nn.Conv1d(64, 64, 7, padding=3),
                                         nn.GroupNorm(8, 64), nn.GELU())
        self.output = nn.Conv1d(64, 1, 7, padding=3)
        nn.init.normal_(self.lead_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor, f0: torch.Tensor,
                f1: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        lead_tokens = tokens[:, None] + self.lead_embedding[None, :, None, :]
        lead_tokens = lead_tokens + self.lead_mlp(self.lead_norm(lead_tokens))
        x = lead_tokens.permute(0, 1, 3, 2).reshape(batch * 7, MODEL_DIM, TOKENS)
        x = F.interpolate(self.to_1250(x), size=1250, mode="linear", align_corners=False)
        skip1 = self.skip_1250(f1)[:, None].expand(batch, 7, 128, 1250).reshape_as(x)
        x = self.refine_1250(x + skip1)
        x = F.interpolate(self.to_5000(x), size=5000, mode="linear", align_corners=False)
        skip0 = self.skip_5000(f0)[:, None].expand(batch, 7, 64, 5000).reshape_as(x)
        return self.output(self.refine_5000(x + skip0)).reshape(batch, 7, WINDOW_SAMPLES)


class CNNTransformerSevenLead(nn.Module):
    def __init__(self, transformer_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if not 1 <= transformer_layers <= 8:
            raise ValueError("transformer_layers must be 1..8")
        self.cnn_encoder = AnchorCNNEncoder()
        self.token_projection = nn.Conv1d(256, MODEL_DIM, 1)
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM, nhead=4, dim_feedforward=256, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.time_transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.register_buffer("time_position", self._sinusoidal(TOKENS, MODEL_DIM),
                             persistent=True)
        self.decoder = SevenLeadFPNDecoder()

    @staticmethod
    def _sinusoidal(length: int, dimension: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        return encoding.unsqueeze(0)

    def forward(self, anchor_i: torch.Tensor) -> torch.Tensor:
        f0, f1, f2 = self.cnn_encoder(anchor_i)
        tokens = self.token_projection(f2).transpose(1, 2)
        tokens = tokens + self.time_position.to(device=tokens.device, dtype=tokens.dtype)
        tokens = self.time_transformer(tokens)  # bidirectional attention within 10 s
        return self.decoder(tokens, f0, f1)
