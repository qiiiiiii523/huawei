"""M1 final backbone and context adapters.

The model has one public prediction path: machine-I at the target time is the
only waveform input to the anchor backbone. Context is encoded to a global
vector and can only condition the backbone through the four named fusion
modes.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ContractError, WINDOW_SAMPLES

NUM_LEADS = 12
NUM_D6_LEADS = 6
F0_CHANNELS = 64
F1_CHANNELS = 128
F2_CHANNELS = 256
F2_TOKENS = 250
MODEL_DIM = 64
FUSION_MODES = ("none", "film", "gated_residual", "film_gated_residual")
CONTEXT_SOURCES = ("watch_ecg", "ecg_machine_d6", "body_scale_d6")


def _check_anchor(anchor_i: torch.Tensor) -> None:
    if anchor_i.ndim != 3 or anchor_i.shape[1:] != (1, WINDOW_SAMPLES):
        raise ContractError(f"anchor_i must have shape [B,1,{WINDOW_SAMPLES}]")


class MultiScaleResidualBlock(nn.Module):
    """Residual local convolutions with configurable dilation/receptive field."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.local = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        kernel = 15
        self.long = nn.Conv1d(channels, channels, kernel, padding=(kernel // 2) * dilation,
                              dilation=dilation, groups=channels)
        self.mix = nn.Conv1d(channels * 2, channels, 1)
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((self.local(values), self.long(values)), dim=1)
        return values + self.activation(self.norm(self.mix(branches)))


class AnchorCNNEncoder(nn.Module):
    """Multi-scale anchor encoder returning F0, F1 and F2 at fixed lengths."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, F0_CHANNELS, 7, padding=3),
                                  nn.GroupNorm(8, F0_CHANNELS), nn.GELU())
        self.f0_block = nn.Sequential(MultiScaleResidualBlock(F0_CHANNELS, 1),
                                      MultiScaleResidualBlock(F0_CHANNELS, 2))
        self.down1 = nn.Sequential(nn.Conv1d(F0_CHANNELS, F1_CHANNELS, 7, stride=4, padding=3),
                                   nn.GroupNorm(8, F1_CHANNELS), nn.GELU())
        self.f1_block = nn.Sequential(MultiScaleResidualBlock(F1_CHANNELS, 1),
                                      MultiScaleResidualBlock(F1_CHANNELS, 2))
        self.down2 = nn.Sequential(nn.Conv1d(F1_CHANNELS, F2_CHANNELS, 7, stride=5, padding=3),
                                   nn.GroupNorm(16, F2_CHANNELS), nn.GELU())
        self.f2_block = nn.Sequential(MultiScaleResidualBlock(F2_CHANNELS, 1),
                                      MultiScaleResidualBlock(F2_CHANNELS, 2))

    def forward(self, anchor_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_anchor(anchor_i)
        f0 = self.f0_block(self.stem(anchor_i))
        f1 = self.f1_block(self.down1(f0))
        f2 = self.f2_block(self.down2(f1))
        if (f0.shape[-1], f1.shape[-1], f2.shape[-1]) != (5000, 1250, F2_TOKENS):
            raise ContractError(f"M1 CNN pyramid has unexpected shapes: {f0.shape}, {f1.shape}, {f2.shape}")
        return f0, f1, f2


class ContextEncoder(nn.Module):
    """Encode one d6 source type with canonical lead and source embeddings."""

    def __init__(self, source_type: str, context_dim: int = MODEL_DIM) -> None:
        super().__init__()
        if source_type not in {"ecg_machine_d6", "body_scale_d6"}:
            raise ContractError("ContextEncoder is only for task-2 d6 sources")
        self.source_type = source_type
        self.channel_stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv1d(32, context_dim, 7, padding=3), nn.GroupNorm(8, context_dim), nn.GELU(),
        )
        self.lead_embedding = nn.Parameter(torch.zeros(NUM_D6_LEADS, context_dim))
        self.source_embedding = nn.Parameter(torch.zeros(context_dim))
        self.projection = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, context_dim),
                                        nn.GELU(), nn.LayerNorm(context_dim))
        nn.init.normal_(self.lead_embedding, std=0.02)
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(self, context: torch.Tensor, lead_mask: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != WINDOW_SAMPLES:
            raise ContractError(f"{self.source_type} context must have shape [B,C,{WINDOW_SAMPLES}]")
        if lead_mask.shape != (context.shape[0], NUM_D6_LEADS):
            raise ContractError("d6 context lead mask must have shape [B,6]")
        if not torch.all(lead_mask.sum(dim=1) == context.shape[1]):
            raise ContractError("d6 lead mask does not match the number of context channels")
        values = self.channel_stem(context.reshape(-1, 1, WINDOW_SAMPLES))
        values = F.adaptive_avg_pool1d(values, 1).reshape(context.shape[0], context.shape[1], -1)
        lead_indices = torch.argsort(lead_mask.to(torch.int64), dim=1)[:, -context.shape[1]:]
        lead_indices = torch.sort(lead_indices, dim=1).values
        values = values + self.lead_embedding[lead_indices]
        return self.projection(values.mean(dim=1) + self.source_embedding)


class WatchContextEncoder(nn.Module):
    """Independent watch encoder; it never receives target-time anchor data."""

    def __init__(self, context_dim: int = MODEL_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.GroupNorm(4, 32), nn.GELU(),
            MultiScaleResidualBlock(32, 2),
            nn.Conv1d(32, context_dim, 7, stride=4, padding=3), nn.GroupNorm(8, context_dim), nn.GELU(),
        )
        self.source_embedding = nn.Parameter(torch.zeros(context_dim))
        self.projection = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, context_dim),
                                        nn.GELU(), nn.LayerNorm(context_dim))
        nn.init.normal_(self.source_embedding, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[1:] != (1, WINDOW_SAMPLES):
            raise ContractError(f"watch context must have shape [B,1,{WINDOW_SAMPLES}]")
        return self.projection(self.encoder(context).mean(dim=-1) + self.source_embedding)


class LeadTimeDecoder(nn.Module):
    """Lead-conditioned FPN decoder from 250 time tokens to 5000 samples."""

    def __init__(self) -> None:
        super().__init__()
        self.lead_embedding = nn.Parameter(torch.zeros(NUM_LEADS, MODEL_DIM))
        self.lead_norm = nn.LayerNorm(MODEL_DIM)
        self.lead_mlp = nn.Sequential(nn.Linear(MODEL_DIM, 128), nn.GELU(), nn.Linear(128, MODEL_DIM))
        self.to_1250 = nn.Conv1d(MODEL_DIM, F1_CHANNELS, 1)
        self.skip_1250 = nn.Conv1d(F1_CHANNELS, F1_CHANNELS, 1)
        self.refine_1250 = nn.Sequential(nn.Conv1d(F1_CHANNELS, F1_CHANNELS, 5, padding=2),
                                         nn.GroupNorm(8, F1_CHANNELS), nn.GELU())
        self.to_5000 = nn.Conv1d(F1_CHANNELS, F0_CHANNELS, 1)
        self.skip_5000 = nn.Conv1d(F0_CHANNELS, F0_CHANNELS, 1)
        self.refine_5000 = nn.Sequential(nn.Conv1d(F0_CHANNELS, F0_CHANNELS, 7, padding=3),
                                         nn.GroupNorm(8, F0_CHANNELS), nn.GELU())
        self.output = nn.Conv1d(F0_CHANNELS, 1, 7, padding=3)
        nn.init.normal_(self.lead_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        lead_tokens = tokens[:, None] + self.lead_embedding[None, :, None, :]
        lead_tokens = lead_tokens + self.lead_mlp(self.lead_norm(lead_tokens))
        values = lead_tokens.permute(0, 1, 3, 2).reshape(batch * NUM_LEADS, MODEL_DIM, F2_TOKENS)
        values = F.interpolate(self.to_1250(values), size=1250, mode="linear", align_corners=False)
        skip1 = self.skip_1250(f1)[:, None].expand(batch, NUM_LEADS, F1_CHANNELS, 1250).reshape_as(values)
        values = self.refine_1250(values + skip1)
        values = F.interpolate(self.to_5000(values), size=5000, mode="linear", align_corners=False)
        skip0 = self.skip_5000(f0)[:, None].expand(batch, NUM_LEADS, F0_CHANNELS, 5000).reshape_as(values)
        high_res = self.refine_5000(values + skip0)
        prediction = self.output(high_res).reshape(batch, NUM_LEADS, WINDOW_SAMPLES)
        return prediction, high_res.reshape(batch, NUM_LEADS, F0_CHANNELS, WINDOW_SAMPLES)


class M1Model(nn.Module):
    """M1-P0/P1 model with a strict, configurable context interface."""

    def __init__(self, *, fusion_mode: str = "none", transformer_layers: int = 4,
                 dropout: float = 0.1, context_dropout: float = 0.0, source_dropout: float = 0.0) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ContractError(f"fusion_mode must be one of {FUSION_MODES}")
        if not 4 <= transformer_layers <= 8:
            raise ContractError("transformer_layers must be between 4 and 8")
        if any(not 0.0 <= value <= 1.0 for value in (context_dropout, source_dropout)):
            raise ContractError("context/source dropout must be in [0,1]")
        self.fusion_mode = fusion_mode
        self.context_dropout = float(context_dropout)
        self.source_dropout = float(source_dropout)
        self.cnn_encoder = AnchorCNNEncoder()
        self.token_projection = nn.Conv1d(F2_CHANNELS, MODEL_DIM, 1)
        layer = nn.TransformerEncoderLayer(d_model=MODEL_DIM, nhead=4, dim_feedforward=256,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.time_transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.register_buffer("time_position", self._sinusoidal(F2_TOKENS, MODEL_DIM), persistent=True)
        self.decoder = LeadTimeDecoder()
        self.watch_context_encoder = WatchContextEncoder(MODEL_DIM)
        self.machine_d6_context_encoder = ContextEncoder("ecg_machine_d6", MODEL_DIM)
        self.body_d6_context_encoder = ContextEncoder("body_scale_d6", MODEL_DIM)
        self.film = nn.Linear(MODEL_DIM, 2 * MODEL_DIM)
        self.gate = nn.Linear(MODEL_DIM, NUM_LEADS)
        self.residual_adapter = nn.Sequential(
            nn.Conv1d(F0_CHANNELS + MODEL_DIM, F0_CHANNELS, 3, padding=1),
            nn.GroupNorm(8, F0_CHANNELS), nn.GELU(), nn.Conv1d(F0_CHANNELS, 1, 3, padding=1),
        )
        self.baseline_head = nn.Sequential(nn.Linear(F2_CHANNELS, 64), nn.GELU(), nn.Linear(64, NUM_LEADS))
        self._initialize_context_adapters()

    @staticmethod
    def _sinusoidal(length: int, dimension: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        return encoding.unsqueeze(0)

    def _initialize_context_adapters(self) -> None:
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, math.log(0.035 / 0.965))
        nn.init.zeros_(self.residual_adapter[-1].weight); nn.init.zeros_(self.residual_adapter[-1].bias)
        nn.init.zeros_(self.baseline_head[-1].weight); nn.init.zeros_(self.baseline_head[-1].bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def anchor_parameter_names(self) -> tuple[str, ...]:
        prefixes = ("watch_context_encoder.", "machine_d6_context_encoder.", "body_d6_context_encoder.",
                    "film.", "gate.", "residual_adapter.")
        return tuple(name for name, _ in self.named_parameters() if not name.startswith(prefixes))

    def _anchor_features(self, anchor_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f0, f1, f2 = self.cnn_encoder(anchor_i)
        tokens = self.token_projection(f2).transpose(1, 2) + self.time_position.to(device=f2.device, dtype=f2.dtype)
        tokens = self.time_transformer(tokens)  # no causal mask: complete 10-second rhythm context
        return f0, f1, f2, tokens

    def _encode_context(self, context_ecg: torch.Tensor, source: str | Sequence[str], lead_mask: torch.Tensor | None) -> torch.Tensor:
        sources = [source] * context_ecg.shape[0] if isinstance(source, str) else list(source)
        if len(sources) != context_ecg.shape[0] or any(item not in CONTEXT_SOURCES for item in sources):
            raise ContractError(f"context_source_type must use {CONTEXT_SOURCES}")
        if sources[0] == "watch_ecg":
            if any(item != "watch_ecg" for item in sources):
                raise ContractError("a batch cannot mix watch and d6 context")
            return self.watch_context_encoder(context_ecg)
        if lead_mask is None or lead_mask.shape != (context_ecg.shape[0], NUM_D6_LEADS):
            raise ContractError("task2 d6 context requires context_lead_mask [B,6]")
        result: list[torch.Tensor | None] = [None] * context_ecg.shape[0]
        for name, encoder in (("ecg_machine_d6", self.machine_d6_context_encoder), ("body_scale_d6", self.body_d6_context_encoder)):
            indices = [i for i, item in enumerate(sources) if item == name]
            if indices:
                encoded = encoder(context_ecg[indices], lead_mask[indices])
                for position, value in zip(indices, encoded):
                    result[position] = value
        if any(value is None for value in result):
            raise ContractError("task2 context must be exactly one machine/body d6 source per sample")
        return torch.stack([value for value in result if value is not None])

    def _apply_context_dropout(self, context: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return context
        if self.context_dropout:
            context = context * (torch.rand(context.shape[0], 1, device=context.device) >= self.context_dropout)
        if self.source_dropout:
            context = context * (torch.rand(context.shape[0], 1, device=context.device) >= self.source_dropout)
        return context

    def forward(self, anchor_i: torch.Tensor, *, context_ecg: torch.Tensor | None = None,
                context_source_type: str | Sequence[str] | None = None,
                context_lead_mask: torch.Tensor | None = None) -> torch.Tensor:
        _check_anchor(anchor_i)
        f0, f1, f2, tokens = self._anchor_features(anchor_i)
        if self.fusion_mode == "none":
            prediction, _ = self.decoder(tokens, f0, f1)
            return prediction
        if context_ecg is None or context_source_type is None:
            raise ContractError(f"fusion_mode={self.fusion_mode} requires context")
        context = self._apply_context_dropout(self._encode_context(context_ecg, context_source_type, context_lead_mask))
        if self.fusion_mode in {"film", "film_gated_residual"}:
            gamma, beta = self.film(context).chunk(2, dim=-1)
            tokens = tokens * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        prediction, high_res = self.decoder(tokens, f0, f1)
        if self.fusion_mode in {"gated_residual", "film_gated_residual"}:
            batch = anchor_i.shape[0]
            context_map = context[:, None, :, None].expand(batch, NUM_LEADS, MODEL_DIM, WINDOW_SAMPLES)
            adapter_input = torch.cat((high_res, context_map), dim=2).reshape(batch * NUM_LEADS,
                                                                                F0_CHANNELS + MODEL_DIM, WINDOW_SAMPLES)
            delta = self.residual_adapter(adapter_input).reshape(batch, NUM_LEADS, WINDOW_SAMPLES)
            gate = torch.sigmoid(self.gate(context))[:, :, None]
            prediction = prediction + gate * delta
        return prediction

    @torch.no_grad()
    def predict_baseline(self, anchor_i: torch.Tensor) -> torch.Tensor:
        """Return model-predicted d12 baseline; true target baseline is absent."""
        _check_anchor(anchor_i)
        _, _, f2 = self.cnn_encoder(anchor_i)
        return self.baseline_head(f2.mean(dim=-1))


M1MaskedCNNLeadTimeTransformer = M1Model
