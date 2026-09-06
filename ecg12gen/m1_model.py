"""M1-main: shared CNN plus factorized lead/time Transformer for 12-lead ECG."""
from __future__ import annotations

import math

import torch
from torch import nn

from .canonical_adapter import canonicalize_input_ecg
from .contracts import ContractError, WINDOW_SAMPLES


PATCH_SIZE = 25
NUM_PATCHES = WINDOW_SAMPLES // PATCH_SIZE
MODEL_DIM = 96
NUM_LEADS = 12


def sinusoidal_time_encoding(length: int, dimension: int) -> torch.Tensor:
    """Return fixed time-position features with shape ``[1, 1, length, dim]``."""
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding.unsqueeze(0).unsqueeze(0)


def _canonicalize_torch_input(ecg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the frozen shared adapter for raw one- and six-lead inputs."""
    if ecg.ndim != 3 or ecg.shape[-1] != WINDOW_SAMPLES:
        raise ContractError(f"ECG must have shape [B,C,{WINDOW_SAMPLES}]")
    if ecg.shape[1] == NUM_LEADS:
        return ecg, None
    canonical, mask = canonicalize_input_ecg(ecg.detach().cpu().numpy())
    return (
        torch.as_tensor(canonical, dtype=ecg.dtype, device=ecg.device),
        torch.as_tensor(mask, dtype=torch.bool, device=ecg.device),
    )


class SharedMultiScaleResidualBlock(nn.Module):
    """A per-lead shared local-morphology block with short and long ECG views."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.short = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.long = nn.Conv1d(in_channels, in_channels, kernel_size=15, padding=14, dilation=2, groups=in_channels)
        self.fuse = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        fused = self.fuse(torch.cat((self.short(values), self.long(values)), dim=1))
        return self.activation(self.norm(fused) + residual)


class SharedLeadCNNEncoder(nn.Module):
    """One CNN parameter set applied independently to every physical lead."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.block1 = SharedMultiScaleResidualBlock(32, 64)
        self.block2 = SharedMultiScaleResidualBlock(64, 64)
        self.patch_projection = nn.Conv1d(64, MODEL_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, canonical_ecg: torch.Tensor) -> torch.Tensor:
        if canonical_ecg.ndim != 3 or canonical_ecg.shape[1:] != (NUM_LEADS, WINDOW_SAMPLES):
            raise ContractError(f"canonical ECG must have shape [B,{NUM_LEADS},{WINDOW_SAMPLES}]")
        batch = canonical_ecg.shape[0]
        values = canonical_ecg.reshape(batch * NUM_LEADS, 1, WINDOW_SAMPLES)
        values = self.patch_projection(self.block2(self.block1(self.stem(values))))
        return values.transpose(1, 2).reshape(batch, NUM_LEADS, NUM_PATCHES, MODEL_DIM)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class LeadTimeTransformerBlock(nn.Module):
    """Pre-norm temporal attention followed by pre-norm physical-lead attention."""

    def __init__(self, dimension: int = MODEL_DIM, heads: int = 4, ffn_dimension: int = 192, dropout: float = 0.10) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dimension)
        self.temporal_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.temporal_ffn_norm = nn.LayerNorm(dimension)
        self.temporal_ffn = FeedForward(dimension, ffn_dimension, dropout)
        self.lead_norm = nn.LayerNorm(dimension)
        self.lead_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.lead_ffn_norm = nn.LayerNorm(dimension)
        self.lead_ffn = FeedForward(dimension, ffn_dimension, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1:] != (NUM_LEADS, NUM_PATCHES, MODEL_DIM):
            raise ContractError(f"Lead-time tokens must have shape [B,{NUM_LEADS},{NUM_PATCHES},{MODEL_DIM}]")
        batch = values.shape[0]

        temporal = values.reshape(batch * NUM_LEADS, NUM_PATCHES, MODEL_DIM)
        normalized = self.temporal_norm(temporal)
        attended, _ = self.temporal_attention(normalized, normalized, normalized, need_weights=False)
        temporal = temporal + self.dropout(attended)
        temporal = temporal + self.temporal_ffn(self.temporal_ffn_norm(temporal))
        values = temporal.reshape(batch, NUM_LEADS, NUM_PATCHES, MODEL_DIM)

        leadwise = values.permute(0, 2, 1, 3).reshape(batch * NUM_PATCHES, NUM_LEADS, MODEL_DIM)
        normalized = self.lead_norm(leadwise)
        attended, _ = self.lead_attention(normalized, normalized, normalized, need_weights=False)
        leadwise = leadwise + self.dropout(attended)
        leadwise = leadwise + self.lead_ffn(self.lead_ffn_norm(leadwise))
        return leadwise.reshape(batch, NUM_PATCHES, NUM_LEADS, MODEL_DIM).permute(0, 2, 1, 3)


class SharedRefinement(nn.Module):
    """A shallow shared per-lead residual smoother for patch-boundary artifacts."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        batch = generated.shape[0]
        residual = self.layers(generated.reshape(batch * NUM_LEADS, 1, WINDOW_SAMPLES))
        return generated + residual.reshape(batch, NUM_LEADS, WINDOW_SAMPLES)


class M1MaskedCNNLeadTimeTransformer(nn.Module):
    """M1-no-adapter/no-baseline-head shared backbone for task1 and task2."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn_encoder = SharedLeadCNNEncoder()
        self.missing_token = nn.Parameter(torch.empty(1, 1, 1, MODEL_DIM))
        self.lead_embedding = nn.Parameter(torch.empty(NUM_LEADS, MODEL_DIM))
        self.register_buffer("time_positional_encoding", sinusoidal_time_encoding(NUM_PATCHES, MODEL_DIM), persistent=True)
        self.blocks = nn.ModuleList([LeadTimeTransformerBlock() for _ in range(3)])
        self.decoder = nn.Linear(MODEL_DIM, PATCH_SIZE)
        self.refinement = SharedRefinement()
        nn.init.normal_(self.missing_token, mean=0.0, std=0.02)
        nn.init.normal_(self.lead_embedding, mean=0.0, std=0.02)

        parameter_count = self.parameter_count
        if not 600_000 <= parameter_count <= 850_000:
            raise RuntimeError(
                "M1 parameter count must be in [600000, 850000], "
                f"but the configured model has {parameter_count:,} parameters."
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        ecg: torch.Tensor,
        lead_mask: torch.Tensor,
        missing_mask: torch.Tensor | None = None,
        observed_d12_model: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate missing leads and exactly retain observed d12-model-space leads.

        ``observed_d12_model`` is required by the training entry point because
        raw device input and canonical d12 can have different frozen scales.
        Its fallback is only for already-canonicalized compatible callers.
        """
        canonical_ecg, inferred_mask = _canonicalize_torch_input(ecg)
        expected_mask_shape = (canonical_ecg.shape[0], NUM_LEADS)
        if lead_mask.shape != expected_mask_shape:
            raise ContractError(f"lead_mask must have shape [B,{NUM_LEADS}]")
        lead_mask = lead_mask.to(device=canonical_ecg.device, dtype=torch.bool)
        if inferred_mask is not None and not torch.equal(lead_mask, inferred_mask):
            raise ContractError("lead_mask does not describe the canonicalized ECG input")
        if missing_mask is None:
            missing_mask = ~lead_mask
        else:
            missing_mask = missing_mask.to(device=canonical_ecg.device, dtype=torch.bool)
            if missing_mask.shape != lead_mask.shape or not torch.equal(missing_mask, ~lead_mask):
                raise ContractError("missing_mask must be the complement of lead_mask")

        if observed_d12_model is None:
            observed = canonical_ecg
        else:
            observed = observed_d12_model.to(device=canonical_ecg.device, dtype=canonical_ecg.dtype)
            if observed.shape != canonical_ecg.shape:
                raise ContractError(f"observed_d12_model must have shape [B,{NUM_LEADS},{WINDOW_SAMPLES}]")

        cnn_tokens = self.cnn_encoder(canonical_ecg)
        missing_tokens = self.missing_token.expand(canonical_ecg.shape[0], NUM_LEADS, NUM_PATCHES, MODEL_DIM)
        values = torch.where(lead_mask[:, :, None, None], cnn_tokens, missing_tokens)
        values = values + self.lead_embedding[None, :, None, :] + self.time_positional_encoding.to(values)
        for block in self.blocks:
            values = block(values)

        generated = self.decoder(values).reshape(canonical_ecg.shape[0], NUM_LEADS, WINDOW_SAMPLES)
        generated = self.refinement(generated)
        return torch.where(lead_mask[:, :, None], observed, generated)
