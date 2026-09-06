"""B2-v1 masked patch transformer for canonical twelve-lead ECG."""
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


def sinusoidal_position_encoding(length: int, dimension: int) -> torch.Tensor:
    """Return the fixed [1, length, dimension] sinusoidal encoding."""
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding.unsqueeze(0)


def _canonicalize_torch_input(ecg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the shared numpy adapter for raw 1/6-lead tensors."""
    if ecg.ndim != 3 or ecg.shape[-1] != WINDOW_SAMPLES:
        raise ContractError(f"ECG must have shape [B,C,{WINDOW_SAMPLES}]")
    if ecg.shape[1] == NUM_LEADS:
        return ecg, None
    canonical, mask = canonicalize_input_ecg(ecg.detach().cpu().numpy())
    return torch.as_tensor(canonical, dtype=ecg.dtype, device=ecg.device), torch.as_tensor(mask, dtype=torch.bool, device=ecg.device)


class B2MaskedPatchTransformer(nn.Module):
    """Fixed B2-v1 architecture with a time-wise lead-mask condition."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv1d(24, MODEL_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=4,
            dim_feedforward=192,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.decoder_norm = nn.LayerNorm(MODEL_DIM)
        self.decoder = nn.Linear(MODEL_DIM, NUM_LEADS * PATCH_SIZE)
        self.register_buffer("positional_encoding", sinusoidal_position_encoding(NUM_PATCHES, MODEL_DIM), persistent=True)

        parameter_count = self.parameter_count
        if not 250_000 <= parameter_count <= 400_000:
            raise RuntimeError(f"B2-v1 parameter count is outside the expected range: {parameter_count}")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        ecg: torch.Tensor,
        lead_mask: torch.Tensor,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct all twelve leads in centered_scaled_d12 model space."""
        canonical_ecg, inferred_mask = _canonicalize_torch_input(ecg)
        if lead_mask.shape != (canonical_ecg.shape[0], NUM_LEADS):
            raise ContractError("lead_mask must have shape [B, 12]")
        lead_mask = lead_mask.to(device=canonical_ecg.device, dtype=torch.bool)
        if inferred_mask is not None and not torch.equal(lead_mask, inferred_mask):
            raise ContractError("lead_mask does not describe the canonicalized ECG input")
        if missing_mask is not None:
            missing_mask = missing_mask.to(device=canonical_ecg.device, dtype=torch.bool)
            if missing_mask.shape != lead_mask.shape or not torch.equal(missing_mask, ~lead_mask):
                raise ContractError("missing_mask must be the complement of lead_mask")

        time_mask = lead_mask.to(dtype=canonical_ecg.dtype).unsqueeze(-1).expand(-1, -1, WINDOW_SAMPLES)
        model_input = torch.cat((canonical_ecg, time_mask), dim=1)
        tokens = self.patch_embedding(model_input).transpose(1, 2)
        tokens = tokens + self.positional_encoding.to(dtype=tokens.dtype, device=tokens.device)
        tokens = self.encoder(tokens)
        patches = self.decoder(self.decoder_norm(tokens))
        patches = patches.reshape(patches.shape[0], NUM_PATCHES, NUM_LEADS, PATCH_SIZE)
        return patches.permute(0, 2, 1, 3).reshape(patches.shape[0], NUM_LEADS, WINDOW_SAMPLES)


def build_b2_input(ecg: torch.Tensor, lead_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize a raw batch and return the model input plus missing mask."""
    canonical, inferred_mask = _canonicalize_torch_input(ecg)
    mask = lead_mask.to(device=canonical.device, dtype=torch.bool)
    if inferred_mask is not None and (mask.shape != inferred_mask.shape or not torch.equal(mask, inferred_mask)):
        raise ContractError("lead_mask does not match the raw ECG channel count")
    return canonical, ~mask
