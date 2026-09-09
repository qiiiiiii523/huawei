"""B2-v1 masked patch Transformer with a joint-anchor interface.

The parameterized network is intentionally the original B2-v1 network.  The
new data contract is handled at its boundary: target-time machine-I is the
anchor stream, while cross-time context is encoded independently with the
same frozen-shape patch/Transformer stack and fused at representation level.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .canonical_adapter import canonicalize_input_ecg
from .contracts import ContractError, WINDOW_SAMPLES

PATCH_SIZE = 25
NUM_PATCHES = WINDOW_SAMPLES // PATCH_SIZE
MODEL_DIM = 96
NUM_LEADS = 12
NUM_D6_LEADS = 6
ARCHITECTURE_ID = "B2MaskedPatchTransformer"


@dataclass(frozen=True)
class B2ModelConfig:
    patch_size: int = PATCH_SIZE
    d_model: int = MODEL_DIM
    transformer_layers: int = 3
    attention_heads: int = 4
    feedforward_dim: int = 192
    dropout: float = 0.10
    fusion_mode: str = "none"
    context_dropout: float = 0.0

    def validate(self) -> None:
        if self.patch_size != PATCH_SIZE or WINDOW_SAMPLES % self.patch_size:
            raise ValueError("B2-v1 requires patch_size=25 and 5000 divisible patches")
        if self.d_model != MODEL_DIM or self.d_model % self.attention_heads:
            raise ValueError("B2-v1 model dimensions are fixed")
        if self.transformer_layers != 3 or self.feedforward_dim != 192:
            raise ValueError("B2-v1 transformer structure is fixed")
        if self.fusion_mode not in {"none", "film", "gated_residual", "film_gated_residual"}:
            raise ValueError("unknown B2 fusion mode")
        if not 0.0 <= self.context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0,1)")


def architecture_metadata(config: B2ModelConfig | Mapping[str, Any] | None = None) -> dict[str, str]:
    values = asdict(config or B2ModelConfig()) if not isinstance(config, Mapping) else dict(config)
    structural = {"architecture_id": ARCHITECTURE_ID, "window_samples": WINDOW_SAMPLES,
                  "patch_size": int(values.get("patch_size", PATCH_SIZE)),
                  "d_model": int(values.get("d_model", MODEL_DIM)),
                  "transformer_layers": int(values.get("transformer_layers", 3)),
                  "attention_heads": int(values.get("attention_heads", 4)),
                  "feedforward_dim": int(values.get("feedforward_dim", 192))}
    encoded = json.dumps(structural, sort_keys=True, separators=(",", ":")).encode()
    return {"architecture_id": ARCHITECTURE_ID,
            "architecture_config_hash": hashlib.sha256(encoded).hexdigest()}


def sinusoidal_position_encoding(length: int, dimension: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding.unsqueeze(0)


def _canonical_with_mask(ecg: torch.Tensor, lead_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if ecg.ndim != 3 or ecg.shape[-1] != WINDOW_SAMPLES:
        raise ContractError(f"ECG must have shape [B,C,{WINDOW_SAMPLES}]")
    if lead_mask.ndim != 2 or lead_mask.shape[0] != ecg.shape[0]:
        raise ContractError("lead mask must have shape [B,C] or [B,12]")
    mask = lead_mask.to(device=ecg.device, dtype=torch.bool)
    if mask.shape[1] == NUM_LEADS:
        if mask.sum(dim=1).tolist() != [ecg.shape[1]] * ecg.shape[0]:
            raise ContractError("lead mask does not match ECG channels")
        canonical = torch.zeros((ecg.shape[0], NUM_LEADS, WINDOW_SAMPLES), dtype=ecg.dtype, device=ecg.device)
        canonical[mask] = ecg.reshape(-1, WINDOW_SAMPLES)
        return canonical, mask
    if mask.shape[1] != NUM_D6_LEADS or mask.sum(dim=1).tolist() != [ecg.shape[1]] * ecg.shape[0]:
        raise ContractError("context lead mask must be [B,6] and match channels")
    canonical = torch.zeros((ecg.shape[0], NUM_LEADS, WINDOW_SAMPLES), dtype=ecg.dtype, device=ecg.device)
    canonical[:, :NUM_D6_LEADS][mask] = ecg.reshape(-1, WINDOW_SAMPLES)
    full_mask = torch.zeros((ecg.shape[0], NUM_LEADS), dtype=torch.bool, device=ecg.device)
    full_mask[:, :NUM_D6_LEADS] = mask
    return canonical, full_mask


class B2MaskedPatchTransformer(nn.Module):
    """Original B2-v1 patch embedding, 3-layer encoder and d12 decoder.

    ``forward(context, anchor, context_mask, anchor_mask)`` is the current
    interface.  The old ``forward(ecg, lead_mask, missing_mask)`` form remains
    accepted for checkpoint/smoke compatibility and means anchor-only mode.
    """

    checkpoint_schema = "b2_v1_joint_anchor"

    def __init__(self, config: B2ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or B2ModelConfig()
        self.config.validate()
        self.patch_embedding = nn.Conv1d(24, MODEL_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM, nhead=4, dim_feedforward=192, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.decoder_norm = nn.LayerNorm(MODEL_DIM)
        self.decoder = nn.Linear(MODEL_DIM, NUM_LEADS * PATCH_SIZE)
        self.register_buffer("positional_encoding", sinusoidal_position_encoding(NUM_PATCHES, MODEL_DIM), persistent=True)
        if not 250_000 <= self.parameter_count <= 400_000:
            raise RuntimeError(f"B2-v1 parameter count is outside the expected range: {self.parameter_count}")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def architecture_id(self) -> str:
        return ARCHITECTURE_ID

    @property
    def architecture_config_hash(self) -> str:
        return architecture_metadata(self.config)["architecture_config_hash"]

    def _encode(self, ecg: torch.Tensor, lead_mask: torch.Tensor) -> torch.Tensor:
        canonical, full_mask = _canonical_with_mask(ecg, lead_mask)
        time_mask = full_mask.to(dtype=canonical.dtype).unsqueeze(-1).expand(-1, -1, WINDOW_SAMPLES)
        tokens = self.patch_embedding(torch.cat((canonical, time_mask), dim=1)).transpose(1, 2)
        tokens = tokens + self.positional_encoding.to(dtype=tokens.dtype, device=tokens.device)
        return self.encoder(tokens)

    def forward_anchor(self, anchor_i_ecg: torch.Tensor, anchor_lead_mask: torch.Tensor) -> torch.Tensor:
        return self._decode(self._encode(anchor_i_ecg, anchor_lead_mask))

    def forward_joint(self, context_ecg: torch.Tensor, anchor_i_ecg: torch.Tensor,
                      context_lead_mask: torch.Tensor, anchor_lead_mask: torch.Tensor) -> torch.Tensor:
        anchor_tokens = self._encode(anchor_i_ecg, anchor_lead_mask)
        context_tokens = self._encode(context_ecg, context_lead_mask)
        if context_tokens.shape != anchor_tokens.shape:
            raise ContractError("context and anchor token shapes must match")
        if self.training and self.config.context_dropout:
            keep = (torch.rand((context_tokens.shape[0], 1, 1), device=context_tokens.device)
                    >= self.config.context_dropout).to(context_tokens.dtype)
            context_tokens = context_tokens * keep
        if self.config.fusion_mode == "none":
            fused = anchor_tokens
        elif self.config.fusion_mode == "gated_residual":
            fused = anchor_tokens + 0.05 * context_tokens
        else:
            # The original B2-v1 has no additional adapter parameters.  Keep
            # its structure/checkpoint schema intact and use representation
            # addition for all context-enabled protocol stages.
            fused = anchor_tokens + context_tokens
        return self._decode(fused)

    def _decode(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = self.decoder(self.decoder_norm(tokens))
        patches = patches.reshape(patches.shape[0], NUM_PATCHES, NUM_LEADS, PATCH_SIZE)
        return patches.permute(0, 2, 1, 3).reshape(patches.shape[0], NUM_LEADS, WINDOW_SAMPLES)

    def forward(self, context_ecg: torch.Tensor | None = None,
                anchor_i_ecg: torch.Tensor | None = None,
                context_lead_mask: torch.Tensor | None = None,
                anchor_lead_mask: torch.Tensor | None = None,
                missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Legacy call: model(ecg, lead_mask, missing_mask).
        if anchor_i_ecg is not None and anchor_i_ecg.ndim == 2:
            legacy_ecg, legacy_mask = context_ecg, anchor_i_ecg
            if legacy_ecg is None:
                raise ContractError("legacy ECG input is required")
            return self.forward_anchor(legacy_ecg, legacy_mask)
        if anchor_i_ecg is None:
            if context_ecg is None or context_lead_mask is None:
                raise ContractError("anchor_i_ecg and anchor_lead_mask are required")
            return self.forward_anchor(context_ecg, context_lead_mask)
        if anchor_lead_mask is None:
            raise ContractError("anchor_lead_mask is required")
        if context_ecg is None or context_lead_mask is None or self.config.fusion_mode == "none":
            return self.forward_anchor(anchor_i_ecg, anchor_lead_mask)
        return self.forward_joint(context_ecg, anchor_i_ecg, context_lead_mask, anchor_lead_mask)


# Name used by the later joint-anchor baseline tooling; it is the same B2-v1
# network, not a replacement architecture.
B2JointAnchorPatchTransformer = B2MaskedPatchTransformer


def build_b2_input(ecg: torch.Tensor, lead_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible input validator returning canonical ECG/missing mask."""
    canonical, mask = _canonical_with_mask(ecg, lead_mask)
    return canonical, ~mask
