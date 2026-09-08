"""B2: conservative joint-anchor Patch Transformer.

Context is encoded independently from the target-time machine-I anchor.  No
context waveform is concatenated with the anchor as if their sample clocks
were synchronous.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .contracts import ContractError, WINDOW_SAMPLES


NUM_LEADS, NUM_D6_LEADS = 12, 6


@dataclass(frozen=True)
class B2ModelConfig:
    patch_size: int = 25
    d_model: int = 192
    time_transformer_layers: int = 4
    context_transformer_layers: int = 2
    attention_heads: int = 6
    dropout: float = 0.10
    fusion_mode: str = "none"
    context_dropout: float = 0.10
    initial_gate: float = 0.03

    def validate(self) -> None:
        if WINDOW_SAMPLES % self.patch_size or self.d_model % self.attention_heads:
            raise ValueError("patch_size must divide 5000 and d_model must divide attention_heads")
        if self.fusion_mode not in {"none", "film", "film_gated_residual"}:
            raise ValueError("fusion_mode must be none, film, or film_gated_residual")
        if not 0.0 <= self.context_dropout < 1.0 or not 0.0 < self.initial_gate < 1.0:
            raise ValueError("invalid context_dropout or initial_gate")


def _position_encoding(tokens: int, d_model: int) -> torch.Tensor:
    position = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(tokens, d_model, dtype=torch.float32)
    encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
    return encoding.unsqueeze(0)


class _TimePatchEncoder(nn.Module):
    def __init__(self, channels: int, config: B2ModelConfig, layers: int) -> None:
        super().__init__()
        self.patch = nn.Conv1d(channels, config.d_model, config.patch_size, config.patch_size)
        block = nn.TransformerEncoderLayer(config.d_model, config.attention_heads, config.d_model * 2,
                                           config.dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers)

    def forward(self, values: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.patch(values).transpose(1, 2) + position)


class WatchContextEncoder(nn.Module):
    """Independent watch-I(A) encoder; produces only a global condition."""
    def __init__(self, config: B2ModelConfig) -> None:
        super().__init__()
        self.encoder = _TimePatchEncoder(1, config, config.context_transformer_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, watch: torch.Tensor, available: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if watch.ndim != 3 or watch.shape[1:] != (1, WINDOW_SAMPLES):
            raise ContractError("watch context must have shape [B,1,5000]")
        return self.norm(self.encoder(watch, position).mean(dim=1)) * available.to(watch.dtype).unsqueeze(1)


class D6ContextEncoder(nn.Module):
    """Source-specific canonical-d6 encoder with lead and source embeddings."""
    def __init__(self, config: B2ModelConfig, source_index: int) -> None:
        super().__init__()
        self.encoder = _TimePatchEncoder(NUM_D6_LEADS, config, config.context_transformer_layers)
        self.d6_lead_embedding = nn.Embedding(NUM_D6_LEADS, config.d_model)
        self.source_embedding = nn.Embedding(2, config.d_model)
        self.source_index = source_index
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, d6: torch.Tensor, lead_mask: torch.Tensor, available: torch.Tensor,
                position: torch.Tensor) -> torch.Tensor:
        if d6.ndim != 3 or d6.shape[1:] != (NUM_D6_LEADS, WINDOW_SAMPLES):
            raise ContractError("d6 context must be canonical [B,6,5000]")
        if lead_mask.shape != (d6.shape[0], NUM_D6_LEADS):
            raise ContractError("d6 context_lead_mask must be [B,6]")
        lead_ids = torch.arange(NUM_D6_LEADS, device=d6.device)
        lead_condition = (self.d6_lead_embedding(lead_ids).unsqueeze(0) * lead_mask.to(d6.dtype).unsqueeze(-1)).sum(1)
        lead_condition = lead_condition / lead_mask.sum(1, keepdim=True).clamp_min(1).to(d6.dtype)
        source_condition = self.source_embedding(torch.full((d6.shape[0],), self.source_index, device=d6.device))
        z = self.norm(self.encoder(d6, position).mean(1) + lead_condition + source_condition)
        return z * available.to(d6.dtype).unsqueeze(1)


class B2JointAnchorPatchTransformer(nn.Module):
    """P0 anchor backbone plus C1/C2 global-latent context conditioning."""
    checkpoint_schema = "b2_joint_anchor_patch_v2"

    def __init__(self, config: B2ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or B2ModelConfig()
        self.config.validate()
        self.num_tokens = WINDOW_SAMPLES // self.config.patch_size
        self.anchor_encoder = _TimePatchEncoder(1, self.config, self.config.time_transformer_layers)
        self.watch_encoder = WatchContextEncoder(self.config)
        self.machine_d6_encoder = D6ContextEncoder(self.config, source_index=0)
        self.body_d6_encoder = D6ContextEncoder(self.config, source_index=1)
        self.task2_latent_fuser = nn.Sequential(nn.Linear(self.config.d_model * 2 + 2, self.config.d_model), nn.GELU(), nn.Linear(self.config.d_model, self.config.d_model))
        self.film = nn.Sequential(nn.Linear(self.config.d_model, self.config.d_model * 2))
        self.residual = nn.Sequential(nn.Linear(self.config.d_model * 2, self.config.d_model), nn.GELU(), nn.Linear(self.config.d_model, self.config.d_model))
        self.gate = nn.Linear(self.config.d_model, self.config.d_model)
        self.lead_embedding = nn.Embedding(NUM_LEADS, self.config.d_model)
        self.decoder_norm = nn.LayerNorm(self.config.d_model)
        self.lead_decoder = nn.Linear(self.config.d_model, self.config.patch_size)
        self.register_buffer("anchor_position_encoding", _position_encoding(self.num_tokens, self.config.d_model), persistent=True)
        nn.init.zeros_(self.film[0].weight); nn.init.zeros_(self.film[0].bias)
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, math.log(self.config.initial_gate / (1 - self.config.initial_gate)))

    @property
    def parameter_count(self) -> int:
        return sum(item.numel() for item in self.parameters())

    def _context_latent(self, *, task_id: str, watch_context: torch.Tensor | None,
                        watch_available: torch.Tensor | None, machine_d6_context: torch.Tensor | None,
                        machine_d6_mask: torch.Tensor | None, machine_available: torch.Tensor | None,
                        body_d6_context: torch.Tensor | None, body_d6_mask: torch.Tensor | None,
                        body_available: torch.Tensor | None, batch: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
        zeros = torch.zeros((batch, self.config.d_model), device=device, dtype=dtype)
        if task_id == "task1":
            if watch_context is None or watch_available is None:
                return zeros
            return self.watch_encoder(watch_context, watch_available, self.anchor_position_encoding.to(device=device, dtype=dtype))
        if task_id != "task2":
            raise ContractError("task_id must be task1 or task2")
        machine_available = torch.zeros(batch, dtype=torch.bool, device=device) if machine_available is None else machine_available.to(device=device, dtype=torch.bool)
        body_available = torch.zeros(batch, dtype=torch.bool, device=device) if body_available is None else body_available.to(device=device, dtype=torch.bool)
        z_machine = zeros if machine_d6_context is None or machine_d6_mask is None else self.machine_d6_encoder(machine_d6_context, machine_d6_mask, machine_available, self.anchor_position_encoding.to(device=device, dtype=dtype))
        z_body = zeros if body_d6_context is None or body_d6_mask is None else self.body_d6_encoder(body_d6_context, body_d6_mask, body_available, self.anchor_position_encoding.to(device=device, dtype=dtype))
        features = torch.cat((z_machine, z_body, machine_available.to(dtype).unsqueeze(1), body_available.to(dtype).unsqueeze(1)), dim=1)
        return self.task2_latent_fuser(features) * (machine_available | body_available).to(dtype).unsqueeze(1)

    def forward(self, anchor_i_ecg: torch.Tensor, anchor_lead_mask: torch.Tensor, *, task_id: str,
                watch_context: torch.Tensor | None = None, watch_available: torch.Tensor | None = None,
                machine_d6_context: torch.Tensor | None = None, machine_d6_mask: torch.Tensor | None = None,
                machine_available: torch.Tensor | None = None, body_d6_context: torch.Tensor | None = None,
                body_d6_mask: torch.Tensor | None = None, body_available: torch.Tensor | None = None) -> torch.Tensor:
        if anchor_i_ecg.ndim != 3 or anchor_i_ecg.shape[1:] != (1, WINDOW_SAMPLES):
            raise ContractError("anchor_i_ecg must have shape [B,1,5000]")
        expected_anchor = torch.zeros((anchor_i_ecg.shape[0], NUM_LEADS), device=anchor_i_ecg.device, dtype=torch.bool); expected_anchor[:, 0] = True
        if anchor_lead_mask.shape != expected_anchor.shape or not torch.equal(anchor_lead_mask.to(device=anchor_i_ecg.device, dtype=torch.bool), expected_anchor):
            raise ContractError("anchor_lead_mask must expose target-time I only")
        position = self.anchor_position_encoding.to(device=anchor_i_ecg.device, dtype=anchor_i_ecg.dtype)
        h_anchor = self.anchor_encoder(anchor_i_ecg, position)
        if self.config.fusion_mode == "none":
            h_joint = h_anchor
        else:
            z = self._context_latent(task_id=task_id, watch_context=watch_context, watch_available=watch_available,
                machine_d6_context=machine_d6_context, machine_d6_mask=machine_d6_mask, machine_available=machine_available,
                body_d6_context=body_d6_context, body_d6_mask=body_d6_mask, body_available=body_available,
                batch=anchor_i_ecg.shape[0], device=anchor_i_ecg.device, dtype=anchor_i_ecg.dtype)
            if self.training and self.config.context_dropout:
                z = z * (torch.rand((z.shape[0], 1), device=z.device) >= self.config.context_dropout).to(z.dtype)
            gamma, beta = self.film(z).chunk(2, dim=1)
            h_film = h_anchor * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            if self.config.fusion_mode == "film": h_joint = h_film
            else:
                expanded = z.unsqueeze(1).expand(-1, h_film.shape[1], -1)
                h_joint = h_film + torch.sigmoid(self.gate(z)).unsqueeze(1) * self.residual(torch.cat((h_film, expanded), dim=-1))
        lead_ids = torch.arange(NUM_LEADS, device=anchor_i_ecg.device)
        decoded = self.lead_decoder(self.decoder_norm(h_joint).unsqueeze(1) + self.lead_embedding(lead_ids).view(1, NUM_LEADS, 1, -1))
        # ``decoded`` is [B, leads, tokens, patch_size].  The token dimension
        # already follows chronological order, so flatten tokens first and
        # patch samples second.  Permuting to [patch_size, tokens] would
        # interleave distant time points and scramble every output waveform.
        return decoded.reshape(anchor_i_ecg.shape[0], NUM_LEADS, WINDOW_SAMPLES)


B2MaskedPatchTransformer = B2JointAnchorPatchTransformer
