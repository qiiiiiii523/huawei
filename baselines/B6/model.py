"""M1-inspired conditional velocity model: local CNN + axial attention + FPN."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from baselines.B5.conditions import MetadataEncoder, TimeEmbedding
from .components import AxialLeadTimeBlock, ConvNormAct, LeadTimeDecoder, MultiScaleCNNEncoder
from .config import ModelConfig
from .resampling import resize_linear_1d


@dataclass
class ConditionCache:
    anchor_features: tuple[torch.Tensor, ...]
    anchor_tokens: torch.Tensor
    metadata: torch.Tensor
    anchor_prediction: torch.Tensor
    anchor: torch.Tensor


class TokenCondition(nn.Module):
    def __init__(self, d: int, time_dim: int, metadata_dim: int):
        super().__init__()
        self.time = nn.Linear(time_dim, d)
        self.film = nn.Linear(metadata_dim, d * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, grid: torch.Tensor, time: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(metadata).chunk(2, dim=-1)
        return grid * (1 + gamma[:, None, None]) + beta[:, None, None] + self.time(time)[:, None, None]


class B6AxialFlow(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        d = config.d_model
        self.anchor_encoder = MultiScaleCNNEncoder(config.cnn_channels, config.dropout)
        self.anchor_projection = nn.Linear(config.cnn_channels[-1], d)
        self.time_position = nn.Parameter(torch.randn(1, config.time_tokens, d) * .02)
        self.lead_embedding = nn.Parameter(torch.randn(12, d) * .02)
        self.lead_state_embedding = nn.Parameter(torch.randn(2, d) * .02)
        observed_state = torch.ones(12, dtype=torch.long)
        observed_state[0] = 0
        self.register_buffer('observed_state', observed_state, persistent=False)
        self.time_embedding = TimeEmbedding(config.time_dim)
        self.metadata_encoder = MetadataEncoder(config.metadata_dim, config.metadata_enabled,
                                                config.metadata_dropout, config.metadata_field_dropout)
        self.state_encoder = nn.Sequential(ConvNormAct(1, config.state_channels[0], 7, stride=4),
                                           ConvNormAct(config.state_channels[0], config.state_channels[1], 5, stride=5))
        self.state_projection = nn.Linear(config.state_channels[-1], d)
        self.axial_blocks = nn.ModuleList([AxialLeadTimeBlock(d, config.num_heads, config.ffn_dim,
                                                            config.dropout, config.attention_axes)
                                          for _ in range(config.num_blocks)])
        self.conditioners = nn.ModuleList([TokenCondition(d, config.time_dim, config.metadata_dim)
                                          for _ in range(config.num_blocks)])
        self.final_norm = nn.LayerNorm(d)
        self.decoder = LeadTimeDecoder(d, config.cnn_channels, config.decoder_channels, config.dropout)
        fine = config.state_fine_channels
        # A full-resolution state path is essential: high-frequency Gaussian noise
        # cannot be represented by the 250-token branch alone.
        self.local_state = nn.Sequential(ConvNormAct(2, fine, 5), nn.Conv1d(fine, fine, 3, padding=1), nn.GELU())
        self.local_condition = nn.Linear(config.time_dim + d, fine)
        self.local_head = nn.Conv1d(fine, 1, 3, padding=1)
        # Preserve absolute state level through the normalized CNN paths too.
        # This is a learnable velocity term, not a target/I replacement.
        self.raw_state_skip = nn.Conv1d(11, 11, 1, groups=11, bias=False)
        nn.init.zeros_(self.raw_state_skip.weight)
        c0 = config.cnn_channels[0]
        self.anchor_head = nn.Sequential(nn.Conv1d(c0, c0, 3, padding=1), nn.GELU(), nn.Conv1d(c0, 1, 1))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_conditions(self, anchor, numeric, sex, field_mask, age_topcoded) -> ConditionCache:
        if anchor.ndim != 3 or anchor.shape[1] != 1 or anchor.shape[-1] < 20 or not torch.isfinite(anchor).all():
            raise ValueError('B6 anchor must be finite [B,1,T], T>=20; canonical windows have 5000 samples')
        f0, f1, f2 = self.anchor_encoder(anchor)
        length = f2.shape[-1]
        position = self.time_position
        if length != self.config.time_tokens:
            # Unit tests can use shorter sequences; prediction pads real windows to 5000.
            position = resize_linear_1d(position.transpose(1, 2), length).transpose(1, 2)
        tokens = self.anchor_projection(f2.transpose(1, 2)) + position
        metadata = self.metadata_encoder(numeric, sex, field_mask, age_topcoded)
        if metadata.shape[0] != anchor.shape[0]:
            raise ValueError('B6 anchor and demographics batch sizes differ')
        return ConditionCache((f0, f1, f2), tokens, metadata, self.anchor_head(f0), anchor)

    def velocity(self, state: torch.Tensor, time: torch.Tensor, condition: ConditionCache) -> torch.Tensor:
        anchor = condition.anchor
        batch, _, length = anchor.shape
        if state.shape != (batch, 11, length) or time.shape != (batch,):
            raise ValueError('B6 FM state/time shape mismatch')
        if not torch.isfinite(state).all() or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError('B6 FM state/time must be finite, with time in [0,1]')
        step = self.time_embedding(time)
        state_tokens = self.state_encoder(state.reshape(batch * 11, 1, length))
        state_tokens = self.state_projection(state_tokens.transpose(1, 2)).reshape(batch, 11, -1, self.config.d_model)
        # I slot receives only visible I. Missing slots receive x_t, never hidden y.
        state_grid = torch.cat((torch.zeros_like(state_tokens[:, :1]), state_tokens), dim=1)
        grid = condition.anchor_tokens[:, None] + state_grid + self.lead_embedding[None, :, None]
        grid = grid + self.lead_state_embedding[self.observed_state][None, :, None]
        for block, conditioner in zip(self.axial_blocks, self.conditioners):
            grid = block(conditioner(grid, step, condition.metadata))
        grid = self.final_norm(grid)
        coarse_velocity = self.decoder(grid[:, 1:], *condition.anchor_features[:2], self.lead_embedding[1:])
        visible = anchor[:, None].expand(-1, 11, -1, -1).reshape(batch * 11, 1, length)
        local = self.local_state(torch.cat((state.reshape(batch * 11, 1, length), visible), dim=1))
        local_condition = torch.cat((step[:, None].expand(-1, 11, -1),
                                     self.lead_embedding[None, 1:].expand(batch, -1, -1)), dim=-1)
        local = local + self.local_condition(local_condition).reshape(batch * 11, -1, 1)
        return coarse_velocity + self.local_head(local).reshape(batch, 11, length) + self.raw_state_skip(state)

    def forward(self, state, time, anchor, numeric, sex, field_mask, age_topcoded):
        condition = self.encode_conditions(anchor, numeric, sex, field_mask, age_topcoded)
        return self.velocity(state, time, condition), condition.anchor_prediction
