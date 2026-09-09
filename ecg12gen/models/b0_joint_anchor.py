"""B0 linear anchor backbone with reserved Task1 P1-C3 modules.

P0 activates only the bias-free linear mapping. C3 uses window-level context
statistics, never pointwise cross-time alignment. No target enters forward().
"""
import hashlib
import json
import torch
from torch import nn


class B0JointAnchor(nn.Module):
    architecture_id = 'b0_linear_anchor_context_v1'

    def __init__(self, context_channels=1, gate_logit_bias=-2.944439):
        super().__init__()
        self.architecture_config = {
            'context_channels': context_channels, 'anchor_channels': 1,
            'output_channels': 12, 'anchor_kernel': 1, 'anchor_bias': False,
            'context_features': ['mean', 'std', 'mean_abs', 'rms'],
            'context_hidden': 16, 'residual_hidden': 16,
            'gate_logit_bias': float(gate_logit_bias), 'baseline_uV': 0,
        }
        self.mapping = nn.Conv1d(1, 12, 1, bias=False)
        self.context_encoder = nn.Sequential(nn.Linear(context_channels * 4, 16), nn.Tanh())
        self.film = nn.Linear(16, 24)
        self.gate = nn.Linear(16, 12)
        self.residual = nn.Sequential(nn.Conv1d(28, 16, 1), nn.Tanh(), nn.Conv1d(16, 12, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_logit_bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.set_stage('P0_anchor_only')

    @property
    def architecture_config_hash(self):
        data = json.dumps(self.architecture_config, sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()

    def set_stage(self, stage):
        if stage not in {'P0_anchor_only', 'P1-C3'}:
            raise ValueError('B0 runs only P0_anchor_only and P1-C3')
        self.stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(stage == 'P1-C3' or name.startswith('mapping.'))

    def forward(self, context_ecg, anchor_i_ecg, context_lead_mask=None, anchor_lead_mask=None):
        if anchor_i_ecg.ndim != 3 or anchor_i_ecg.shape[1] != 1:
            raise ValueError('Anchor must have shape [B,1,T]')
        if anchor_lead_mask is not None:
            if (anchor_lead_mask.shape != (len(anchor_i_ecg), 12)
                or anchor_lead_mask.dtype != torch.bool
                or not anchor_lead_mask[:, 0].all() or anchor_lead_mask[:, 1:].any()):
                raise ValueError('Target-time observed mask must be I-only')
        base = self.mapping(anchor_i_ecg)
        if self.stage == 'P0_anchor_only':
            return base
        expected = self.architecture_config['context_channels']
        if context_ecg is None or context_ecg.ndim != 3 or context_ecg.shape[:2] != (len(base), expected):
            raise ValueError('C3 requires matching context batch/channels')
        if context_lead_mask is None or context_lead_mask.shape != context_ecg.shape[:2]:
            raise ValueError('C3 requires context mask')
        x = context_ecg * context_lead_mask.to(context_ecg.dtype).unsqueeze(-1)
        # Fixed-length, time-aggregated features; no sample-to-sample matching.
        features = torch.cat((x.mean(-1), x.std(-1, unbiased=False),
                              x.abs().mean(-1), x.square().mean(-1).clamp_min(1e-12).sqrt()), dim=1)
        z = self.context_encoder(features)
        delta_gamma, beta = self.film(z).chunk(2, dim=1)
        modulated = (1 + delta_gamma.unsqueeze(-1)) * base + beta.unsqueeze(-1)
        gate = self.gate(z).sigmoid().unsqueeze(-1)
        residual_input = torch.cat((modulated, z.unsqueeze(-1).expand(-1, -1, base.shape[-1])), dim=1)
        return modulated + gate * self.residual(residual_input)
