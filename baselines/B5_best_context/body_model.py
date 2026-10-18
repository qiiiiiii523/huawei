"""Small external II corrector. B5-best FM/slow networks are not trained here."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .body_features import BODY_PROTOCOL, encode_windows, relation_features, restore_ii
from .adapters import _visible_digest


class BodyIIAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.beat = nn.Sequential(nn.Conv1d(2, 24, 9, padding=4), nn.SiLU(),
            nn.Conv1d(24, 24, 9, stride=2, padding=4), nn.SiLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.context = nn.Sequential(nn.Linear(51, 64), nn.SiLU(), nn.Linear(64, 64))
        self.signal = nn.Sequential(nn.Conv1d(2, 32, 15, padding=7), nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=4), nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=8, dilation=2), nn.SiLU())
        self.film = nn.Linear(64, 64)
        self.head = nn.Conv1d(32, 1, 7, padding=3)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, lead_i, base, beat, stats, enabled):
        features = self.signal(torch.stack((lead_i, base), dim=1))
        context = self.context(torch.cat((self.beat(beat), stats), dim=1)) * enabled[:, None]
        gamma, beta = self.film(context).chunk(2, dim=1)
        modified = features * (1 + .15 * torch.tanh(gamma)[..., None]) + .15 * beta[..., None]
        return base + .35 * torch.tanh(self.head(torch.nn.functional.silu(modified))[:, 0])


def load_body_checkpoint(path, base_sha, device, expected_group=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("protocol") != BODY_PROTOCOL or payload.get("base_checkpoint_sha256") != base_sha
            or payload.get("sampling") != {"seed": 42, "solver": "heun", "steps": 16, "samples": 16}
            or payload.get("slow_beta") != .75
            or payload.get("group") not in ("no_context", "context")
            or (expected_group and payload["group"] != expected_group)):
        raise ValueError("Body checkpoint is not aligned with this frozen B5-best/group/settings")
    model = BodyIIAdapter().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


class BodyAdapter:
    """Inference contract: predicted ECG + visible I + historical body record ONLY.

    Disabling/unavailable context keeps the learned generic correction, matching
    the teammate's ablation. A degenerate current-I window receives no correction.
    """
    def __init__(self, checkpoint, base_sha, device, *, mode="correct", expected_group=None):
        if mode not in ("correct", "disabled"):
            raise ValueError("Use correct or disabled; shuffled donors are supplied by the study")
        self.model, self.payload = load_body_checkpoint(checkpoint, base_sha, device, expected_group)
        self.device, self.mode, self.cache = device, mode, {}

    @torch.no_grad()
    def __call__(self, prediction, observed_i, history):
        key = None if history is None else (history.record_id, _visible_digest(history))
        if key not in self.cache:
            self.cache[key] = relation_features(history)
        beat, stats, available = self.cache[key]
        enabled = available and self.mode == "correct" and self.payload["group"] == "context"
        signal, base, scales = encode_windows(observed_i, prediction[1])
        args = [signal, base, np.repeat(beat[None], 12, axis=0),
                np.repeat(stats[None], 12, axis=0), np.full(12, enabled, np.float32)]
        output = self.model(*(torch.from_numpy(np.asarray(x, np.float32).copy()).to(self.device) for x in args))
        result = prediction.copy()
        result[1] = restore_ii(prediction[1], base, output.cpu().numpy(), scales)
        delta = result[1].astype(np.float64) - prediction[1]
        return result, {"adapter": "body_II_" + self.payload["group"], "body_protocol": BODY_PROTOCOL,
            "body_context_used": bool(enabled), "body_mode": self.mode, "history_available": bool(available),
            "fallback": "none" if enabled else "learned_generic_II_correction",
            "body_checkpoint_epoch": self.payload["epoch"],
            "max_II_correction_uV": float(np.max(np.abs(delta))),
            "rms_II_correction_uV": float(np.sqrt(np.mean(delta ** 2))),
            "zero_current_I_windows": int((scales == 0).sum())}
