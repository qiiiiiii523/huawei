"""Target-free 120-second TCN correction for the official B5-final package."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from baselines.B5_best.common import sha

from .predict import Predictor as BasePredictor
from .profile import BASE_SHA, BODY_SHA, Assets as BaseAssets, resolve_assets as resolve_base_assets
from .trend_corrector.model import OUTPUT_INDICES, TrendCorrector, compose_d12, make_features


PROTOCOL = "B5-final-tcn120-official-submission-v1"
TCN_PROTOCOL = "B5-final-120s-zero-mean-tcn-v1"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


@dataclass(frozen=True)
class Assets:
    base: BaseAssets
    tcn: Path
    tcn_payload: dict

    def report(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "base_checkpoint": str(self.base.base), "base_sha256": BASE_SHA,
            "body_checkpoint": str(self.base.body), "body_sha256": BODY_SHA,
            "tcn_checkpoint": str(self.tcn), "tcn_sha256": sha(self.tcn),
            "tcn_step": int(self.tcn_payload["step"]),
            "tcn_validation_r": float(self.tcn_payload["score"]),
            "sampling": self.base.report()["sampling"],
            "composition": "B5-final three-task output -> zero-mean full-120s TCN correction -> official CSV",
            "training_started": False, "target_used_at_inference": False,
        }


def resolve_assets(root: str | Path, profile_path: str | Path, settings: dict) -> Assets:
    root = Path(root).resolve()
    base = resolve_base_assets(root, profile_path,
                               checkpoint=settings["base_checkpoint"],
                               body_checkpoint=settings["body_checkpoint"])
    tcn = _resolve(root, settings["tcn_checkpoint"])
    if not tcn.is_file():
        raise FileNotFoundError(f"Missing selected TCN checkpoint: {tcn}")
    if sha(tcn) != settings["tcn_sha256"]:
        raise ValueError("Wrong TCN checkpoint SHA256")
    payload = torch.load(tcn, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    scales = np.asarray(config.get("scales_uV"), dtype=np.float32)
    if (payload.get("step") != 1750 or payload.get("config", {}).get("protocol") != TCN_PROTOCOL
            or not isinstance(payload.get("model"), dict) or scales.shape != (12,)
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("Selected TCN checkpoint has an unsupported protocol/configuration")
    return Assets(base, tcn, payload)


class Predictor:
    """Apply the TCN to every official task after the normal B5-final route."""

    def __init__(self, device: torch.device, *, assets: Assets):
        self.device, self.assets = device, assets
        self.base = BasePredictor(device, assets=assets.base)
        scales = np.asarray(assets.tcn_payload["config"]["scales_uV"], dtype=np.float32)
        self.model = TrendCorrector(scales).to(device)
        self.model.load_state_dict(assets.tcn_payload["model"], strict=True)
        self.model.eval()
        self.scales = scales

    @torch.no_grad()
    def _correct(self, prediction: np.ndarray) -> np.ndarray:
        raw = np.asarray(prediction, dtype=np.float32)
        if raw.shape != (12, 60_000) or not np.isfinite(raw).all():
            raise ValueError("B5-final output must be finite [12,60000]")
        features = make_features(raw, self.scales)[None].to(self.device)
        correction, _ = self.model(features)
        direct = torch.from_numpy(raw[list(OUTPUT_INDICES)].copy()).to(self.device)[None] + correction
        result = compose_d12(torch.from_numpy(raw[:1].copy()).to(self.device)[None], direct)[0].cpu().numpy()
        if not np.array_equal(result[0], raw[0]):
            raise AssertionError("TCN changed visible I")
        if not np.allclose(result[list(OUTPUT_INDICES)].mean(axis=-1), raw[list(OUTPUT_INDICES)].mean(axis=-1), atol=2e-5):
            raise AssertionError("TCN changed a direct II/V1--V6 full-record mean")
        if result.shape != (12, 60_000) or not np.isfinite(result).all():
            raise ValueError("TCN produced an invalid output")
        return result.astype(np.float32, copy=False)

    def predict_tasks(self, *args, **kwargs):
        predictions, statuses = self.base.predict_tasks(*args, **kwargs)
        corrected = {name: self._correct(value) for name, value in predictions.items()}
        tagged = {name: {**status, "tcn120_applied": True, "tcn_step": 1750,
                         "tcn_zero_mean_direct_leads": True}
                  for name, status in statuses.items()}
        return corrected, tagged
