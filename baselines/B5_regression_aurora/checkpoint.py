"""Strict checkpoint contract for the physiology-decomposed regression model."""
from __future__ import annotations

from pathlib import Path
import torch

from . import ARCHITECTURE, PROTOCOL


def save(path: str | Path, *, model, optimizer, config: dict, stage: str, preprocessor, epoch: int) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"protocol": PROTOCOL, "architecture": ARCHITECTURE, "stage": stage, "epoch": int(epoch),
                "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "model_config": config["model"], "loss": config["loss"],
                "scales": {key: value.copy() for key, value in preprocessor.scale_uV_by_source.items()}}, temporary)
    temporary.replace(path)


def load(path: str | Path, *, model, expected_stage: str | None = None) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint is not compatible with B5-regression-AURORA")
    if expected_stage and payload.get("stage") != expected_stage:
        raise ValueError(f"Expected stage {expected_stage}, found {payload.get('stage')}")
    model.load_state_dict(payload["model"], strict=True)
    return payload
