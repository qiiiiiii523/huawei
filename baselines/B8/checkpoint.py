"""Strict, self-describing B8 checkpoints; B7/FM checkpoints are rejected."""
from __future__ import annotations
from pathlib import Path
import os
import torch
from . import ARCHITECTURE_ID, CHECKPOINT_VERSION
from .model import B8M1TimeRegressor

def save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp"); torch.save(payload, temporary); os.replace(temporary, path)

def load(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("architecture_id") != ARCHITECTURE_ID or value.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Expected a B8 M1 time-only regression checkpoint")
    return value

def model_from_checkpoint(value, device="cpu"):
    config = value["architecture"]
    model = B8M1TimeRegressor(**{key: config[key] for key in ("d_model", "blocks", "heads", "ffn", "dropout")})
    model.load_state_dict(value["ema"], strict=True)
    return model.to(device).eval()
