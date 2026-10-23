"""B7-only strict checkpoints. B5 payloads are intentionally rejected."""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch

from . import ARCHITECTURE_ID, CHECKPOINT_VERSION, GENERATED_LEADS
from .config import ModelConfig
from .leads import validate_scales
from .model import B7Transformer


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(value, dict) or value.get("architecture_id") != ARCHITECTURE_ID
            or value.get("format_version") != CHECKPOINT_VERSION
            or tuple(value.get("generated_leads", ())) != GENERATED_LEADS):
        raise ValueError("Expected a B7-T checkpoint. B5 and B7-U7 weights/migration are not supported.")
    ModelConfig.from_dict(value["model_config"])
    validate_scales(value["scales_uV"])
    return value


def model_from_checkpoint(value, device="cpu"):
    if value.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Only B7-T architecture is supported")
    model = B7Transformer(ModelConfig.from_dict(value["model_config"]))
    model.load_state_dict(value["ema"], strict=True)
    return model.to(device).eval()


def pack_rng():
    # Save numpy state as tensors/basic types to support weights_only loading.
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], torch.tensor(n[1].astype("int64")), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype("uint32"), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires matching CUDA resume environment")
        torch.cuda.set_rng_state_all(state["cuda"])
