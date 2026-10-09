"""Auditable checkpoints containing weights, scales, schema and RNG states."""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from . import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from .config import ModelConfig
from .model import B5UNet


def scales_payload(preprocessor: ECGPreprocessor) -> dict[str, list[float]]:
    return {key: value.astype(np.float32).tolist() for key, value in preprocessor.scale_uV_by_source.items()}


def scales_digest(payload: dict[str, list[float]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(payload):
        digest.update(key.encode())
        digest.update(np.asarray(payload[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def pack_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA device count; use fine-tune initialization instead")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_save(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Expected a B5-U checkpoint, not a diffusion or arbitrary model checkpoint")
    if value.get("condition_schema") != CONDITION_SCHEMA or value.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Checkpoint condition/preprocessing protocol mismatch")
    config = ModelConfig.from_dict(value["model_config"])
    if value.get("architecture_hash") != config.fingerprint:
        raise ValueError("Checkpoint architecture hash mismatch")
    if value.get("scales_sha256") != scales_digest(value["scales"]):
        raise ValueError("Checkpoint scale checksum mismatch")
    return value


def checkpoint_preprocessor(checkpoint: dict[str, Any], config: PreprocessingConfig) -> ECGPreprocessor:
    scales = {key: np.asarray(values, dtype=np.float32) for key, values in checkpoint["scales"].items()}
    for source, scale in scales.items():
        if source not in config.expected_leads or scale.shape != (config.expected_leads[source],):
            raise ValueError("Checkpoint scale shape/source mismatch")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid checkpoint scale")
    if "d12" not in scales or "ecg_machine_i" not in scales or not np.array_equal(scales["d12"][:1], scales["ecg_machine_i"]):
        raise ValueError("Checkpoint must preserve common d12-I/anchor scale")
    return ECGPreprocessor(config, scales)


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device, use_ema: bool = True) -> B5UNet:
    model = B5UNet(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["ema"] if use_ema else checkpoint["model"], strict=True)
    return model.to(device).eval()
