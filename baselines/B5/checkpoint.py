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
from . import ARCHITECTURE_ID, TEMPORAL_ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
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
    if not isinstance(value, dict) or value.get("architecture_id") not in {ARCHITECTURE_ID, TEMPORAL_ARCHITECTURE_ID}:
        raise ValueError("Expected a B5-U or B5-T checkpoint")
    if value.get("condition_schema") != CONDITION_SCHEMA or value.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Checkpoint condition/preprocessing protocol mismatch")
    config = ModelConfig.from_dict(value["model_config"])
    if value.get("architecture_id") != config.architecture_id:
        raise ValueError("Checkpoint architecture ID disagrees with temporal configuration")
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


def initialize_model(model: B5UNet, checkpoint: dict[str, Any], use_ema: bool = True) -> dict[str, Any]:
    """Strict same-architecture load or explicitly audited B5 -> zero-branch B5-T."""
    source_config = ModelConfig.from_dict(checkpoint['model_config'])
    if checkpoint.get('architecture_id') != source_config.architecture_id or checkpoint.get('architecture_hash') != source_config.fingerprint:
        raise ValueError('Unverified initialization checkpoint architecture')
    if checkpoint.get('condition_schema') != CONDITION_SCHEMA or checkpoint.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Initialization condition/preprocessing protocol mismatch')
    state = checkpoint['ema'] if use_ema else checkpoint['model']
    destination = model.state_dict()
    new_keys = sorted(key for key in destination if key.startswith('temporal_branch.'))
    if source_config.fingerprint == model.config.fingerprint:
        model.load_state_dict(state, strict=True)
        return {'mode': 'strict_same_architecture', 'source_architecture_id': source_config.architecture_id,
                'target_architecture_id': model.config.architecture_id, 'loaded_keys': len(state), 'new_keys': []}
    if not model.config.temporal_enabled or source_config.temporal_enabled:
        raise ValueError('Only B5 -> B5-T initialization is allowed; exact resume requires the same architecture')
    if ModelConfig.from_dict(model.config.core_dict).fingerprint != source_config.fingerprint:
        raise ValueError('B5 -> B5-T requires identical original backbone and condition configuration')
    expected = set(destination)-set(new_keys)
    if set(state) != expected:
        raise ValueError('B5 source keys differ from the unchanged backbone; refusing partial loading')
    for key in expected:
        if state[key].shape != destination[key].shape:
            raise ValueError(f'Initialization tensor shape mismatch: {key}')
    incompatibility = model.load_state_dict(state, strict=False)
    if set(incompatibility.missing_keys) != set(new_keys) or incompatibility.unexpected_keys:
        raise ValueError('Only new temporal-branch keys may be missing')
    model.temporal_branch.reset_output_projection()
    return {'mode': 'B5_to_B5T_zero_residual', 'source_architecture_id': source_config.architecture_id,
            'target_architecture_id': model.config.architecture_id, 'loaded_keys': len(state),
            'new_keys': new_keys, 'output_projection_zero_initialized': True}
