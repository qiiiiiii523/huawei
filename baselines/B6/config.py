"""Versioned B6 configuration, independent of the public main contract."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .objective import slow_settings


@dataclass(frozen=True)
class ModelConfig:
    cnn_channels: tuple[int, int, int] = (64, 128, 256)
    decoder_channels: tuple[int, int] = (128, 64)
    state_channels: tuple[int, int] = (32, 64)
    state_fine_channels: int = 16
    d_model: int = 128
    num_blocks: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    time_tokens: int = 250
    attention_axes: str = "both"
    time_dim: int = 128
    metadata_dim: int = 128
    metadata_enabled: bool = True
    metadata_dropout: float = 0.2
    metadata_field_dropout: float = 0.05

    def __post_init__(self) -> None:
        for values, count in ((self.cnn_channels, 3), (self.decoder_channels, 2), (self.state_channels, 2)):
            if len(values) != count or any(not isinstance(c, int) or c < 4 for c in values):
                raise ValueError("B6 channel lists must have the expected lengths and widths >=4")
        if self.state_fine_channels < 4 or self.d_model < 8 or self.num_heads < 1 or self.d_model % self.num_heads:
            raise ValueError("B6 attention dimension must be divisible by positive head count")
        if not 1 <= self.num_blocks <= 8 or self.ffn_dim < self.d_model or self.time_tokens != 250:
            raise ValueError("B6 requires 1..8 blocks, ffn_dim >= d_model, and 250 canonical time tokens")
        if self.attention_axes not in {"both", "time_only", "lead_only"} or not 0 <= self.dropout < 1:
            raise ValueError("Invalid B6 attention axes or network dropout")
        if self.time_dim < 4 or self.time_dim % 2 or self.metadata_dim < 4:
            raise ValueError("time_dim must be even; embedding dimensions must be >=4")
        if not 0 <= self.metadata_dropout < 1 or not 0 <= self.metadata_field_dropout < 1:
            raise ValueError("Invalid metadata dropout configuration")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        args = dict(value)
        for key in ("cnn_channels", "decoder_channels", "state_channels"):
            if key in args:
                args[key] = tuple(args[key])
        return cls(**args)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base":
            continue
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _read(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError("Configuration inheritance cycle")
    seen.add(path)
    with path.open(encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a configuration mapping: {path}")
    if "base" in value:
        value = _merge(_read(path.parent / str(value["base"]), seen), value)
    return value


def repository_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "ecg12gen" / "preprocessing.py").is_file():
            return candidate
    raise ValueError("B6 config must be located inside the huawei repository")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _read(path, set())
    config["repository_root"] = str(repository_root(path))
    config["config_path"] = str(path)
    if config.get("stage") not in {"local", "public", "finetune"}:
        raise ValueError("stage must be local, public, or finetune")
    ModelConfig.from_dict(config.get("model", {}))
    training = config["training"]
    for key in ("epochs", "batch_size", "gradient_accumulation", "validate_every"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be >=1")
    if int(training["workers"]) < 0 or float(training["learning_rate"]) <= 0:
        raise ValueError("Invalid worker count or learning rate")
    if not 0 <= int(training["seed"]) < 2**32 or float(training["gradient_clip"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("Invalid seed, clipping threshold, or weight decay")
    if not 0 <= float(training["ema_decay"]) < 1:
        raise ValueError("EMA decay must be in [0,1)")
    sampling = config["sampling"]
    if sampling["solver"] not in {"heun", "euler"} or int(sampling["steps"]) < 1 or int(sampling["samples"]) < 1:
        raise ValueError("Invalid sampling settings")
    if any(float(config["loss"][key]) < 0 for key in ("huber", "pcc", "anchor", "physiology")):
        raise ValueError("Loss weights cannot be negative")
    if float(config["loss"]["huber_delta"]) <= 0:
        raise ValueError("Huber delta must be positive")
    slow_settings(config["loss"])  # Legacy configs default to a disabled auxiliary.
    if not isinstance(training.get("validate_initial", False), bool):
        raise ValueError("training.validate_initial must be a boolean")
    tasks = config["validation"]["tasks"]
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in {"task1", "task2"} for task in tasks):
        raise ValueError("validation.tasks must contain distinct task1/task2 entries")
    if config["validation"]["selection_task"] not in tasks or int(config["validation"]["batch_size"]) < 1:
        raise ValueError("Invalid validation checkpoint task or batch size")
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    raw = str(config["paths"][key])
    root = Path(config["repository_root"])
    data = config["paths"].get("data_root", "../Data")
    data_path = Path(str(data).replace("${repo}", str(root)))
    if not data_path.is_absolute():
        data_path = (root / data_path).resolve()
    raw = raw.replace("${repo}", str(root)).replace("${data}", str(data_path))
    candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
