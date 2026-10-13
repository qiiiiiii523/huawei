"""Configuration validation for paired machine-d6 control/context experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from baselines.B5.config import ModelConfig, _read, repository_root
from baselines.B5.objective import validate_mainline_loss


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _read(path, set())
    config["repository_root"] = str(repository_root(path))
    config["config_path"] = str(path)
    ModelConfig.from_dict(config["model"])
    if config.get("stage") != "context_d6_adaptation":
        raise ValueError("context_d6 requires stage=context_d6_adaptation")
    if config.get("device") != "ecg_machine_d6" or config.get("body_scale_variant") != "A_raw_window":
        raise ValueError("This entry point is fixed to ecg_machine_d6/A_raw_window")
    context = config["context"]
    if not isinstance(context.get("enabled"), bool) or int(context.get("dimension", 0)) < 4:
        raise ValueError("context.enabled must be bool and context.dimension >=4")
    if not 0 <= float(context.get("dropout", 0)) <= 1:
        raise ValueError("context.dropout must be in [0,1]")
    training = config["training"]
    for key in ("epochs", "batch_size", "gradient_accumulation", "validate_every"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be >=1")
    if int(training["workers"]) != 0:
        raise ValueError("First context-d6 implementation requires workers=0 for deterministic context dropout")
    if float(training["learning_rate"]) <= 0 or float(training["gradient_clip"]) <= 0:
        raise ValueError("Invalid learning rate or gradient clip")
    validate_mainline_loss(config["loss"])
    sampling = config["sampling"]
    if sampling["solver"] not in {"heun", "euler"} or int(sampling["steps"]) < 1 or int(sampling["samples"]) < 1:
        raise ValueError("Invalid sampling settings")
    if config["validation"].get("selection_metric") != "r_missing11":
        raise ValueError("Per-device experiment must select by machine r_missing11")
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config["repository_root"])
    candidate = Path(str(config["paths"][key]).replace("${repo}", str(root)))
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

