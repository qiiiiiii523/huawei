"""Configuration loading with repository-relative paths."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def load_yaml_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    return config, path.parent.parent

def resolve_config_path(config: dict[str, Any], repository_root: Path, key: str) -> Path:
    paths = config["paths"]
    raw = str(paths[key]).replace("${data_root}", str(paths["data_root"]))
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (repository_root / candidate).resolve()
