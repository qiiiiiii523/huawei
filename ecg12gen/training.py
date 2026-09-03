"""模型无关的公共训练协议工具；不包含任何 B0/B1/B2 模型或训练循环。"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int = 42, deterministic: bool = True) -> dict[str, Any]:
    """Set the shared random seed for Python, NumPy, and optionally PyTorch.

    The function deliberately does not create a model, data loader, optimizer,
    or training run. It is the common reproducibility hook for later B models.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    state: dict[str, Any] = {"seed": seed, "deterministic": deterministic, "torch_available": False}
    try:
        import torch  # type: ignore
    except ImportError:
        return state
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    state["torch_available"] = True
    return state


def load_training_protocol(path: str | Path = "configs/training_protocol_v1.yaml") -> dict[str, Any]:
    """Load and minimally validate the fixed shared experiment protocol."""
    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    required = {"protocol_version", "reproducibility", "data", "loader", "optimization_defaults", "validation", "results"}
    if not isinstance(protocol, dict) or required - set(protocol):
        raise ValueError("training protocol is missing required protocol sections")
    if protocol["reproducibility"].get("seed") != 42:
        raise ValueError("The initial main protocol fixes seed=42")
    if protocol["data"].get("split_strategy") != "fixed_subject_level":
        raise ValueError("Only fixed subject-level splitting is allowed")
    return protocol
