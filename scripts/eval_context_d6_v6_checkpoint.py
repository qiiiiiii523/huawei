"""Evaluate one V6 reliability-gated bottleneck cross-attention checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.config import ModelConfig
from baselines.context_d6.config import load_config, resolve_path
from baselines.context_d6.data import MachineD6Dataset
from baselines.context_d6.runtime import _preprocessor
from baselines.context_d6_v2.runtime import validate
from baselines.context_d6_v6 import ContextD6V6UNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()

    config = load_config(args.config)
    config["sampling"] = dict(config["sampling"])
    config["sampling"].update({"solver": "heun", "steps": args.steps,
                               "samples": args.samples, "seed": 42})
    base = load_checkpoint(resolve_path(config, "init_checkpoint"))
    preprocessor = _preprocessor(config, base)
    dataset = MachineD6Dataset(config, "validation", preprocessor)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = "B5-U-context-d6-v6-reliability-gated"
    if checkpoint.get("architecture_id") != expected:
        raise ValueError(f"Expected {expected}, got {checkpoint.get('architecture_id')}")
    context_config = checkpoint["context_config"]
    model = ContextD6V6UNet(
        ModelConfig.from_dict(checkpoint["model_config"]),
        int(context_config["dimension"]),
        initial_query_fraction=float(context_config.get("initial_query_fraction", 0.2)),
        injection_layers=tuple(int(layer) for layer in context_config.get(
            "injection_layers", [0, 1, 2])),
        cross_attention_heads=int(context_config.get("cross_attention_heads", 4)),
        cross_attention_dropout=float(context_config.get("cross_attention_dropout", 0.1)),
        reliability_hidden=int(context_config.get("reliability_hidden", 64)),
        reliability_logit_clamp=float(context_config.get("reliability_logit_clamp", 2.0)))
    model.load_state_dict(checkpoint[args.weights], strict=True)
    device = torch.device(args.device)
    model.to(device)
    summary = validate(model, dataset, config, device, Path(args.output))
    summary["checkpoint_weights"] = args.weights
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
