from __future__ import annotations

import argparse


def training_cli() -> None:
    parser = argparse.ArgumentParser(description="V2 multiscale attentive machine-context experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("Training was not started; pass --execute-training after reviewing the config.")
    from baselines.context_d6.config import load_config
    from .runtime import run_training
    run_training(load_config(args.config), args.device)

