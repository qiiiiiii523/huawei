"""Training commands require an explicit opt-in; --help never loads Torch."""
from __future__ import annotations

import argparse


def training_cli(public: bool) -> None:
    parser = argparse.ArgumentParser(description="B6-M1-CFM public pretraining" if public else "B6-M1-CFM Huawei training/fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", help="Initialize model from a compatible pretrained EMA checkpoint")
    parser.add_argument("--resume", help="Resume optimizer/EMA/RNG state of this same run")
    parser.add_argument("--execute-training", action="store_true", help="Explicit opt-in to start actual training")
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("Training was not started. Configure the training machine, then explicitly pass --execute-training.")
    from .config import load_config
    config = load_config(args.config)
    if public != (config["stage"] == "public"):
        parser.error("Command and configuration stage disagree")
    from .runtime import run_training
    run_training(config, args.device, args.init_checkpoint, args.resume)
