"""Fine-tuning requires explicit opt-in; --help never loads Torch."""
from __future__ import annotations

import argparse


def validate_training_start(config: dict, init_path: str | None, resume_path: str | None) -> None:
    if init_path and resume_path:
        raise ValueError("Choose either initialization or exact resume")
    stage = config.get("stage")
    if stage not in {"local", "finetune"}:
        raise ValueError("B5 only runs Huawei local or finetune stages")
    if stage == "local" and init_path:
        raise ValueError("Huawei-only scratch training must not use --init-checkpoint")
    if stage == "finetune" and not (init_path or resume_path):
        raise ValueError("Fine-tuning requires --init-checkpoint or --resume")


def training_cli() -> None:
    parser = argparse.ArgumentParser(description='B5-U Huawei scratch training or compatible-checkpoint fine-tuning')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--init-checkpoint', help='Initialize from compatible public-pretrained or fine-tuned EMA weights')
    parser.add_argument('--resume', help='Resume this fine-tuning run, including optimizer/EMA/RNG state')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if not args.execute_training:
        parser.error('Training was not started. Configure the training machine, then explicitly pass --execute-training.')
    from .config import load_config
    config = load_config(args.config)
    try:
        validate_training_start(config, args.init_checkpoint, args.resume)
    except ValueError as error:
        parser.error(str(error))
    from .runtime import run_training
    run_training(config, args.device, args.init_checkpoint, args.resume)
