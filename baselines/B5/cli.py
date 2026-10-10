"""Fine-tuning requires explicit opt-in; --help never loads Torch."""
from __future__ import annotations

import argparse


def training_cli() -> None:
    parser = argparse.ArgumentParser(description='B5-U Huawei fine-tuning from an existing compatible checkpoint')
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--init-checkpoint', help='Initialize from compatible public-pretrained or fine-tuned EMA weights')
    parser.add_argument('--resume', help='Resume this fine-tuning run, including optimizer/EMA/RNG state')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if not args.execute_training:
        parser.error('Training was not started. Configure the training machine, then explicitly pass --execute-training.')
    if bool(args.init_checkpoint) == bool(args.resume):
        parser.error('Provide exactly one of --init-checkpoint or --resume for fine-tuning.')
    from .config import load_config
    config = load_config(args.config)
    from .runtime import run_training
    run_training(config, args.device, args.init_checkpoint, args.resume)
