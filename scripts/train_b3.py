"""Train B3-P0 or B3-P1 under the frozen main joint-anchor contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b3_train import train_b3


def _indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("context-lead-indices cannot be empty")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--stage", choices=("P0_anchor_only", "P1-C3"), required=True)
    parser.add_argument("--fusion-mode", choices=("none", "film_gated_residual"), default="none")
    parser.add_argument("--p0-checkpoint", default=None, help="Required for every P1 run")
    parser.add_argument("--context-source-type", choices=("watch_ecg", "ecg_machine_d6", "body_scale_d6"), default=None)
    parser.add_argument("--context-lead-indices", dest="context_channel_indices", type=_indices,
                        default=None, help="Canonical d6 indices, e.g. 0,1,2,3,4,5")
    parser.add_argument("--body-scale-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--source-dropout", type=float, default=0.0)
    parser.add_argument("--anchor-lr", type=float, default=None)
    parser.add_argument("--context-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-anchor-epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task_id == "task1" and args.context_source_type not in {None, "watch_ecg"}:
        parser.error("task1 requires watch_ecg context")
    if args.task_id == "task2" and args.stage == "P1-C3" and args.context_source_type not in {"ecg_machine_d6", "body_scale_d6"}:
        parser.error("task2 requires exactly one d6 --context-source-type")
    if args.stage == "P0_anchor_only":
        args.anchor_lr = 1e-3 if args.anchor_lr is None else args.anchor_lr
        args.context_lr = 2e-3 if args.context_lr is None else args.context_lr
        args.freeze_anchor_epochs = 0 if args.freeze_anchor_epochs is None else args.freeze_anchor_epochs
        args.warmup_epochs = 0
        args.min_lr_ratio = 1.0
    else:
        args.anchor_lr = 2e-5 if args.anchor_lr is None else args.anchor_lr
        args.context_lr = 5e-4 if args.context_lr is None else args.context_lr
        args.freeze_anchor_epochs = 10 if args.freeze_anchor_epochs is None else args.freeze_anchor_epochs
    if args.freeze_anchor_epochs < 0 or args.warmup_epochs < 0:
        parser.error("freeze/warmup epochs must be non-negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("min-lr-ratio must be in [0,1]")
    checkpoint = train_b3(args)
    print(f"B3 complete: {checkpoint}")


if __name__ == "__main__":
    main()
