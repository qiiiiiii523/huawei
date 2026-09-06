"""Run or resume the frozen 14-experiment B1 official matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

from b1.official_v1 import DEFAULT_OUTPUT_ROOT, read_matrix, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", help="Experiment id; repeatable. Omit for all 14.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="One epoch in the separate smoke directory.")
    parser.add_argument("--max-batches", type=int, default=None, help="Only valid with --smoke.")
    args = parser.parse_args()
    if args.max_batches is not None and not args.smoke:
        parser.error("--max-batches is only permitted with --smoke")
    specs = read_matrix()
    selected = set(args.experiment or [])
    if selected:
        unknown = selected - {spec.id for spec in specs}
        if unknown:
            parser.error(f"Unknown experiment ids: {sorted(unknown)}")
        specs = [spec for spec in specs if spec.id in selected]
    for spec in specs:
        run_experiment(spec, output_root=args.output_root, smoke=args.smoke,
                       max_batches=args.max_batches, device_name=args.device)


if __name__ == "__main__":
    main()
