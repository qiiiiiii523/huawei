"""Train M1-main with the shared B2 data routes and fixed V0 selection rule."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b2_data import (
    build_pseudo_dataset,
    build_strict_dataset,
    build_weak_dataset,
    fit_b2_preprocessor,
)
from ecg12gen.m1_model import M1MaskedCNNLeadTimeTransformer
from ecg12gen.m1_train import fit_m1
from ecg12gen.training import seed_everything


def _default_experiment_path(task_id: str, route: str, weak_profile: str | None, pseudo_variant: str) -> Path:
    if route == "A":
        suffix = "_arm_a_weak_a1.yaml" if weak_profile == "A1" else "_arm_a_weak.yaml"
    elif route == "B":
        suffix = "_arm_b_sync_weak.yaml"
    else:
        suffix = "_arm_c_sync_rpeak_c2.yaml" if pseudo_variant == "C2" else "_arm_c_sync_rpeak.yaml"
    return ROOT / "configs" / "experiments" / f"{task_id}{suffix}"


def _load_experiment(path: Path, task_id: str, route: str) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("task_id") != task_id or config.get("experiment_arm") != route:
        raise SystemExit(f"Experiment config does not match {task_id} route {route}: {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--route", choices=("A", "B", "C"), required=True)
    parser.add_argument("--stage", choices=("strict", "mixed"), default="strict")
    parser.add_argument("--task2-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--pseudo-variant", choices=("C1", "C2"), default="C1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--weak-profile", choices=("A0", "A1"), default=None)
    args = parser.parse_args()

    if args.task_id == "task2" and args.route == "C":
        raise SystemExit("Route C is task1-only")
    if args.route == "C" and args.stage != "mixed":
        raise SystemExit("Route C requires --stage mixed")
    if args.route == "A" and args.stage != "strict":
        raise SystemExit("Route A is raw weak adaptation and requires --stage strict")
    seed_everything(42, deterministic=True)

    experiment_path = Path(args.experiment_config) if args.experiment_config else _default_experiment_path(
        args.task_id, args.route, args.weak_profile, args.pseudo_variant
    )
    if not experiment_path.is_absolute():
        experiment_path = ROOT / experiment_path
    experiment = _load_experiment(experiment_path, args.task_id, args.route)
    experiment_variant = str(experiment.get("experiment_variant", ""))
    weak_profile = args.weak_profile or ("A1" if experiment_variant == "A1" else "A0")
    if args.route == "A" and experiment_variant not in {"A0", "A1"}:
        raise SystemExit("Route A requires an A0 or A1 experiment config")
    if args.route == "A" and weak_profile != experiment_variant:
        raise SystemExit("--weak-profile disagrees with experiment config")
    if args.route == "C" and experiment_variant != args.pseudo_variant:
        raise SystemExit("--pseudo-variant disagrees with experiment config")

    preprocessor = fit_b2_preprocessor(args.config, args.task_id, args.task2_variant)
    strict = build_strict_dataset(args.config, args.task_id, preprocessor)
    validation = build_weak_dataset(args.config, args.task_id, "validation", preprocessor, args.task2_variant)
    weak = None
    pseudo = None
    if args.route in {"A", "B"} and (args.route == "A" or args.stage == "mixed"):
        weak = build_weak_dataset(args.config, args.task_id, "train", preprocessor, args.task2_variant)
    if args.route == "C":
        pseudo = build_pseudo_dataset(args.config, preprocessor, args.pseudo_variant)

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / f"m1_{args.task_id}_{args.route}_{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = M1MaskedCNNLeadTimeTransformer()
    (output_dir / "preprocessing_scales.json").write_text(
        json.dumps({key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}, indent=2), encoding="utf-8"
    )
    (output_dir / "m1_run.json").write_text(json.dumps({
        "task_id": args.task_id, "route": args.route, "stage": args.stage,
        "task2_variant": args.task2_variant, "pseudo_variant": args.pseudo_variant,
        "experiment_config": str(experiment_path), "experiment_variant": experiment_variant,
        "weak_profile": weak_profile, "parameter_count": model.parameter_count,
        "seed": 42, "deterministic": True, "adapter": False, "baseline_head": False,
        "random_lead_masking": False, "random_point_masking": False,
        "raw_baseline_policy": "fixed_zero_uV",
    }, indent=2), encoding="utf-8")
    checkpoint = fit_m1(
        model, strict, validation, preprocessor.scale_uV_by_source["d12"], args.task_id, output_dir,
        device=args.device, epochs=args.epochs, route=args.route, stage=args.stage,
        weak_dataset=weak, pseudo_dataset=pseudo, strict_reference_dataset=strict,
        weak_profile=weak_profile,
    )
    print(f"M1 complete: checkpoint={checkpoint}; parameters={model.parameter_count}")


if __name__ == "__main__":
    main()
