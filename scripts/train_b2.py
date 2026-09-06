"""Train the B2-v1 model under the frozen shared protocol."""
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
    build_strict_validation_dataset,
    build_weak_dataset,
    fit_b2_preprocessor,
)
from ecg12gen.b2_model import B2MaskedPatchTransformer
from ecg12gen.b2_train import check_weak_protocol_compatibility, fit_b2, fit_b2_staged
from ecg12gen.preprocessing import PreprocessingConfig
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
    if not isinstance(config, dict) or config.get("task_id") != task_id:
        raise SystemExit(f"Experiment config does not match task: {path}")
    if config.get("experiment_arm") != route:
        raise SystemExit(f"Experiment config does not match route {route}: {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--route", choices=("A", "B", "C"), required=True)
    parser.add_argument("--stage", choices=("strict", "mixed", "staged"), default="strict")
    parser.add_argument("--task2-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--pseudo-variant", choices=("C1", "C2"), default="C1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--strict-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument(
        "--finetune-mode", choices=("frozen_encoder", "full"), default="frozen_encoder",
        help="staged B/C adaptation: freeze Transformer encoder or update all parameters",
    )
    parser.add_argument("--strict-anchor-weight", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--weak-profile", choices=("A0", "A1"), default=None)
    args = parser.parse_args()
    # Seed before constructing any B2 model.  This makes A0/A1 differ only
    # by their configured loss profile when they are launched in separate
    # processes under the shared protocol.
    seed_everything(42, deterministic=True)

    if args.route == "C" and args.task_id != "task1":
        raise SystemExit("Route C is task1-only")
    if args.route == "C" and args.stage != "staged":
        raise SystemExit("Route C requires --stage staged")
    if args.route == "B" and args.stage == "mixed":
        raise SystemExit("Route B mixed training is legacy; use --stage staged for strict pretrain then weak fine-tune")

    experiment_path = Path(args.experiment_config) if args.experiment_config else _default_experiment_path(args.task_id, args.route, args.weak_profile, args.pseudo_variant)
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
    if args.route == "A" or args.route == "B" and args.stage == "mixed":
        check_weak_protocol_compatibility(ROOT / "configs" / "training_protocol_v1.yaml", ROOT / "configs" / "losses.yaml")

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / f"b2_{args.task_id}_{args.route}_{args.stage}"
    preprocessor = fit_b2_preprocessor(args.config, args.task_id, args.task2_variant)
    strict = build_strict_dataset(args.config, args.task_id, preprocessor)
    strict_validation = build_strict_validation_dataset(args.config, args.task_id, preprocessor)
    validation = build_weak_dataset(args.config, args.task_id, "validation", preprocessor, args.task2_variant)
    weak = None
    pseudo = None
    if args.route == "B" and args.stage in {"mixed", "staged"}:
        weak = build_weak_dataset(args.config, args.task_id, "train", preprocessor, args.task2_variant)
    if args.route == "A":
        weak = build_weak_dataset(args.config, args.task_id, "train", preprocessor, args.task2_variant)
    if args.route == "C":
        pseudo = build_pseudo_dataset(args.config, preprocessor, args.pseudo_variant)

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    (output_dir / "preprocessing_scales.json").write_text(
        json.dumps({key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}, indent=2), encoding="utf-8"
    )
    (output_dir / "b2_run.json").write_text(json.dumps({
        "task_id": args.task_id, "route": args.route, "stage": args.stage,
        "task2_variant": args.task2_variant, "pseudo_variant": args.pseudo_variant,
        "experiment_config": str(experiment_path), "experiment_variant": experiment_variant,
        "weak_profile": weak_profile, "parameter_count": B2MaskedPatchTransformer().parameter_count,
        "seed": 42, "deterministic": True,
        "training_protocol": "strict_pretrain_then_finetune" if args.stage == "staged" else "single_stage",
        "finetune_mode": args.finetune_mode,
        "strict_anchor_weight": args.strict_anchor_weight,
        "finetune_policy": (
            "freeze_transformer_encoder_train_stem_and_head_with_strict_anchor"
            if args.stage == "staged" and args.finetune_mode == "frozen_encoder"
            else "full_parameter_finetune" if args.stage == "staged" else "not_applicable"
        ),
        "strict_validation_protocol": "held_out_d12_same_window_diagnostic_only",
        "strict_epochs": args.strict_epochs if args.strict_epochs is not None else args.epochs,
        "finetune_epochs": args.finetune_epochs if args.finetune_epochs is not None else args.epochs,
        "raw_baseline_policy": "fixed_zero_uV", "preprocessing_config": str(preprocessing_config),
    }, indent=2, default=str), encoding="utf-8")
    model = B2MaskedPatchTransformer()
    if args.stage == "staged":
        checkpoint = fit_b2_staged(
            model, strict, validation, preprocessor.scale_uV_by_source["d12"], args.task_id,
            output_dir, args.device,
            args.strict_epochs if args.strict_epochs is not None else args.epochs,
            args.finetune_epochs if args.finetune_epochs is not None else args.epochs,
            args.route, weak, pseudo, strict if args.route == "B" else None,
            strict_validation_dataset=strict_validation, weak_profile=weak_profile,
            finetune_mode=args.finetune_mode, strict_anchor_weight=args.strict_anchor_weight,
        )
    else:
        checkpoint = fit_b2(
            model, strict, validation, preprocessor.scale_uV_by_source["d12"], args.task_id,
            output_dir, args.device, args.epochs, args.route, args.stage, weak, pseudo, strict,
            weak_profile=weak_profile,
        )
    print(f"B2 complete: checkpoint={checkpoint}; parameters={B2MaskedPatchTransformer().parameter_count}")


if __name__ == "__main__":
    main()
