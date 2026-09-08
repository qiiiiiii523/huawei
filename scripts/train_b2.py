"""Run reproducible B2-P0/C1/C2 joint-anchor experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg12gen.b2_data import (build_joint_dataset, build_strict_dataset, dataset_summary,
                               fit_b2_preprocessor, task2_dual_intersection)
from ecg12gen.b2_model import B2JointAnchorPatchTransformer, B2ModelConfig
from ecg12gen.b2_train import P0_STAGE, P1_STAGE, fit_b2


def _spec(name: str) -> tuple[dict[str, object], dict[str, object]]:
    with (ROOT / "configs" / "experiments" / "b2_patch_transformer.yaml").open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    try: return config["model"], config["experiments"][name]
    except KeyError as error: raise SystemExit(f"Unknown B2 experiment: {name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="B2-P0, B2-C1, B2-C2, T1-P0, T1-C2-watch, T1-shuffle-watch, T2-P0, T2-machine, T2-body, T2-both, or T2-shuffle")
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--p0-checkpoint")
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--body-scale-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--context-channel-indices", type=int, nargs="+", default=None)
    parser.add_argument("--common-intersection", action="store_true", help="Evaluate/train machine or body on the verified body+machine common intersection")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None); args = parser.parse_args()
    model_raw, exp = _spec(args.experiment)
    if exp.get("task_id") and exp["task_id"] != args.task_id: raise SystemExit("--task-id disagrees with experiment")
    stage, view = str(exp["stage"]), str(exp["context_view"])
    if stage == P1_STAGE and not args.p0_checkpoint: raise SystemExit("P1_joint_anchor requires --p0-checkpoint")
    if stage == P0_STAGE and args.p0_checkpoint: raise SystemExit("P0 does not accept --p0-checkpoint")
    indices = tuple(args.context_channel_indices) if args.context_channel_indices else None
    if args.task_id == "task1" and indices is not None: raise SystemExit("task1 watch context has one fixed channel")
    output = Path(args.output_dir or ROOT / "results" / f"b2_{args.experiment}"); output.mkdir(parents=True, exist_ok=True)
    intersection: dict[str, object] = {}
    if args.task_id == "task2":
        for split in ("train", "validation"):
            _, counts = task2_dual_intersection(args.config, split, args.body_scale_variant, indices); intersection[split] = counts
        (output / "task2_common_intersection.json").write_text(json.dumps(intersection, indent=2), encoding="utf-8")
        if view == "both" and (not intersection["train"]["both_rows"] or not intersection["validation"]["both_rows"]):
            raise SystemExit(f"T2-both is disabled: exact machine/body intersection is empty; {intersection}")
    preprocessor = fit_b2_preprocessor(args.config, args.task_id, args.body_scale_variant, indices)
    train = build_strict_dataset(args.config, preprocessor) if stage == P0_STAGE else build_joint_dataset(args.config, args.task_id, "train", preprocessor, args.body_scale_variant, indices, view, common_intersection=args.common_intersection)
    validation_view = "auto" if stage == P0_STAGE else view
    validation = build_joint_dataset(args.config, args.task_id, "validation", preprocessor, args.body_scale_variant, indices, validation_view, common_intersection=args.common_intersection)
    model_values = dict(model_raw); model_values["time_transformer_layers"] = model_values.pop("anchor_time_transformer_layers")
    model_values.update({"fusion_mode": exp["fusion_mode"], "context_dropout": 0.10, "initial_gate": 0.03})
    model_config = B2ModelConfig(**model_values)
    (output / "preprocessing_scales.json").write_text(json.dumps({k: v.tolist() for k, v in preprocessor.scale_uV_by_source.items()}, indent=2), encoding="utf-8")
    (output / "b2_run.json").write_text(json.dumps({"experiment": args.experiment, "task_id": args.task_id, "stage": stage,
        "context_view": view, "common_intersection": args.common_intersection, "diagnostic_only": bool(exp.get("diagnostic_only", False)), "p0_checkpoint": args.p0_checkpoint,
        "model_config": model_config.__dict__, "train": dataset_summary(train), "validation": dataset_summary(validation),
        "task2_common_intersection": intersection, "training_output_i_replacement": "forbidden",
        "validation_input_contract": "joint_anchor_test_like"}, indent=2), encoding="utf-8")
    checkpoint = fit_b2(B2JointAnchorPatchTransformer(model_config), train, validation, preprocessor.scale_uV_by_source["d12"],
                        args.task_id, output, stage=stage, p0_checkpoint=args.p0_checkpoint, device=args.device, epochs=args.epochs)
    print(f"B2 complete: {args.experiment}; checkpoint={checkpoint}")


if __name__ == "__main__": main()
