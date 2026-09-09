"""Train the original B2-v1 network on the main joint-anchor protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b2_data import (build_joint_dataset, build_strict_dataset,
                               dataset_summary, fit_b2_preprocessor)
from ecg12gen.b2_model import B2MaskedPatchTransformer, B2ModelConfig, architecture_metadata
from ecg12gen.b2_train import P0_STAGE, P1_STAGES, fit_b2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True)
    parser.add_argument("--stage", choices=(P0_STAGE, *sorted(P1_STAGES)), required=True)
    parser.add_argument("--p0-checkpoint", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "common.yaml"))
    parser.add_argument("--body-scale-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--context-view", choices=("auto", "machine", "body", "watch"), default="auto")
    parser.add_argument("--context-channel-indices", nargs="+", type=int, default=None)
    parser.add_argument("--fusion-mode", choices=("film", "gated_residual", "film_gated_residual"), default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.stage in P1_STAGES and not args.p0_checkpoint:
        parser.error("P1 stages require --p0-checkpoint")
    expected_fusion = {"P1-C1": "film", "P1-C2": "gated_residual", "P1-C3": "film_gated_residual"}
    fusion_mode = "none" if args.stage == P0_STAGE else expected_fusion[args.stage]
    if args.fusion_mode is not None and args.fusion_mode != fusion_mode:
        parser.error(f"{args.stage} requires fusion mode {fusion_mode}")
    if args.task_id == "task2" and args.context_view == "watch":
        parser.error("task2 context must be machine or body d6")
    context_indices = tuple(args.context_channel_indices) if args.context_channel_indices else None
    if args.task_id == "task2" and args.stage in P1_STAGES and args.context_view == "auto":
        parser.error("task2 P1 runs require one mutually exclusive --context-view machine or body")

    preprocessor = fit_b2_preprocessor(args.config, args.task_id, args.body_scale_variant, context_indices)
    strict = build_strict_dataset(args.config, preprocessor)
    validation = build_joint_dataset(args.config, args.task_id, "validation", preprocessor,
                                     args.body_scale_variant, context_indices,
                                     args.context_view if args.stage in P1_STAGES else "auto")
    train = strict if args.stage == P0_STAGE else build_joint_dataset(
        args.config, args.task_id, "train", preprocessor, args.body_scale_variant,
        context_indices, args.context_view)
    model = B2MaskedPatchTransformer(B2ModelConfig(fusion_mode=fusion_mode))
    output = Path(args.output_dir) if args.output_dir else ROOT / "results" / f"b2_{args.task_id}_{args.stage}"
    output.mkdir(parents=True, exist_ok=True)
    architecture = architecture_metadata(model.config)
    (output / "preprocessing_scales.json").write_text(
        json.dumps({key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}, indent=2), encoding="utf-8")
    (output / "b2_run.json").write_text(json.dumps({
        "task_id": args.task_id, "stage": args.stage, "fusion_mode": fusion_mode,
        "body_scale_variant": args.body_scale_variant, "context_view": args.context_view,
        "context_channel_indices": context_indices, "model_config": vars(model.config),
        "architecture_id": architecture["architecture_id"], "architecture_config_hash": architecture["architecture_config_hash"],
        "parameter_count": model.parameter_count, "train": dataset_summary(train), "validation": dataset_summary(validation),
        "raw_baseline_policy": "fixed_zero_uV", "p0_checkpoint": args.p0_checkpoint,
    }, indent=2, default=str), encoding="utf-8")
    checkpoint = fit_b2(model, train, validation, preprocessor.scale_uV_by_source["d12"],
                        args.task_id, output, stage=args.stage, p0_checkpoint=args.p0_checkpoint,
                        device=args.device, epochs=args.epochs)
    print(f"B2 complete: checkpoint={checkpoint}; parameters={model.parameter_count}")


if __name__ == "__main__":
    main()
