"""Matched body-scale validation of frozen B5 and A-context interventions.

The existing control and A/epoch_0015 results are reused as references.
This script does not train or alter any checkpoint. Full runs default to the
same seed-42, Heun-16, K=16 protocol as the first body-scale ablation.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import checkpoint_preprocessor, load_checkpoint, model_from_checkpoint
from baselines.B5.config import ModelConfig
from baselines.B5.context_body import B5BodyContext, sample_body
from baselines.B5.context_body_attention import B5BodyContextAttention
from baselines.B5.context_body_beat_rhythm import B5BodyContextBeatRhythm
from baselines.B5.context_body_meanstd import B5BodyContextMeanStd
from baselines.B5.flow import sample
from baselines.B5.metadata import DemographicsTable
from baselines.B5.objective import file_sha256
from ecg12gen.evaluate import _evaluate_record_subset, write_report
from ecg12gen.preprocessing import PreprocessingConfig
from scripts.prepare_task2_context_scales import with_context_scales
from scripts.train_b5_body_context import BatchAdapter, BodySubset, on_device


ROOT = Path(__file__).resolve().parents[1]
MODES = ("base", "A_true", "A_disabled", "A_shuffled")
SOURCE = ROOT / "results/B5/context_start/best.pt"
A_CHECKPOINT = ROOT / "results/context_body/A/best.pt"
FROZEN_CHECKPOINT = ROOT / "results/context_body/A_frozen_adapter_v1/best.pt"
FROZEN_MEANSTD_CHECKPOINT = ROOT / "results/context_body/A_frozen_meanstd_v1/best.pt"
FROZEN_ATTENTION_CHECKPOINT = ROOT / "results/context_body/A_frozen_attention_v1/best.pt"
FEATURE_CHECKPOINTS = {kind: ROOT / f"results/context_body/A_frozen_{kind.removeprefix('frozen_')}_v1/best.pt"
                       for kind in ("frozen_beat", "frozen_rhythm", "frozen_beat_rhythm")}


def body_pairs(dataset: BodySubset):
    """Take one context record per target pair, preserving validation order."""
    pairs = {}
    for index in range(len(dataset)):
        item = dataset[index]
        pairs.setdefault(item.pair_id, item)
    if len(pairs) * 12 != len(dataset):
        raise ValueError("Body validation does not contain 12 windows per pair")
    if len({item.subject_id for item in pairs.values()}) != len(pairs):
        raise ValueError("Expected one body validation pair per subject")
    return pairs


def cross_subject_rotation(pairs):
    """Fixed record-level derangement; never mix windows of the same pair."""
    names = sorted(pairs)
    if len(names) < 2:
        raise ValueError("Need at least two validation pairs to shuffle context")
    for shift in range(1, len(names)):
        mapping = {name: names[(index + shift) % len(names)]
                   for index, name in enumerate(names)}
        if all(pairs[name].subject_id != pairs[donor].subject_id
               for name, donor in mapping.items()):
            return mapping
    raise ValueError("Could not create a cross-subject context derangement")


def cross_subject_permutation(pairs, seed: int):
    """Seeded record-level derangement for independent context controls."""
    names = sorted(pairs)
    if len(names) < 2:
        raise ValueError("Need at least two validation pairs to shuffle context")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        donors = rng.permutation(names)
        mapping = dict(zip(names, donors))
        if all(pairs[name].subject_id != pairs[donor].subject_id
               for name, donor in mapping.items()):
            return mapping
    raise ValueError("Could not sample a cross-subject context derangement")


class InterventionCollator:
    def __init__(self, adapter, mode: str, pairs, mapping):
        self.adapter, self.mode, self.pairs, self.mapping = adapter, mode, pairs, mapping

    def __call__(self, samples):
        if self.mode == "A_shuffled":
            changed = [replace(item, context=self.pairs[self.mapping[item.pair_id]].context)
                       for item in samples]
            batch = self.adapter(changed)
            for row, item in zip(batch["evaluation_metadata"], samples):
                row["shuffled_context_donor_pair_id"] = self.mapping[item.pair_id]
        else:
            batch = self.adapter(samples)
        if self.mode == "A_disabled":
            batch["context_time_mask"].zero_()
            batch["context_window_weights"].zero_()
            batch["context_available"].zero_()
        return batch


def load_model(mode: str, source, device, checkpoint_kind: str = "A"):
    if mode == "base" and checkpoint_kind == "A":
        return model_from_checkpoint(source, device, use_ema=True)
    if mode == "base":
        raise ValueError("Frozen-adapter diagnostics do not support base mode")
    path = {"A": A_CHECKPOINT, "frozen": FROZEN_CHECKPOINT,
            "frozen_meanstd": FROZEN_MEANSTD_CHECKPOINT,
            "frozen_attention": FROZEN_ATTENTION_CHECKPOINT, **FEATURE_CHECKPOINTS}.get(checkpoint_kind)
    if path is None:
        raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    settings = payload.get("run_config", {})
    if checkpoint_kind == "A":
        if (payload.get("format") != "b5-body-subset-v1"
                or settings.get("group") != "A"
                or settings.get("checkpoint_sha256") != file_sha256(SOURCE)
                or settings.get("steps") != 16 or settings.get("samples") != 16
                or payload.get("epoch") != 15):
            raise ValueError("A checkpoint does not match the first A ablation")
    elif checkpoint_kind in ("frozen", "frozen_meanstd", "frozen_attention", *FEATURE_CHECKPOINTS):
        expected_experiment = {"frozen": "A_frozen_adapter_v1",
                               "frozen_meanstd": "A_frozen_meanstd_v1",
                               "frozen_attention": "A_frozen_attention_v1",
                               **{kind: f"A_{kind}_v1" for kind in FEATURE_CHECKPOINTS}}[checkpoint_kind]
        if (payload.get("format") != "b5-body-frozen-adapter-v1"
                or settings.get("experiment") != expected_experiment
                or settings.get("source_checkpoint_sha256") != file_sha256(SOURCE)
                or settings.get("steps") != 16 or settings.get("samples") != 16
                or not settings.get("frozen_backbone") or payload.get("epoch") not in (5, 10)
                or (checkpoint_kind == "frozen_meanstd" and settings.get("encoder_variant") != "mean_std")
                or (checkpoint_kind == "frozen_attention" and settings.get("encoder_variant") != "attention")
                or (checkpoint_kind in FEATURE_CHECKPOINTS and
                    settings.get("encoder_variant") != checkpoint_kind.removeprefix("frozen_"))):
            raise ValueError("Frozen adapter checkpoint does not match the pilot")
        for name, tensor in source["ema"].items():
            if not torch.equal(payload["ema"][name].cpu(), tensor):
                raise ValueError(f"Frozen B5 tensor differs from source: {name}")
    else:
        raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")
    model_class = {"frozen_meanstd": B5BodyContextMeanStd,
                   "frozen_attention": B5BodyContextAttention}.get(checkpoint_kind, B5BodyContext)
    config = ModelConfig.from_dict(source["model_config"])
    model = (B5BodyContextBeatRhythm(config, checkpoint_kind.removeprefix("frozen_"))
             if checkpoint_kind in FEATURE_CHECKPOINTS else model_class(config))
    model.load_state_dict(payload["ema"], strict=True)
    return model.to(device).eval()


def check_existing_roster(pairs):
    reference = ROOT / "results/context_body/A/validation/epoch_0015/window_metadata.csv"
    if not reference.is_file():
        raise FileNotFoundError(f"Existing A reference is required: {reference}")
    with reference.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = {(row["pair_id"], row["target_record_id"], int(row["start_sample_500hz"]))
                for row in rows}
    expected = {(pair.pair_id, pair.target_record_id, start)
                for pair in pairs.values() for start in range(0, 60000, 5000)}
    if len(rows) != 300 or observed != expected:
        raise ValueError("Current validation roster differs from saved A/epoch_0015")


@torch.no_grad()
def evaluate(model, loader, preprocessor, mode: str, device, output: Path,
             steps: int, samples: int):
    predictions, targets, metadata = [], [], []
    scale = preprocessor.scale_uV_by_source["d12"][None, :, None]
    started = time.monotonic()
    for number, batch in enumerate(loader, start=1):
        inputs = on_device(batch, device)
        if mode == "base":
            generated = sample(model, inputs, batch["keys"], seed=42, steps=steps,
                               solver="heun", samples=samples)
        else:
            generated = sample_body(model, inputs, batch["keys"], seed=42, steps=steps,
                                    solver="heun", samples=samples)
        predictions.append(generated.float().cpu().numpy() * scale)
        targets.append(batch["target_uV"])
        metadata.extend(batch["evaluation_metadata"])
        if number % 10 == 0 or number == len(loader):
            print(json.dumps({"mode": mode, "batches_done": number,
                              "batches_total": len(loader),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    summary, details = _evaluate_record_subset(prediction, target, "task2", metadata)
    summary.update({"input_type": "body_scale_d6", "diagnostic_mode": mode,
                    "context_used": mode in ("A_true", "A_shuffled"),
                    "context_intervention": mode.removeprefix("A_") if mode != "base" else "none",
                    "solver": "heun", "integration_steps": steps, "samples": samples,
                    "seed": 42, "checkpoint_selection_metric": "body_scale_d6_r_missing11"})
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "prediction_uV.npy", prediction)
    np.save(output / "target_uV.npy", target)
    write_report(output, summary, details, title="B5 body-scale diagnostic evaluation")
    with (output / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in metadata for key in row}))
        writer.writeheader()
        writer.writerows(metadata)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--checkpoint-kind",
                        choices=("A", "frozen", "frozen_meanstd", "frozen_attention", *FEATURE_CHECKPOINTS), default="A")
    parser.add_argument("--preflight", action="store_true", help="Check data/model only; no inference/output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Independent A_shuffled donor permutation; omitted preserves legacy mapping")
    args = parser.parse_args()
    if args.shuffle_seed is not None and args.mode != "A_shuffled":
        parser.error("--shuffle-seed is only valid for A_shuffled")
    if args.checkpoint_kind != "A" and args.mode == "base":
        parser.error("Frozen checkpoint cannot be used with base mode")
    if min(args.batch_size, args.steps, args.samples) < 1:
        parser.error("Batch size, steps and samples must be positive")
    if not args.preflight and (args.steps != 16 or args.samples != 16):
        parser.error("Formal diagnostics require Heun16/K16")
    if args.device == "cuda" and not torch.cuda.is_available() and not args.preflight:
        parser.error("CUDA is unavailable; use --preflight locally or start the GPU instance")

    source = load_checkpoint(SOURCE)
    base = checkpoint_preprocessor(source, PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml"))
    preprocessor = with_context_scales(
        base, ROOT / "results/B5/context_params/body_A_scales.npz",
        "body_scale_d6", "A_raw_window")
    dataset = BodySubset("validation", "A_raw_window")
    try:
        pairs = body_pairs(dataset)
        check_existing_roster(pairs)
        mapping = (cross_subject_permutation(pairs, args.shuffle_seed)
                   if args.shuffle_seed is not None else cross_subject_rotation(pairs))
        demographics = DemographicsTable(ROOT / "data/userinfobean.csv")
        adapter = BatchAdapter(preprocessor, demographics, 0.0, seed=42)
        collator = InterventionCollator(adapter, args.mode, pairs, mapping)
        model = load_model(args.mode, source, torch.device("cpu" if args.preflight else args.device),
                           checkpoint_kind=args.checkpoint_kind)
        first = collator([dataset[0]])
        if args.mode == "A_disabled" and first["context_available"].any():
            raise ValueError("Disabled context is still available")
        if args.mode == "A_shuffled" and not first["context_available"].all():
            raise ValueError("Shuffled context must remain available")
        if args.mode in ("A_disabled", "A_shuffled"):
            original = adapter([dataset[0]])
            torch.testing.assert_close(first["anchor"], original["anchor"], rtol=0, atol=0)
            torch.testing.assert_close(first["target"], original["target"], rtol=0, atol=0)
            if args.mode == "A_shuffled" and torch.equal(first["context"], original["context"]):
                raise ValueError("Shuffled context is identical to the real context")
        checkpoint_path = (SOURCE if args.mode == "base" else
                           {"A": A_CHECKPOINT, "frozen": FROZEN_CHECKPOINT,
                            "frozen_meanstd": FROZEN_MEANSTD_CHECKPOINT,
                            "frozen_attention": FROZEN_ATTENTION_CHECKPOINT,
                            **FEATURE_CHECKPOINTS}[args.checkpoint_kind])
        info = {"mode": args.mode, "checkpoint_kind": args.checkpoint_kind,
                "windows": len(dataset), "pairs": len(pairs),
                "source_checkpoint_sha256": file_sha256(SOURCE),
                "model_checkpoint_sha256": file_sha256(checkpoint_path),
                "context_variant": "A_raw_window",
                "shuffle_seed": args.shuffle_seed if args.mode == "A_shuffled" else None,
                "shuffle_mapping": mapping if args.mode == "A_shuffled" else None}
        if args.preflight:
            inputs = on_device(first, torch.device("cpu"))
            with torch.no_grad():
                if args.mode == "base":
                    smoke = sample(model, inputs, first["keys"], seed=42, steps=1,
                                   solver="heun", samples=1)
                else:
                    smoke = sample_body(model, inputs, first["keys"], seed=42, steps=1,
                                        solver="heun", samples=1)
                    if args.mode == "A_disabled":
                        fallback = sample(model, inputs, first["keys"], seed=42, steps=1,
                                          solver="heun", samples=1)
                        torch.testing.assert_close(smoke, fallback, rtol=0, atol=0)
            if smoke.shape != (1, 12, 5000) or not torch.isfinite(smoke).all():
                raise ValueError("Single-window smoke inference failed")
            print(json.dumps({**info, "shuffle_mapping": None, "preflight": "passed",
                              "smoke_shape": list(smoke.shape)},
                             ensure_ascii=False))
            return
        output_name = (f"A_shuffled_seed_{args.shuffle_seed}"
                       if args.shuffle_seed is not None else args.mode)
        output_root = ROOT / "results/context_body" / {
            "A": "diagnostics", "frozen": "A_frozen_adapter_v1/diagnostics",
            "frozen_meanstd": "A_frozen_meanstd_v1/diagnostics",
            "frozen_attention": "A_frozen_attention_v1/diagnostics",
            **{kind: f"A_{kind}_v1/diagnostics" for kind in FEATURE_CHECKPOINTS}}[args.checkpoint_kind]
        output = output_root / output_name
        if output.exists():
            raise FileExistsError(f"Diagnostic output already exists: {output}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=collator, pin_memory=args.device == "cuda")
        summary = evaluate(model, loader, preprocessor, args.mode, torch.device(args.device),
                           output, args.steps, args.samples)
        (output / "run_config.json").write_text(
            json.dumps(info | {"batch_size": args.batch_size, "solver": "heun",
                               "steps": args.steps, "samples": args.samples, "seed": 42},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"mode": args.mode, "r_missing11": summary["r_missing11"],
                          "v1_v6_rmse_uV": summary["task2_missing_lead_mean_rmse_uV"]}))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
