"""Matched Task-2 body-scale continued-training runs: control, A and B.

This is a device-subset experiment, not the organizer's combined Task-2 score.
The initial B5 best.pt is read-only. Each group starts from its EMA weights.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from baselines.B5.checkpoint import (atomic_save, checkpoint_preprocessor, load_checkpoint,
                                     pack_rng, restore_rng)
from baselines.B5.config import ModelConfig
from baselines.B5.context_body import B5BodyContext, sample_body
from baselines.B5.flow import sample
from baselines.B5.losses import flow_loss
from baselines.B5.metadata import DemographicsTable
from baselines.B5.model import B5UNet
from baselines.B5.objective import file_sha256
from baselines.B5.runtime import seed_all
from ecg12gen.evaluate import _evaluate_record_subset, write_report
from ecg12gen.preprocessing import PreprocessingConfig
from ecg12gen.record_context import collate_record_context
from ecg12gen.record_dataset import JointAnchorDataset
from scripts.prepare_task2_context_scales import with_context_scales


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"control": ("A_raw_window", "body_A_scales.npz"),
            "A": ("A_raw_window", "body_A_scales.npz"),
            "B": ("B_detrend_0p2Hz_then_window", "body_B_scales.npz")}
TENSOR_FIELDS = ("anchor", "target", "quality_mask", "numeric", "sex", "field_mask",
                 "age_topcoded", "context", "context_time_mask", "context_window_weights",
                 "context_available")
LOSS = {"huber": 0.1, "pcc": 0.1, "anchor": 0.02, "physiology": 0.0, "huber_delta": 1.0}


class BodySubset(Dataset):
    def __init__(self, split: str, variant: str) -> None:
        self.base = JointAnchorDataset(ROOT / "configs/common.yaml", "task2", split,
                                       body_scale_variant=variant)
        self.indices = [index for index, row_index in enumerate(self.base._indices)
                        if self.base._rows[row_index]["input_type"] == "body_scale_d6"]
        if not self.indices or (split == "validation" and len(self.indices) % 12):
            raise ValueError("Missing or incomplete body-scale target pairs")
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base[self.indices[index]]

    def close(self) -> None:
        self.base.close()


class BatchAdapter:
    def __init__(self, preprocessor, demographics, dropout: float, seed: int) -> None:
        self.preprocessor, self.demographics, self.dropout = preprocessor, demographics, dropout
        self.rng = np.random.default_rng(seed)

    def __call__(self, samples):
        raw = collate_record_context(samples, self.preprocessor,
                                     context_dropout=self.dropout, rng=self.rng if self.dropout else None)
        result = {"anchor": raw["anchor_i"], "target": raw["target"],
                  "quality_mask": raw["target_quality_mask"],
                  "context": raw["context"], "context_time_mask": raw["context_time_mask"],
                  "context_window_weights": raw["context_window_weights"],
                  "context_available": raw["context_available"]}
        demographic_rows = [self.demographics.get(sample.subject_id) for sample in samples]
        for field in ("numeric", "sex", "field_mask", "age_topcoded"):
            result[field] = np.stack([row[field] for row in demographic_rows])
        result = {key: torch.from_numpy(np.asarray(value).copy()) for key, value in result.items()}
        result["keys"] = [sample.target_record_id + ":" + str(sample.meta["start_sample_500hz"])
                          for sample in samples]
        result["target_uV"] = raw["target_uV"]
        result["evaluation_metadata"] = raw["evaluation_metadata"]
        return result


def on_device(batch, device):
    return {key: value.to(device, non_blocking=True) if key in TENSOR_FIELDS else value
            for key, value in batch.items()}


@torch.no_grad()
def validate(model, loader, preprocessor, group: str, device, output: Path,
             steps: int, samples: int):
    model.eval()
    predictions, targets, metadata = [], [], []
    d12_scale = preprocessor.scale_uV_by_source["d12"][None, :, None]
    for batch in loader:
        inputs = on_device(batch, device)
        if group == "control":
            generated = sample(model, inputs, batch["keys"], seed=42, steps=steps,
                               solver="heun", samples=samples)
        else:
            generated = sample_body(model, inputs, batch["keys"], seed=42, steps=steps,
                                    solver="heun", samples=samples)
        predictions.append(generated.float().cpu().numpy() * d12_scale)
        targets.append(batch["target_uV"])
        metadata.extend(batch["evaluation_metadata"])
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    summary, details = _evaluate_record_subset(prediction, target, "task2", metadata)
    summary.update({"input_type": "body_scale_d6", "group": group, "context_used": group != "control",
                    "solver": "heun", "integration_steps": steps, "samples": samples,
                    "selection_metric": "body_scale_d6_r_missing11"})
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "prediction_uV.npy", prediction)
    np.save(output / "target_uV.npy", target)
    write_report(output, summary, details, title="B5 body-scale subset raw-uV record evaluation")
    with (output / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in metadata for key in row}))
        writer.writeheader()
        writer.writerows(metadata)
    return summary


def run(args) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation < 1 or args.validate_every < 1:
        raise ValueError("Training counts must be positive")
    if args.samples < 1 or args.steps < 1 or args.learning_rate <= 0:
        raise ValueError("Sampling counts and learning rate must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    source = ROOT / "results/B5/context_start/best.pt"
    initial = load_checkpoint(source)
    config = ModelConfig.from_dict(initial["model_config"])
    variant, scale_file = VARIANTS[args.group]
    base = checkpoint_preprocessor(initial, PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml"))
    preprocessor = with_context_scales(base, ROOT / "results/B5/context_params" / scale_file,
                                       "body_scale_d6", variant)
    model = (B5UNet(config) if args.group == "control" else B5BodyContext(config)).to(device)
    missing, unexpected = model.load_state_dict(initial["ema"], strict=args.group == "control")
    if unexpected or (args.group != "control" and
                      (not missing or not all(key.startswith(("body_encoder.", "body_film.")) for key in missing))):
        raise ValueError("Base checkpoint and body adapter are incompatible")
    # Model construction consumes random draws; reseed before every group for matched order/noise.
    seed_all(42, deterministic=True)
    train_data, val_data = BodySubset("train", variant), BodySubset("validation", variant)
    output = ROOT / "results/context_body" / args.group
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is not empty: {output}; use --resume or archive it")
    output.mkdir(parents=True, exist_ok=True)
    settings = {"group": args.group, "variant": variant, "epochs": args.epochs,
                "batch_size": args.batch_size, "accumulation": args.accumulation,
                "learning_rate": args.learning_rate, "validate_every": args.validate_every,
                "steps": args.steps, "samples": args.samples, "seed": 42,
                "checkpoint_sha256": file_sha256(source), "train_windows": len(train_data),
                "validation_windows": len(val_data), "model_config": initial["model_config"],
                "loss": LOSS, "context_dropout": 0 if args.group == "control" else 0.1,
                "amp": args.amp}
    settings = json.loads(json.dumps(settings))
    ledger = output / "run_config.json"
    if args.resume:
        if not ledger.is_file() or json.loads(ledger.read_text(encoding="utf-8")) != settings:
            raise ValueError("Resume settings/checkpoint hash do not match run_config.json")
    else:
        ledger.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    train_adapter = BatchAdapter(preprocessor, DemographicsTable(ROOT / "data/userinfobean.csv"),
                                 1.0 if args.group == "control" else 0.1, seed=42)
    val_adapter = BatchAdapter(preprocessor, train_adapter.demographics, 0.0, seed=42)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=0, collate_fn=train_adapter, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=val_adapter, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    start_epoch, updates, best_r = 0, 0, float("-inf")
    if args.resume:
        saved = torch.load(output / "last.pt", map_location=device, weights_only=True)
        if saved.get("run_config") != settings:
            raise ValueError("Checkpoint differs from current run")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["grad_scaler"])
        ema = saved["ema"]
        start_epoch, updates, best_r = saved["epoch"], saved["updates"], saved["best_r"]
        generator.set_state(saved["loader_rng"])
        train_adapter.rng.bit_generator.state = saved["collate_rng"]
        restore_rng(saved["rng"])
    scale = torch.as_tensor(preprocessor.scale_uV_by_source["d12"], device=device)
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for index, batch in enumerate(train_loader):
                batch = on_device(batch, device)
                group_start = (index // args.accumulation) * args.accumulation
                divisor = min(args.accumulation, len(train_loader) - group_start)
                with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                    loss, _ = flow_loss(model, batch, LOSS, scale, body_context=args.group != "control")
                scaler.scale(loss / divisor).backward()
                running += float(loss.detach())
                if (index + 1) % args.accumulation == 0 or index + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    with torch.no_grad():
                        for key, value in model.state_dict().items():
                            ema[key].mul_(0.999).add_(value, alpha=0.001)
            record = {"epoch": epoch + 1, "updates": updates, "train_loss": running / len(train_loader)}
            improved = False
            if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
                evaluation = (B5UNet(config) if args.group == "control" else B5BodyContext(config)).to(device)
                evaluation.load_state_dict(ema, strict=True)
                summary = validate(evaluation, val_loader, preprocessor, args.group, device,
                                   output / "validation" / f"epoch_{epoch + 1:04d}", args.steps, args.samples)
                record["validation"] = summary
                improved = summary["r_missing11"] > best_r
                if improved:
                    best_r = summary["r_missing11"]
                del evaluation
            payload = {"format": "b5-body-subset-v1", "run_config": settings,
                       "epoch": epoch + 1, "updates": updates, "best_r": best_r,
                       "model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(),
                       "grad_scaler": scaler.state_dict(), "rng": pack_rng(),
                       "loader_rng": generator.get_state(), "collate_rng": train_adapter.rng.bit_generator.state}
            if improved:
                atomic_save(output / "best.pt", payload)
            atomic_save(output / "last.pt", payload)
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"epoch": epoch + 1, "updates": updates,
                              "loss": record["train_loss"], "best_body_r": best_r}), flush=True)
    finally:
        train_data.close()
        val_data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast; off by default, as in B5")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
