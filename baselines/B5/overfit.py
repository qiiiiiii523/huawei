"""Explicit, train-only tiny-set diagnostic. Never run automatically."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--windows", type=int, default=1, help="Distinct training records with all target leads reliable")
    parser.add_argument("--steps", type=int, default=1000, help="Optimizer updates, not epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", help="New directory, default results/B5/overfit_wN")
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if not args.execute_training:
        parser.error("Training was not started. Tiny-set optimization also requires --execute-training.")
    if min(args.windows, args.steps, args.evaluate_every) < 1 or args.learning_rate <= 0:
        parser.error("Window count, update count, evaluation interval and learning rate must be positive")
    import numpy as np
    import torch
    from .config import ModelConfig, load_config
    from .data import HuaweiTrainDataset, demographics_table, load_preprocessor
    from .flow import sample
    from .losses import flow_loss
    from .model import B5UNet
    from .runtime import INPUT_FIELDS, collate, device_from_name, seed_all, to_device
    config = load_config(args.config)
    processor = load_preprocessor(config)
    dataset = HuaweiTrainDataset(config, processor, demographics_table(config))
    selected, seen = [], set()
    for index, row in enumerate(dataset.rows):
        if row["target_record_id"] in seen:
            continue
        item = dataset[index]
        if item["quality_mask"].all() and np.all(np.ptp(item["target"], axis=-1) > 0):
            selected.append(item)
            seen.add(row["target_record_id"])
        if len(selected) == args.windows:
            break
    if len(selected) != args.windows:
        raise ValueError("Not enough distinct fully reliable training records for the diagnostic")
    output = Path(args.output_dir).resolve() if args.output_dir else Path(config["repository_root"]) / "results" / "B5" / f"overfit_w{args.windows}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Tiny-set diagnostic requires a new/empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = device_from_name(args.device)
    seed_all(int(config["training"]["seed"]), bool(config["training"]["deterministic"]))
    # Disable regularization only in this explicitly labeled diagnostic, not in primary configs.
    model_config = replace(ModelConfig.from_dict(config["model"]), metadata_dropout=0., metadata_field_dropout=0.)
    model = B5UNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    batch = to_device(collate(selected), device)
    scale = torch.as_tensor(processor.scale_uV_by_source["d12"], device=device)
    source_target = batch["target"].detach()
    raw_target = source_target.cpu().numpy() * processor.scale_uV_by_source["d12"][None, :, None]
    np.save(output / "training_target_uV.npy", raw_target, allow_pickle=False)
    summary = {"diagnostic": "train_only_tiny_set_overfit", "validation_score": False,
               "training_records": args.windows, "optimizer_steps": args.steps,
               "metadata_dropout": 0., "weight_decay": 0., "learning_rate": args.learning_rate,
               "source_keys": batch["keys"], "sampling": config["sampling"]}
    (output / "settings.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def inspect(step: int, average_loss: float | None) -> None:
        model.eval()
        with torch.no_grad():
            prediction = sample(model, {key: batch[key] for key in INPUT_FIELDS}, batch["keys"],
                                int(config["sampling"]["seed"]), int(config["sampling"]["steps"]),
                                config["sampling"]["solver"], int(config["sampling"]["samples"]))
            p, y = prediction[:, 1:], source_target[:, 1:]
            centered_p, centered_y = p - p.mean(dim=-1, keepdim=True), y - y.mean(dim=-1, keepdim=True)
            denominator = (centered_p.square().sum(-1) * centered_y.square().sum(-1)).sqrt().clamp_min(1e-8)
            correlation = ((centered_p * centered_y).sum(-1) / denominator).mean()
            rmse_scaled = ((p - y).square().mean(-1).sqrt()).mean()
            rmse_uV = (((p - y) * scale[None, 1:, None]).square().mean(-1).sqrt()).mean()
            record = {"step": step, "average_train_loss": average_loss,
                      "training_window_r_missing11": float(correlation),
                      "training_window_mean_rmse_scaled": float(rmse_scaled),
                      "training_window_mean_rmse_uV": float(rmse_uV)}
            np.save(output / "last_prediction_uV.npy", prediction.cpu().numpy() * processor.scale_uV_by_source["d12"][None, :, None], allow_pickle=False)
        with (output / "diagnostic.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        model.train()

    inspect(0, None)
    running, count = 0., 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        # Fresh random t/noise, fixed visible conditions/targets. Not memorizing one noisy state.
        loss, _ = flow_loss(model, batch, config["loss"], scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]), error_if_nonfinite=True)
        optimizer.step()
        running, count = running + float(loss.detach()), count + 1
        if step % args.evaluate_every == 0 or step == args.steps:
            inspect(step, running / count)
            running, count = 0., 0
    # Intentionally no reusable checkpoint: this is a diagnostic, not a pretraining stage.


if __name__ == "__main__":
    main()
