"""Training and raw-uV validation for the B4 conditional diffusion baseline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b4_data import B4PreparedDataset, b4_collate
from .b4_diffusion import B4Diffusion
from .evaluate import evaluate_joint_anchor_predictions, write_report
from .training import seed_everything


def _loader(dataset: B4PreparedDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=b4_collate,
        generator=generator,
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def validate_b4(
    diffusion: B4Diffusion,
    dataset: B4PreparedDataset,
    d12_scale_uV: np.ndarray,
    device: torch.device,
    output_dir: Path,
    *,
    batch_size: int,
    sampling_steps: int,
    eta: float,
    max_batches: int | None = None,
) -> dict[str, float | str]:
    diffusion.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    quality_masks: list[np.ndarray] = []
    generator = torch.Generator(device=device.type)
    generator.manual_seed(42)
    for batch_index, raw_batch in enumerate(_loader(dataset, batch_size, False, 42)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _move(raw_batch, device)
        model_prediction = diffusion.sample(
            batch["anchor_model"],
            sampling_steps=sampling_steps,
            eta=eta,
            generator=generator,
        )
        prediction_uV = model_prediction.cpu().numpy() * d12_scale_uV[None, :, None]
        raw_anchor = raw_batch["raw_anchor_uV"].numpy()
        prediction_uV[:, :1] = raw_anchor
        predictions.append(prediction_uV.astype(np.float32))
        targets.append(raw_batch["raw_target_uV"].numpy())
        anchors.append(raw_anchor)
        lead_i_quality = np.ones((len(raw_anchor), 1), dtype=bool)
        quality_masks.append(np.concatenate((lead_i_quality, raw_batch["target_quality_mask"].numpy()), axis=1))
    if not predictions:
        raise ValueError("B4 validation produced no batches")
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    anchor = np.concatenate(anchors)
    target_quality_mask = np.concatenate(quality_masks)
    summary, raw_details, _, prediction_submit = evaluate_joint_anchor_predictions(
        prediction,
        target,
        anchor,
        dataset.samples[0].task_id,
        target_quality_mask,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "prediction_raw.npy", prediction)
    np.save(output_dir / "prediction_submit.npy", prediction_submit)
    np.save(output_dir / "validation_target.npy", target)
    np.save(output_dir / "validation_anchor_i.npy", anchor)
    write_report(output_dir, summary, raw_details, title="B4 missing-11 raw-uV validation report")
    (output_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fit_b4(
    diffusion: B4Diffusion,
    train_dataset: B4PreparedDataset,
    validation_dataset: B4PreparedDataset,
    d12_scale_uV: np.ndarray,
    output_dir: str | Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    sampling_steps: int,
    eta: float,
    device: str,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> Path:
    seed_everything(42, deterministic=True)
    torch_device = torch.device(device)
    diffusion.to(torch_device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = _loader(train_dataset, batch_size, True, 42)
    best_metric = -float("inf")
    history: list[dict[str, Any]] = []
    checkpoint_path = output / "b4_best.pt"
    for epoch in range(1, epochs + 1):
        diffusion.train()
        total_loss = 0.0
        steps = 0
        for batch_index, raw_batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            batch = _move(raw_batch, torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(
                batch["missing_target_model"],
                batch["anchor_model"],
                batch["target_quality_mask"],
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        metrics = validate_b4(
            diffusion,
            validation_dataset,
            d12_scale_uV,
            torch_device,
            output,
            batch_size=batch_size,
            sampling_steps=sampling_steps,
            eta=eta,
            max_batches=max_validation_batches,
        )
        metric = float(metrics["r_missing11"])
        row = {"epoch": epoch, "train_loss": total_loss / max(steps, 1), "validation": metrics}
        history.append(row)
        print(f"epoch={epoch}; train_loss={row['train_loss']:.6f}; r_missing11={metric:.6f}")
        if metric > best_metric:
            best_metric = metric
            checkpoint = {
                **diffusion.denoiser.architecture_metadata,
                "model": diffusion.denoiser.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "stage": "P0_anchor_only",
                "training_steps": diffusion.training_steps,
                "sampling_steps": sampling_steps,
                "eta": eta,
                "checkpoint_metric": "r_missing11",
                "checkpoint_metric_value": metric,
                "target_d12_scale_uV": np.asarray(d12_scale_uV, dtype=np.float32).tolist(),
            }
            torch.save(checkpoint, checkpoint_path)
    (output / "history.json").write_text(
        json.dumps({"checkpoint_metric": "r_missing11", "best_metric": best_metric, "history": history}, indent=2),
        encoding="utf-8",
    )
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "r_missing11"))
        writer.writeheader()
        writer.writerows(
            {"epoch": row["epoch"], "train_loss": row["train_loss"], "r_missing11": row["validation"]["r_missing11"]}
            for row in history
        )
    return checkpoint_path
