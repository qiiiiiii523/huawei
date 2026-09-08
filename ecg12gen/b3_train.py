"""B3-P0/P1 training, validation and checkpoint rules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import ContractError
from .evaluate import evaluate_joint_anchor_predictions, evaluate_task2_diagnostics
from .losses import joint_anchor_sync_loss, strict_anchor_pretrain_loss
from .b3_data import B3JointDataset, B3StrictDataset, collate_b3, fit_b3_preprocessor
from .b3_model import B3Model
from .training import seed_everything


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward(model: B3Model, batch: dict[str, Any]) -> torch.Tensor:
    if model.fusion_mode == "none":
        # This branch intentionally does not read context, source type, or masks.
        return model(batch["anchor_i"])
    return model(batch["anchor_i"], context_ecg=batch["context"],
                 context_source_type=batch["context_source_type"],
                 context_lead_mask=batch.get("context_lead_mask"))


def _loader(dataset: Any, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=False, collate_fn=collate_b3, generator=generator)


def _raw_prediction(model_output: np.ndarray, baseline: np.ndarray, d12_scale: np.ndarray) -> np.ndarray:
    return model_output.astype(np.float32) * d12_scale[None, :, None] + baseline[:, :, None].astype(np.float32)


@torch.no_grad()
def validate_v0(model: B3Model, dataset: B3JointDataset, d12_scale: np.ndarray,
                task_id: str, device: torch.device, *, batch_size: int = 4,
                shuffled_context: bool = False) -> dict[str, Any]:
    """Run test-like validation; target is never passed to the model."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    loader = _loader(dataset, batch_size, False, 42)
    for batch in loader:
        moved = _move(batch, device)
        if shuffled_context and model.fusion_mode != "none":
            permutation = torch.roll(torch.arange(moved["context"].shape[0], device=device), shifts=1)
            moved["context"] = moved["context"][permutation]
            moved["context_lead_mask"] = moved["context_lead_mask"][permutation]
            moved["context_source_type"] = [moved["context_source_type"][int(i)] for i in permutation.cpu()]
        output = _forward(model, moved)
        baseline = model.predict_baseline(moved["anchor_i"]).cpu().numpy()
        predictions.append(_raw_prediction(output.cpu().numpy(), baseline, d12_scale))
        targets.append(batch["target_raw"].numpy())
        anchors.append(batch["anchor_raw"].numpy())
    prediction_raw = np.concatenate(predictions).astype(np.float32)
    target_raw = np.concatenate(targets).astype(np.float32)
    anchor_raw = np.concatenate(anchors).astype(np.float32)
    summary, raw_details, submit_details, prediction_submit = evaluate_joint_anchor_predictions(
        prediction_raw, target_raw, anchor_raw, task_id,
    )
    subject_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    if task_id == "task2":
        # Keep the shared evaluator's required metadata names without exposing
        # target arrays to the model.
        metadata_rows = [{"subject_id": dataset[index]["subject_id"],
                          "input_type": dataset[index]["context_source_type"]}
                         for index in range(len(dataset))]
        subject_rows, device_rows = evaluate_task2_diagnostics(prediction_submit, target_raw, metadata_rows)
        summary["task2_subject_macro_r_submit_12"] = float(np.nanmean([r["twelve_lead_mean_pearson_r"] for r in subject_rows]))
        summary["task2_v1_v6_rmse_uV"] = float(np.nanmean([r["generated_v1_v6_mean_rmse_uV"] for r in device_rows]))
    return {"summary": summary, "raw_details": raw_details, "submit_details": submit_details,
            "prediction_raw": prediction_raw, "prediction_submit": prediction_submit,
            "target_raw": target_raw, "anchor_raw": anchor_raw,
            "task2_subject_rows": subject_rows, "task2_device_rows": device_rows}


def _set_anchor_frozen(model: B3Model, frozen: bool) -> None:
    anchor_names = set(model.anchor_parameter_names)
    for name, parameter in model.named_parameters():
        if name in anchor_names:
            parameter.requires_grad = not frozen


def _optimizer(model: B3Model, base_lr: float, context_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    context_prefixes = ("watch_context_encoder.", "machine_d6_context_encoder.", "body_d6_context_encoder.",
                        "film.", "gate.", "residual_adapter.")
    context_params = [p for name, p in model.named_parameters() if name.startswith(context_prefixes)]
    anchor_params = [p for name, p in model.named_parameters() if not name.startswith(context_prefixes)]
    groups = [{"params": anchor_params, "lr": base_lr}, {"params": context_params, "lr": context_lr}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _save_run_metadata(output_dir: Path, args: Any, preprocessor: Any, model: B3Model) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preprocessing_scales.json").write_text(
        json.dumps({key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "b3_run.json").write_text(json.dumps({
        "task_id": args.task_id, "stage": args.stage, "fusion_mode": args.fusion_mode,
        "parameter_count": model.parameter_count, "seed": args.seed, "deterministic": True,
        "architecture_id": model.architecture_id,
        "architecture_config_hash": model.architecture_config_hash,
        "context_dropout": args.context_dropout, "source_dropout": args.source_dropout,
        "freeze_anchor_epochs": args.freeze_anchor_epochs, "anchor_lr": args.anchor_lr,
        "context_lr": args.context_lr, "loss": "main.strict_anchor_pretrain_loss" if args.stage == "P0_anchor_only" else "main.joint_anchor_sync_loss",
    }, indent=2), encoding="utf-8")


def train_b3(args: Any) -> Path:
    if args.stage not in {"P0_anchor_only", "P1-C3"}:
        raise ContractError("B3 supports only P0_anchor_only or P1-C3")
    if args.stage == "P0_anchor_only" and args.fusion_mode != "none":
        raise ContractError("P0_anchor_only must use fusion_mode=none")
    if args.stage == "P1-C3" and args.fusion_mode != "film_gated_residual":
        raise ContractError("B3 P1-C3 must use fusion_mode=film_gated_residual")
    if args.stage == "P1-C3" and not args.p0_checkpoint:
        raise ContractError("P1-C3 requires --p0-checkpoint")
    seed_everything(args.seed, deterministic=True)
    config_path = Path(args.config)
    if args.task_id == "task2" and args.stage == "P1-C3" and not args.context_source_type:
        raise ContractError("Task 2 B3 runs require one --context-source-type")
    preprocessor = fit_b3_preprocessor(config_path, args.task_id, args.body_scale_variant,
                                        args.context_channel_indices,
                                        args.context_source_type if args.stage == "P1-C3" else None)
    output_dir = Path(args.output_dir)
    model = B3Model(fusion_mode=args.fusion_mode, transformer_layers=args.transformer_layers,
                    dropout=args.dropout, context_dropout=args.context_dropout,
                    source_dropout=args.source_dropout).to(args.device)

    if args.stage == "P0_anchor_only":
        train_dataset: Any = B3StrictDataset(config_path, preprocessor)
    else:
        checkpoint = torch.load(Path(args.p0_checkpoint), map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "P0_anchor_only" or checkpoint.get("task_id") != args.task_id:
            raise ContractError("--p0-checkpoint must be a B3 P0 checkpoint for the same task")
        if checkpoint.get("architecture_id") != model.architecture_id:
            raise ContractError(
                "architecture mismatch: P0 checkpoint has a different architecture_id; "
                f"expected {model.architecture_id!r}"
            )
        if checkpoint.get("architecture_config_hash") != model.architecture_config_hash:
            raise ContractError(
                "architecture mismatch: P0 checkpoint architecture_config_hash is incompatible "
                f"with {model.architecture_config_hash}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        train_dataset = B3JointDataset(config_path, args.task_id, "train", preprocessor,
                                       args.body_scale_variant, args.context_channel_indices,
                                       args.context_source_type)
    validation_dataset = B3JointDataset(config_path, args.task_id, "validation", preprocessor,
                                        args.body_scale_variant, args.context_channel_indices,
                                        args.context_source_type)
    _save_run_metadata(output_dir, args, preprocessor, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.anchor_lr, weight_decay=args.weight_decay) if args.stage == "P0_anchor_only" else _optimizer(model, args.anchor_lr, args.context_lr, args.weight_decay)
    train_loader = _loader(train_dataset, args.batch_size, True, args.seed)
    history: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_checkpoint = output_dir / "b3_best.pt"
    for epoch in range(args.epochs):
        if args.stage == "P1-C3":
            _set_anchor_frozen(model, epoch < args.freeze_anchor_epochs)
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            moved = _move(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, moved)
            if args.stage == "P0_anchor_only":
                loss = strict_anchor_pretrain_loss(prediction, moved["target"], moved["anchor_i"])
            else:
                loss = joint_anchor_sync_loss(prediction, moved["target"], moved["anchor_i"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = validate_v0(model, validation_dataset, preprocessor.scale_uV_by_source["d12"],
                                 args.task_id, args.device, batch_size=args.batch_size)
        metric_name = "task1_r1" if args.task_id == "task1" else "task2_r2"
        metric = float(validation["summary"]["r_submit_12"])
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": validation["summary"]}
        if args.stage == "P1-C3":
            shuffled = validate_v0(model, validation_dataset, preprocessor.scale_uV_by_source["d12"],
                                   args.task_id, args.device, batch_size=args.batch_size, shuffled_context=True)
            row["shuffled_context"] = {"r_submit_12": float(shuffled["summary"]["r_submit_12"]),
                                        "r_missing11": float(shuffled["summary"]["r_missing11"])}
        history.append(row)
        if metric > best_value:
            best_value = metric
            torch.save({"model": model.state_dict(), "task_id": args.task_id, "stage": args.stage,
                        "fusion_mode": args.fusion_mode, "parameter_count": model.parameter_count,
                        "architecture_id": model.architecture_id,
                        "architecture_config_hash": model.architecture_config_hash,
                        "transformer_layers": args.transformer_layers,
                        "best_metric": metric, "epoch": epoch + 1}, best_checkpoint)
            np.save(output_dir / "prediction_raw.npy", validation["prediction_raw"])
            np.save(output_dir / "prediction_submit.npy", validation["prediction_submit"])
            np.save(output_dir / "validation_target.npy", validation["target_raw"])
            np.save(output_dir / "anchor_i_raw.npy", validation["anchor_raw"])
            (output_dir / "validation_summary.json").write_text(json.dumps(validation["summary"], indent=2), encoding="utf-8")
            if args.task_id == "task2":
                (output_dir / "task2_diagnostics.json").write_text(json.dumps({
                    "subject_macro": validation["task2_subject_rows"],
                    "machine_body_stratified": validation["task2_device_rows"],
                    "generated_leads": ["V1", "V2", "V3", "V4", "V5", "V6"],
                }, indent=2), encoding="utf-8")
        print(f"epoch={epoch + 1} loss={row['train_loss']:.6f} {metric_name}={metric:.6f}")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_checkpoint
