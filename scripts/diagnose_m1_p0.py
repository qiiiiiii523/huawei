"""Read-only/standalone diagnostics for an M1-P0 checkpoint.

The script never edits the checkpoint or shared scoring code.  It writes only
to ``--output-dir`` and labels smoke checkpoints explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.contracts import D12_LEADS
from ecg12gen.evaluate import evaluate_joint_anchor_predictions
from ecg12gen.m1_axial import M1AxialLeadTimeModel, architecture_config_hash
from ecg12gen.m1_axial_train import forward, loader, move, _loss
from ecg12gen.m1_data import build_m1_datasets, fit_m1_preprocessor, m1_collate
from ecg12gen.training import seed_everything


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                              capture_output=True).stdout.strip()
    return {"repository": str(ROOT), "branch": run("branch", "--show-current"),
            "status_porcelain": run("status", "--porcelain").splitlines()}


def subject_audit(train_ds, validation_ds) -> dict[str, Any]:
    train = {str(x.meta["subject_id"]) for x in train_ds.samples}
    validation = {str(x.meta["subject_id"]) for x in validation_ds.samples}
    return {"train_subjects": len(train), "validation_subjects": len(validation),
            "overlap_count": len(train & validation), "overlap_subject_ids": sorted(train & validation)}


def load_model(checkpoint_path: Path, task_id: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("architecture_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no architecture_config")
    if checkpoint.get("architecture_config_hash") != architecture_config_hash(config):
        raise ValueError("checkpoint architecture hash mismatch")
    if checkpoint.get("stage") != "P0_anchor_only" or checkpoint.get("fusion_mode", "none") != "none":
        raise ValueError("diagnostic requires an M1 P0 checkpoint")
    model = M1AxialLeadTimeModel(fusion_mode="none", task_id=task_id, config=config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def infer(model, dataset, scale: np.ndarray, device: torch.device, batch_size: int):
    predictions, targets, anchors, metas = [], [], [], []
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                    collate_fn=m1_collate)
    with torch.inference_mode():
        for raw in dl:
            batch = move(raw, device)
            predictions.append(forward(model, batch).cpu().numpy())
            targets.append(raw["raw_target_uV"].numpy())
            anchors.append(raw["raw_anchor_uV"].numpy())
            metas.extend(raw["meta"])
    prediction = np.concatenate(predictions).astype(np.float32) * scale[None, :, None]
    return prediction, np.concatenate(targets), np.concatenate(anchors), metas


def centered(values: np.ndarray) -> np.ndarray:
    return values - np.median(values, axis=2, keepdims=True)


def metric_rows(split: str, prediction: np.ndarray, target: np.ndarray, anchor: np.ndarray):
    summary, raw_details, _, _ = evaluate_joint_anchor_predictions(prediction, target, anchor, "task1")
    centered_summary, centered_details, _, _ = evaluate_joint_anchor_predictions(
        centered(prediction), centered(target), centered(anchor), "task1")
    rows = []
    for raw, ctr in zip(raw_details, centered_details):
        rows.append({"split": split, "lead": raw["lead"], "r_raw_uV": raw["pearson_r"],
                     "rmse_raw_uV": raw["rmse_uV"], "r_centered_uV": ctr["pearson_r"],
                     "rmse_centered_uV": ctr["rmse_uV"]})
    compact = {"split": split, "n_windows": len(prediction),
               "r_missing11_raw_uV": summary["r_missing11"],
               "r_missing11_centered_uV": centered_summary["r_missing11"],
               "mean_rmse_raw_uV": summary["twelve_lead_mean_rmse_uV"],
               "mean_rmse_centered_uV": centered_summary["twelve_lead_mean_rmse_uV"]}
    return compact, rows


def sample_rows(split: str, prediction: np.ndarray, target: np.ndarray, metas: list[dict[str, Any]]):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    per_lead_rmse = np.sqrt(np.mean(error * error, axis=2))
    rows = []
    for i, (lead_rmse, meta) in enumerate(zip(per_lead_rmse, metas)):
        target_min = target[i].min(axis=1); target_max = target[i].max(axis=1)
        rows.append({"split": split, "index": i, "subject_id": meta.get("subject_id", ""),
                     "window_id": meta.get("window_id", ""), "target_record_id": meta.get("target_record_id", ""),
                     "missing11_rmse_uV": float(lead_rmse[1:].mean()), "v3_rmse_uV": float(lead_rmse[8]),
                     "v3_target_min_uV": float(target_min[8]), "v3_target_max_uV": float(target_max[8]),
                     "v3_target_polarity": "positive" if abs(target_max[8]) >= abs(target_min[8]) else "negative"})
    return rows, per_lead_rmse


def plot_cases(output: Path, split: str, prediction: np.ndarray, target: np.ndarray,
               metas: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    selected = []
    for key in ("v3_rmse_uV", "missing11_rmse_uV"):
        for row in sorted(rows, key=lambda x: float(x[key]), reverse=True)[:3]:
            if int(row["index"]) not in selected:
                selected.append(int(row["index"]))
    lead_indices = [0, 1, 7, 8, 9]
    time = np.arange(target.shape[2]) / 500.0
    for rank, index in enumerate(selected):
        fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        for ax, lead_index in zip(axes, lead_indices):
            ax.plot(time, target[index, lead_index], lw=.8, label="true")
            ax.plot(time, prediction[index, lead_index], lw=.8, alpha=.8, label="pred")
            lo, hi = float(target[index, lead_index].min()), float(target[index, lead_index].max())
            polarity = "positive" if abs(hi) >= abs(lo) else "negative"
            ax.set_ylabel(D12_LEADS[lead_index])
            ax.set_title(f"true range [{lo:.1f}, {hi:.1f}] uV; dominant polarity={polarity}", fontsize=9)
            ax.grid(alpha=.2)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("seconds")
        meta = metas[index]
        fig.suptitle(f"{split} index={index} subject={meta.get('subject_id')} window={meta.get('window_id')}")
        fig.tight_layout()
        fig.savefig(output / f"{split}_case_{rank:02d}_index_{index}.png", dpi=130)
        plt.close(fig)


def overfit_probe(dataset, config: dict[str, Any], scale: np.ndarray, device: torch.device,
                  windows: int, epochs: int, lr: float, output: Path) -> dict[str, Any]:
    seed_everything(42, deterministic=True)
    subset = Subset(dataset, list(range(min(windows, len(dataset)))))
    model = M1AxialLeadTimeModel(fusion_mode="none", task_id="task1", config=config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    dl = DataLoader(subset, batch_size=min(4, len(subset)), shuffle=True, num_workers=0,
                    collate_fn=m1_collate, generator=torch.Generator().manual_seed(42))
    history = []
    for epoch in range(epochs + 1):
        model.eval()
        pred, target, anchor, _ = infer(model, subset, scale, device, min(4, len(subset)))
        summary, _, _, _ = evaluate_joint_anchor_predictions(pred, target, anchor, "task1")
        centered_summary, _, _, _ = evaluate_joint_anchor_predictions(centered(pred), centered(target), centered(anchor), "task1")
        row = {"epoch": epoch, "r_missing11_raw_uV": summary["r_missing11"],
               "r_missing11_centered_uV": centered_summary["r_missing11"],
               "rmse_raw_uV": summary["twelve_lead_mean_rmse_uV"],
               "rmse_centered_uV": centered_summary["twelve_lead_mean_rmse_uV"]}
        history.append(row)
        if epoch == epochs:
            break
        model.train()
        losses = []
        for raw in dl:
            batch = move(raw, device); opt.zero_grad(set_to_none=True)
            loss = _loss(model, forward(model, batch), batch, torch.as_tensor(scale, device=device))
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        row["train_objective"] = float(np.mean(losses))
    write_csv(output / "overfit_history.csv", history)
    return {"windows": len(subset), "epochs": epochs, "learning_rate": lr,
            "initial": history[0], "final": history[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "common.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overfit-windows", type=int, default=8)
    parser.add_argument("--overfit-epochs", type=int, default=20)
    parser.add_argument("--overfit-lr", type=float, default=1e-3)
    parser.add_argument("--skip-overfit", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    state = git_state(); state["checkpoint"] = str(args.checkpoint.resolve())
    state["checkpoint_is_smoke"] = "smoke" in str(args.checkpoint).lower()
    dump_json(output / "run_context.json", state)

    pre = fit_m1_preprocessor(args.config, "task1", "P0_anchor_only", None)
    train, validation = build_m1_datasets(args.config, "task1", "P0_anchor_only", None, pre)
    scale = pre.scale_uV_by_source["d12"].astype(np.float32)
    audit = subject_audit(train, validation)
    with args.config.open(encoding="utf-8") as handle:
        common = yaml.safe_load(handle)
    audit.update({"canonical_lead_order": list(D12_LEADS),
                  "configured_lead_order": common["signal"]["twelve_lead_order"],
                  "configured_unit": common["signal"]["ecg_unit"],
                  "normalization": "per-window/per-lead median subtraction, train-fitted d12 P5-P95 scale, clip +/-12",
                  "inverse_used_by_m1_validation": "multiply d12 scale only; no per-window baseline is restored or predicted",
                  "r_missing11_indices": list(range(1, 12)), "r_missing11_leads": list(D12_LEADS[1:])})
    dump_json(output / "protocol_audit.json", audit)

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint.resolve(), "task1", device)
    checkpoint_scale = np.asarray(checkpoint["target_d12_scale_uV"], dtype=np.float32)
    if not np.allclose(scale, checkpoint_scale, rtol=0, atol=1e-5):
        raise ValueError("checkpoint scale differs from current task1 train-fitted scale")

    all_metric_rows, split_rows, sample_table = [], [], []
    cached = {}
    for split, dataset in (("train", train), ("validation", validation)):
        pred, target, anchor, metas = infer(model, dataset, scale, device, args.batch_size)
        cached[split] = (pred, target, metas)
        compact, rows = metric_rows(split, pred, target, anchor)
        split_rows.append(compact); all_metric_rows.extend(rows)
        samples, _ = sample_rows(split, pred, target, metas); sample_table.extend(samples)
        plot_cases(output, split, pred, target, metas, samples)
    write_csv(output / "train_vs_validation_by_lead.csv", all_metric_rows)
    write_csv(output / "train_vs_validation_summary.csv", split_rows)
    write_csv(output / "sample_errors.csv", sample_table)

    overfit = None
    if not args.skip_overfit:
        overfit = overfit_probe(train, checkpoint["architecture_config"], scale, device,
                                args.overfit_windows, args.overfit_epochs, args.overfit_lr, output)
    dump_json(output / "diagnostic_summary.json", {"run_context": state, "protocol_audit": audit,
                                                    "split_metrics": split_rows, "overfit": overfit})
    print(json.dumps({"output_dir": str(output), "checkpoint_is_smoke": state["checkpoint_is_smoke"],
                      "split_metrics": split_rows, "overfit": overfit}, indent=2))


if __name__ == "__main__":
    main()
