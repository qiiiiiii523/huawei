"""B1 task1 route-A L0/L1 trainer with resumable, auditable outputs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg12gen.dataset import UnifiedECGDataset
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.evaluate import evaluate_predictions, write_report
from ecg12gen.losses import (masked_huber_loss, pair_invariant_stat_loss,
                             physiology_constraint_loss, spectral_stat_loss)
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.training import seed_everything
from b1.data import ROOT, local_config
from b1.model import LightweightUNet


def _load_raw_arrays(config, task: str, split: str, dataset: UnifiedECGDataset):
    directory = config.path(f"{task}_output")
    x = np.load(directory / f"{task}_{split}_input.npy", mmap_mode="r")
    y = np.load(directory / f"{task}_{split}_target.npy", mmap_mode="r")
    indices = np.asarray(dataset._indices, dtype=np.int64)
    return np.asarray(x[indices], dtype=np.float32), np.asarray(y[indices], dtype=np.float32), indices


def _prepare(config):
    train_ds = UnifiedECGDataset(config, "task1", "train")
    val_ds = UnifiedECGDataset(config, "task1", "validation")
    x_train, y_train, _ = _load_raw_arrays(config, "task1", "train", train_ds)
    x_val, y_val, val_indices = _load_raw_arrays(config, "task1", "validation", val_ds)
    pp_cfg = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
    pre = ECGPreprocessor.fit(pp_cfg, {"watch_ecg": x_train, "d12": y_train})
    x_train_model, _, _ = pre.transform_batch(x_train, "watch_ecg")
    y_train_model, _, _ = pre.transform_batch(y_train, "d12")
    x_val_model, _, _ = pre.transform_batch(x_val, "watch_ecg")
    y_val_model, _, _ = pre.transform_batch(y_val, "d12")
    # Convert the observed watch morphology into the canonical d12 model space.
    d12_scale = pre.scale_uV_by_source["d12"]
    watch_scale = pre.scale_uV_by_source["watch_ecg"]
    observed_train = np.zeros((len(x_train_model), 12, 5000), dtype=np.float32)
    observed_val = np.zeros((len(x_val_model), 12, 5000), dtype=np.float32)
    observed_train[:, 0] = x_train_model[:, 0] * watch_scale[0] / d12_scale[0]
    observed_val[:, 0] = x_val_model[:, 0] * watch_scale[0] / d12_scale[0]

    # Independent strict-train d12 reference bank for the weak-pair spectral term.
    strict = StrictD12PretrainDataset(config, "d12_i_pretrain")
    grouped: dict[str, list[int]] = {}
    for row in strict.rows:
        grouped.setdefault(row["source_task_id"], []).append(int(row["source_array_index"]))
    bank_parts = []
    for task, indices in grouped.items():
        directory = config.path(f"{task}_output")
        raw = np.asarray(np.load(directory / f"{task}_train_target.npy", mmap_mode="r")[indices], dtype=np.float32)
        bank_parts.append(pre.transform_batch(raw, "d12")[0])
    strict_bank = np.concatenate(bank_parts, axis=0)
    return (x_train, y_train, x_train_model, y_train_model, observed_train,
            x_val, y_val, x_val_model, y_val_model, observed_val,
            val_indices, pre, strict_bank)


def _save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_loss_log(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(path: Path, model, optimizer, epoch, best_r1, history, args):
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best_r1": best_r1, "history": history, "args": vars(args),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state()}, path)


def _load_checkpoint(path: Path, model, optimizer):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
    return int(state["epoch"]), float(state["best_r1"]), list(state.get("history", []))


def _evaluate(model, tensors, pre, raw_target, raw_input, val_indices, metadata_path, out_dir, epoch, device):
    x_model, observed, y_model = tensors
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_model), 16):
            batch = x_model[start:start + 16].to(device)
            preds.append(model(batch).cpu().numpy())
    pred_model = np.concatenate(preds, axis=0)
    pred_uV = pred_model * pre.scale_uV_by_source["d12"][None, :, None]
    overall, details = evaluate_predictions(pred_uV, raw_target, "task1")
    copied = pred_uV.copy()
    copied[:, 0] = raw_input[:, 0]
    copy_overall, copy_details = evaluate_predictions(copied, raw_target, "task1")
    eval_dir = out_dir / "evaluation" / f"epoch_{epoch:03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_report(eval_dir / "raw", overall, details)
    write_report(eval_dir / "copy_at_eval", copy_overall, copy_details)
    np.save(eval_dir / "prediction_raw_uV.npy", pred_uV)
    np.save(eval_dir / "prediction_copy_at_eval_uV.npy", copied)
    np.save(eval_dir / "validation_target_uV.npy", raw_target)
    _save_json(eval_dir / "summary.json", {"raw": overall, "copy_at_eval": copy_overall,
                                             "copy_policy": "replace_lead_I_with_watch_input"})
    return overall, copy_overall


def run(args):
    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    config = local_config()
    out_dir = ROOT / "results" / "b1" / f"B1_T1_A_{args.loss_variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = _prepare(config)
    (x_train_raw, y_train_raw, x_train, y_train, observed_train,
     x_val_raw, y_val_raw, x_val, y_val, observed_val, val_indices, pre, strict_bank) = arrays
    model = LightweightUNet(1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    start_epoch, best_r1 = 0, float("-inf")
    latest = out_dir / "checkpoint_latest.pt"
    if args.resume and latest.exists():
        start_epoch, best_r1, history = _load_checkpoint(latest, model, optimizer)
    run_info = {"experiment": f"B1_T1_A_{args.loss_variant}", "loss_variant": args.loss_variant,
                "route": "A_raw_weak", "commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "seed": args.seed, "epochs": args.epochs, "train_windows": len(x_train), "validation_windows": len(x_val_raw),
                "strict_reference_windows": len(strict_bank), "raw_baseline_policy": "zero_model_baseline",
                "preprocessing_scales_uV": {k: v.tolist() for k, v in pre.scale_uV_by_source.items()},
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "device": str(device), "torch_cuda_available": torch.cuda.is_available()}
    _save_json(out_dir / "run_info.json", run_info)
    _save_json(out_dir / "resolved_config.json", {"base": str(ROOT / "b1/task1_a_config.yaml"), "args": vars(args)})
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(observed_train), torch.from_numpy(y_train),
                          torch.from_numpy(np.ones((len(x_train), 12), dtype=bool)))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    strict_bank_t = torch.from_numpy(strict_bank).to(device)
    d12_scale_t = torch.from_numpy(pre.scale_uV_by_source["d12"]).view(1, 12, 1).to(device)
    elapsed_start = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "observed": 0.0, "spectral": 0.0, "physiology": 0.0, "pair_invariant": 0.0}
        n_batches = 0
        for xb, observed, yb, mask in loader:
            xb, observed, yb, mask = (xb.to(device), observed.to(device), yb.to(device), mask.to(device))
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            observed_loss = masked_huber_loss(pred, observed, mask)
            refs = strict_bank_t[torch.randint(0, len(strict_bank_t), (len(xb),))]
            spectral = spectral_stat_loss(pred, refs)
            physiology = physiology_constraint_loss(pred * d12_scale_t)
            pair = pair_invariant_stat_loss(pred, yb) if args.loss_variant == "L1" else pred.new_zeros(())
            loss = .10 * observed_loss + .20 * spectral + .05 * physiology + (.20 * pair if args.loss_variant == "L1" else 0.0)
            loss.backward()
            optimizer.step()
            sums["loss"] += float(loss.detach()); sums["observed"] += float(observed_loss.detach())
            sums["spectral"] += float(spectral.detach()); sums["physiology"] += float(physiology.detach()); sums["pair_invariant"] += float(pair.detach())
            n_batches += 1
        train_row = {"epoch": epoch, **{k: v / n_batches for k, v in sums.items()}}
        raw, copied = _evaluate(model, (torch.from_numpy(x_val), torch.from_numpy(observed_val), torch.from_numpy(y_val)), pre, y_val_raw, x_val_raw, val_indices, ROOT / "task1_output/task1_window_metadata.csv", out_dir, epoch, device)
        train_row.update({"validation_task1_r1": raw["task1_r1"], "copy_at_eval_task1_r1": copied["task1_r1"], "epoch_seconds": (time.perf_counter() - elapsed_start) / epoch})
        history.append(train_row)
        if raw["task1_r1"] > best_r1:
            best_r1 = raw["task1_r1"]
            _checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, epoch, best_r1, history, args)
        _write_loss_log(out_dir / "training_log.csv", history)
        _checkpoint(latest, model, optimizer, epoch, best_r1, history, args)
        print(json.dumps(train_row, ensure_ascii=False), flush=True)
    _save_json(out_dir / "run_summary.json", {"status": "completed", "best_task1_r1": best_r1, "history": history})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss-variant", choices=("L0", "L1"), required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=.0001)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
