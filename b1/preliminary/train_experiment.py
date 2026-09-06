"""B1 matrix runner for task-1 and task-2 ablations.

This runner keeps the frozen public protocol intact while selecting one
experiment arm at runtime.  Raw arrays are read-only; every experiment gets
its own results directory and records its resolved data arm, route, loss and
device.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from b1.data import ROOT, local_config
from b1.model import LightweightUNet
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.evaluate import evaluate_centered_diagnostic, evaluate_predictions, write_report
from ecg12gen.losses import masked_huber_loss, pair_invariant_stat_loss, physiology_constraint_loss, spectral_stat_loss
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.training import seed_everything


def _json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _load_strict_bank(config, pre, mode: str) -> np.ndarray:
    strict = StrictD12PretrainDataset(config, mode)
    groups: dict[str, list[int]] = {}
    for row in strict.rows:
        groups.setdefault(row["source_task_id"], []).append(int(row["source_array_index"]))
    parts = []
    for task, indices in groups.items():
        raw = np.asarray(np.load(config.path(f"{task}_output") / f"{task}_train_target.npy", mmap_mode="r")[indices], dtype=np.float32)
        parts.append(pre.transform_batch(raw, "d12")[0])
    return np.concatenate(parts, axis=0)


def _task1_arrays(config, pre, pseudo: str | None = None):
    from ecg12gen.dataset import UnifiedECGDataset
    train_ds, val_ds = UnifiedECGDataset(config, "task1", "train"), UnifiedECGDataset(config, "task1", "validation")
    def load(ds, split):
        d = config.path("task1_output")
        idx = np.asarray(ds._indices, dtype=np.int64)
        x = np.asarray(np.load(d / f"task1_{split}_input.npy", mmap_mode="r")[idx], dtype=np.float32)
        y = np.asarray(np.load(d / f"task1_{split}_target.npy", mmap_mode="r")[idx], dtype=np.float32)
        return x, y
    xtr, ytr = load(train_ds, "train"); xv, yv = load(val_ds, "validation")
    xtr_m = pre.transform_batch(xtr, "watch_ecg")[0]; ytr_m = pre.transform_batch(ytr, "d12")[0]
    xv_m = pre.transform_batch(xv, "watch_ecg")[0]; yv_m = pre.transform_batch(yv, "d12")[0]
    if pseudo:
        root = ROOT.parent / ("task1_rpeak_pseudo_output" if pseudo == "C1" else "task1_rpeak_pseudo_output_c2")
        root = root / root.name
        xp = np.asarray(np.load(root / "task1_rpeak_train_input.npy", mmap_mode="r"), dtype=np.float32)
        yp = np.asarray(np.load(root / "task1_rpeak_train_target.npy", mmap_mode="r"), dtype=np.float32)
        xp_m = pre.transform_batch(xp, "watch_ecg")[0]; yp_m = pre.transform_batch(yp, "d12")[0]
        return xtr, ytr, xtr_m, ytr_m, xv, yv, xv_m, yv_m, xp_m, yp_m
    return xtr, ytr, xtr_m, ytr_m, xv, yv, xv_m, yv_m, None, None


def _task2_arrays(config, pre, variant: str):
    d = config.path("task2_output")
    def rows(split):
        with (d / "task2_window_metadata.csv").open(encoding="utf-8-sig", newline="") as f:
            all_rows = list(csv.DictReader(f))
        return sorted([r for r in all_rows if r["split"] == split], key=lambda r: int(r["array_index"]))
    if variant == "A":
        out = []
        for split in ("train", "validation"):
            rs = rows(split); x = np.asarray(np.load(d / f"task2_{split}_input.npy", mmap_mode="r"), dtype=np.float32)
            y = np.asarray(np.load(d / f"task2_{split}_target.npy", mmap_mode="r"), dtype=np.float32)
            out.append((x, y, rs))
        return out
    b = config.path("task2_body_scale_ablation")
    out = []
    for split in ("train", "validation"):
        with (b / "window_metadata_B_raw.csv").open(encoding="utf-8-sig", newline="") as f:
            rs = sorted([r for r in csv.DictReader(f) if r["split"] == split], key=lambda r: int(r["local_array_index"]))
        x = np.asarray(np.load(b / f"body_scale_{split}_input_B_raw_detrended_0p2Hz.npy", mmap_mode="r"), dtype=np.float32)
        yall = np.asarray(np.load(d / f"task2_{split}_target.npy", mmap_mode="r"), dtype=np.float32)
        y = yall[np.asarray([int(r["canonical_array_index"]) for r in rs], dtype=np.int64)]
        out.append((x, y, rs))
    return out


def _transform_task2(pre, x, y, rows):
    x_m = np.empty_like(x, dtype=np.float32)
    for source in ("ecg_machine_d6", "body_scale_d6"):
        idx = np.asarray([i for i, r in enumerate(rows) if r.get("input_type") == source], dtype=np.int64)
        if idx.size:
            x_m[idx] = pre.transform_batch(x[idx], source)[0]
    y_m = pre.transform_batch(y, "d12")[0]
    return x_m, y_m


def _observed_model_view(pre, x_m, rows, task):
    """Put visible input leads into canonical d12 model scale."""
    n, c, t = x_m.shape
    out = np.zeros((n, 12, t), dtype=np.float32)
    if task == "task1":
        out[:, 0] = x_m[:, 0] * pre.scale_uV_by_source["watch_ecg"][0] / pre.scale_uV_by_source["d12"][0]
        return out
    for i, row in enumerate(rows):
        source = row.get("input_type", "body_scale_d6")
        scale = pre.scale_uV_by_source[source]
        out[i, :6] = x_m[i] * scale[:, None] / pre.scale_uV_by_source["d12"][:6, None]
    return out


def _eval(model, x_m, raw_x, raw_y, pre, task, out, epoch, device, metadata=None, save_arrays=False):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(x_m), 16):
            preds.append(model(torch.from_numpy(x_m[i:i+16]).to(device)).cpu().numpy())
    pred = np.concatenate(preds) * pre.scale_uV_by_source["d12"][None, :, None]
    overall, details = evaluate_predictions(pred, raw_y, task)
    copied = pred.copy(); copied[:, : (1 if task == "task1" else 6)] = raw_x[:, : (1 if task == "task1" else 6)]
    copy_overall, copy_details = evaluate_predictions(copied, raw_y, task)
    ed = out / "evaluation" / f"epoch_{epoch:03d}"; ed.mkdir(parents=True, exist_ok=True)
    write_report(ed / "raw", overall, details); write_report(ed / "copy_at_eval", copy_overall, copy_details)
    centered, centered_details = evaluate_centered_diagnostic(pred, raw_y, task)
    write_report(ed / "centered_diagnostic", centered, centered_details)
    if save_arrays:
        np.save(ed / "prediction_raw_uV.npy", pred); np.save(ed / "prediction_copy_at_eval_uV.npy", copied); np.save(ed / "validation_target_uV.npy", raw_y)
    _json(ed / "summary.json", {"raw": overall, "copy_at_eval": copy_overall, "centered_diagnostic": centered})
    return overall, copy_overall


def _fit_pre(task, variant):
    config = local_config(); pp = PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml")
    if task == "task1":
        from ecg12gen.dataset import UnifiedECGDataset
        ds = UnifiedECGDataset(config, "task1", "train"); d = config.path("task1_output"); idx = np.asarray(ds._indices, dtype=np.int64)
        y = np.asarray(np.load(d / "task1_train_target.npy", mmap_mode="r")[idx], dtype=np.float32)
        x = np.asarray(np.load(d / "task1_train_input.npy", mmap_mode="r")[idx], dtype=np.float32)
        return config, ECGPreprocessor.fit(pp, {"watch_ecg": x, "d12": y})
    train, val = _task2_arrays(config, None, variant)
    x, y, rows = train
    fit = {"d12": y}
    for source in ("ecg_machine_d6", "body_scale_d6"):
        idx = np.asarray([i for i, r in enumerate(rows) if r.get("input_type") == source], dtype=np.int64)
        if idx.size: fit[source] = x[idx]
    if variant == "B": fit = {"d12": y, "body_scale_d6": x}
    return config, ECGPreprocessor.fit(pp, fit)


def run(args):
    torch.set_num_threads(args.torch_threads); seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    config, pre = _fit_pre(args.task, args.data_variant)
    if args.task == "task1":
        values = _task1_arrays(config, pre, args.route if args.route in {"C1", "C2"} else None)
        xtr_raw, ytr_raw, xtr, ytr, xv_raw, yv_raw, xv, yv, xp, yp = values
        in_ch = 1; task = "task1"; route = args.route
        obs_tr = _observed_model_view(pre, xtr, None, task)
        obs_v = _observed_model_view(pre, xv, None, task)
    else:
        train, val = _task2_arrays(config, pre, args.data_variant)
        xtr_raw, ytr_raw, rows_tr = train; xv_raw, yv_raw, rows_v = val
        xtr, ytr = _transform_task2(pre, xtr_raw, ytr_raw, rows_tr); xv, yv = _transform_task2(pre, xv_raw, yv_raw, rows_v)
        obs_tr = _observed_model_view(pre, xtr, rows_tr, task="task2")
        obs_v = _observed_model_view(pre, xv, rows_v, task="task2")
        xp = yp = None; in_ch = 6; task = "task2"; route = args.route
    out = ROOT / "results" / "b1" / args.experiment; out.mkdir(parents=True, exist_ok=True)
    model = LightweightUNet(in_ch).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    strict = _load_strict_bank(config, pre, "d12_i_pretrain" if task == "task1" else "d12_six_pretrain")
    strict_t = torch.from_numpy(strict).to(device); scale_t = torch.from_numpy(pre.scale_uV_by_source["d12"]).view(1, 12, 1).to(device)
    history = []
    train_x, train_y = xtr, ytr
    if xp is not None:
        train_x = np.concatenate([train_x, xp]); train_y = np.concatenate([train_y, yp])
    # Pseudo-pair windows do not carry a separate observed view; use the same
    # visible input convention as the weak branch for their consistency term.
    train_obs = np.concatenate([obs_tr, np.zeros((len(train_x) - len(obs_tr), 12, 5000), dtype=np.float32)]) if len(train_x) > len(obs_tr) else obs_tr
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_obs), torch.ones((len(train_x), 12), dtype=torch.bool))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # Route B/C has a documented strict/pseudo warm-start before weak fine-tuning.
    if route in {"B", "C1", "C2"}:
        for epoch in range(1, args.pretrain_epochs + 1):
            model.train()
            for xb, yb, mask in DataLoader(TensorDataset(torch.from_numpy(strict[:, :in_ch]), torch.from_numpy(strict), torch.ones((len(strict), 12), dtype=torch.bool)), batch_size=args.batch_size, shuffle=True):
                xb, yb = xb.to(device), yb.to(device); opt.zero_grad(set_to_none=True); pred = model(xb)
                loss = masked_huber_loss(pred, yb, mask.to(device)) + .10 * physiology_constraint_loss(pred)
                loss.backward(); opt.step()
    start = time.perf_counter()
    best_r = float("-inf"); best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); sums = {"loss": 0.0, "observed": 0.0, "spectral": 0.0, "physiology": 0.0, "pair_invariant": 0.0}; nb = 0
        for xb, yb, observed, mask in loader:
            xb, yb, observed, mask = xb.to(device), yb.to(device), observed.to(device), mask.to(device); opt.zero_grad(set_to_none=True); pred = model(xb)
            observed_loss = masked_huber_loss(pred, observed, torch.cat([torch.ones((len(xb),in_ch),device=device),torch.zeros((len(xb),12-in_ch),device=device)],1))
            spectral = spectral_stat_loss(pred, strict_t[torch.randint(0, len(strict_t), (len(xb),))])
            phys = physiology_constraint_loss(pred); pair = pair_invariant_stat_loss(pred, yb) if args.loss_variant == "L1" else pred.new_zeros(())
            loss = .10 * observed_loss + .20 * spectral + .05 * phys + (.20 * pair if args.loss_variant == "L1" else 0.0)
            loss.backward(); opt.step(); nb += 1
            for k, v in (("loss",loss),("observed",observed_loss),("spectral",spectral),("physiology",phys),("pair_invariant",pair)): sums[k] += float(v.detach())
        raw, copied = _eval(model, xv, xv_raw, yv_raw, pre, task, out, epoch, device, save_arrays=False)
        row = {"epoch": epoch, **{k:v/nb for k,v in sums.items()}, "validation_official_r": raw["task1_r1"] if task=="task1" else raw["task2_r2"], "copy_at_eval_r": copied["task1_r1"] if task=="task1" else copied["task2_r2"], "epoch_seconds": (time.perf_counter()-start)/epoch}
        history.append(row); _csv(out / "training_log.csv", history); print(json.dumps(row), flush=True)
        metric = float(row["validation_official_r"])
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "metric": metric}, out / "checkpoint_latest.pt")
        if metric > best_r:
            best_r, best_epoch = metric, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "metric": metric}, out / "checkpoint_best.pt")
    # Re-evaluate the selected official checkpoint and retain only one set of
    # full validation arrays per experiment, keeping the 50 GB data disk safe.
    state = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    _eval(model, xv, xv_raw, yv_raw, pre, task, out, best_epoch, device, save_arrays=True)
    _json(out / "run_info.json", {"experiment":args.experiment,"task":task,"route":route,"data_variant":args.data_variant,"loss_variant":args.loss_variant,"device":str(device),"torch":torch.__version__,"cuda":torch.cuda.is_available(),"train_windows":len(train_x),"validation_windows":len(xv_raw),"pretrain_epochs":args.pretrain_epochs,"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()})
    _json(out / "run_summary.json", {"status":"completed","best_epoch":best_epoch,"best_validation_official_r":best_r,"history":history})


def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=("task1","task2"),required=True); p.add_argument("--route",choices=("A","B","C1","C2"),required=True); p.add_argument("--data-variant",choices=("A","B"),default="A"); p.add_argument("--loss-variant",choices=("L0","L1"),required=True); p.add_argument("--experiment",required=True); p.add_argument("--epochs",type=int,default=100); p.add_argument("--pretrain-epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--seed",type=int,default=42); p.add_argument("--deterministic",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--torch-threads",type=int,default=8); p.add_argument("--learning-rate",type=float,default=.001); p.add_argument("--weight-decay",type=float,default=.0001); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); run(p.parse_args())


if __name__ == "__main__": main()
