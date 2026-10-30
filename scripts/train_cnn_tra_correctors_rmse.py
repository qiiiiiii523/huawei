"""RMSE experiment: retrain the fine corrector, then its dependent chest-mid stage.

The strict training records generate fitting inputs. The frozen step-3000
validation cache is used only for scoring and never for gradient updates.
The fine loss adds raw-uV RMSE without mean subtraction or lead scaling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d, median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.cnn_tra_data import HuaweiSevenLeadData, LEAD_INDICES, TASKS
from ecg12gen.cnn_tra_corrector import (FineCorrector, ChestMidCorrector,
                                         compose_d12, make_features, mid_features)
from ecg12gen.cnn_tra_corrector_data import (Record, load_train_cache,
                                              load_validation_records,
                                              prepare_train_cache, sha256)
from ecg12gen.evaluate import (evaluate_record_predictions,
                                correlation_score_from_summaries)


FINE_UPDATES = 2000
MID_UPDATES = 2000
BATCH = 8
VALIDATE_EVERY = 250
SEED = 42
WEIGHT7 = torch.tensor([3., 1., 1., 1., 1., 1., 1.])
RAW_RMSE_WEIGHT_PER_UV = 0.0005


def _weighted_corr_loss(pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    r = (p * t).sum(dim=-1) / torch.sqrt(
        p.square().sum(dim=-1).clamp_min(1.) *
        t.square().sum(dim=-1).clamp_min(1.))
    active = mask * weights.to(device=pred.device, dtype=pred.dtype)[None]
    return ((1 - r) * active).sum() / active.sum().clamp_min(1.)


def _weighted_raw_rmse(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Raw waveform RMS error in uV: no centering and no per-lead scaling."""
    per_lead = torch.sqrt((pred - target).square().mean(dim=-1) + 1e-6)
    active = mask * weights.to(device=pred.device, dtype=pred.dtype)[None]
    return (per_lead * active).sum() / active.sum().clamp_min(1.)


def _fine_label(base7: np.ndarray, target7: np.ndarray) -> torch.Tensor:
    residual = target7.astype(np.float64) - base7.astype(np.float64)
    return torch.from_numpy(np.median(residual.reshape(7, 1200, 50),
                                      axis=-1).astype(np.float32))


def _mid_label(current_chest: np.ndarray, target_chest: np.ndarray) -> torch.Tensor:
    residual = target_chest.astype(np.float64) - current_chest.astype(np.float64)
    med = np.median(residual.reshape(6, 1200, 50), axis=-1)
    med = median_filter(med, size=(1, 3), mode="nearest")
    fast = gaussian_filter1d(med, 5, axis=-1, mode="reflect")
    slow = gaussian_filter1d(med, 100, axis=-1, mode="reflect")
    return F.interpolate(torch.from_numpy((fast - slow).astype(np.float32))[None],
                         size=480, mode="linear", align_corners=False)[0]


def _fine_items(records: list[Record], scale8: np.ndarray) -> list[dict]:
    items = []
    for record in records:
        target7 = record.target12[list(LEAD_INDICES)].copy()
        mask7 = record.mask12[list(LEAD_INDICES)].copy()
        items.append({"record": record,
                      "features": make_features(record.anchor, record.base7, scale8, 0.1),
                      "base7": torch.from_numpy(np.array(record.base7, copy=True)),
                      "target7": torch.from_numpy(target7),
                      "mask7": torch.from_numpy(mask7.astype(np.float32)),
                      "label": _fine_label(record.base7, target7)})
    return items


def _fine_loss(model: FineCorrector, batch: list[dict],
               device: torch.device) -> tuple[torch.Tensor, dict]:
    features = torch.stack([item["features"] for item in batch]).to(device)
    base = torch.stack([item["base7"] for item in batch]).to(device)
    target = torch.stack([item["target7"] for item in batch]).to(device)
    mask = torch.stack([item["mask7"] for item in batch]).to(device)
    correction, controls = model(features)
    prediction = base + correction
    r_loss = _weighted_corr_loss(prediction, target, mask, WEIGHT7)
    raw_rmse = _weighted_raw_rmse(prediction, target, mask, WEIGHT7)
    labels = torch.stack([item["label"] for item in batch]).to(device)
    active = mask * WEIGHT7.to(device)[None]
    trend = F.smooth_l1_loss(controls, labels / model.scales[None, :, None],
                             reduction="none")
    trend = (trend * active[:, :, None]).sum() / (active.sum().clamp_min(1.) * 1200)
    curvature = (controls[..., 2:] - 2 * controls[..., 1:-1] +
                 controls[..., :-2]).square()
    curvature = (curvature * active[:, :, None]).sum() / (
        active.sum().clamp_min(1.) * curvature.shape[-1])
    curvature *= (0.25 / 0.1) ** 4
    loss = r_loss + RAW_RMSE_WEIGHT_PER_UV * raw_rmse + 0.2 * trend + 0.01 * curvature
    return loss, {"r_loss": float(r_loss.detach()), "trend": float(trend.detach()),
                  "curvature": float(curvature.detach()),
                  "raw_rmse_uV": float(raw_rmse.detach())}


@torch.no_grad()
def _fine_prediction(model: FineCorrector, item: dict,
                     device: torch.device) -> np.ndarray:
    model.eval()
    correction, _ = model(item["features"][None].to(device))
    return item["record"].base7 + correction[0].cpu().numpy()


@torch.no_grad()
def _mid_items(records: list[Record], fine: FineCorrector,
               scale8: np.ndarray, device: torch.device) -> list[dict]:
    fine_items = _fine_items(records, scale8)
    items = []
    for index, item in enumerate(fine_items, 1):
        record = item["record"]
        fine7 = _fine_prediction(fine, item, device).astype(np.float32)
        mask6 = record.mask12[6:12].copy()
        items.append({"record": record, "fine7": fine7,
                      "features": mid_features(record.anchor, record.base7, fine7, scale8),
                      "target_chest": torch.from_numpy(np.array(record.target12[6:12], copy=True)),
                      "current_chest": torch.from_numpy(np.array(fine7[1:], copy=True)),
                      "mask6": torch.from_numpy(mask6.astype(np.float32)),
                      "label": _mid_label(fine7[1:], record.target12[6:12])})
        if index % 50 == 0 or index == len(fine_items):
            print(f"[prepare-mid] {index}/{len(fine_items)}", flush=True)
    return items


def _mid_loss(model: ChestMidCorrector, batch: list[dict],
              device: torch.device) -> tuple[torch.Tensor, dict]:
    features = torch.stack([item["features"] for item in batch]).to(device)
    current = torch.stack([item["current_chest"] for item in batch]).to(device)
    target = torch.stack([item["target_chest"] for item in batch]).to(device)
    mask = torch.stack([item["mask6"] for item in batch]).to(device)
    correction, controls = model(features)
    r_loss = _weighted_corr_loss(current + correction, target, mask,
                                 torch.ones(6))
    labels = torch.stack([item["label"] for item in batch]).to(device)
    trend = F.smooth_l1_loss(controls, labels / model.scales[None, :, None],
                             reduction="none")
    trend = (trend * mask[:, :, None]).sum() / (mask.sum().clamp_min(1.) * 480)
    loss = r_loss + 0.2 * trend
    return loss, {"r_loss": float(r_loss.detach()), "trend": float(trend.detach())}


def _assemble_validation(predictions: list[np.ndarray], records: list[Record],
                         data: HuaweiSevenLeadData) -> dict:
    arrays = {task: np.empty(data.validation_arrays[task].shape, dtype=np.float32)
              for task in TASKS}
    filled = {task: np.zeros(len(data.validation_rows[task]), dtype=bool) for task in TASKS}
    for seven, record in zip(predictions, records):
        task = record.meta["task"]
        indices = record.meta["validation_array_indices"]
        full = compose_d12(torch.from_numpy(record.anchor[None]),
                           torch.from_numpy(seven[None]))[0].numpy()
        windows = full.reshape(12, 12, 5000).transpose(1, 0, 2)
        arrays[task][indices] = windows
        filled[task][indices] = True
    if any(not filled[task].all() for task in TASKS):
        raise ValueError("validation prediction misses windows")
    summaries = {}
    for task in TASKS:
        quality = data.validation_masks[task]
        summary, _ = evaluate_record_predictions(
            arrays[task], data.validation_arrays[task], task,
            data.validation_rows[task], target_quality_mask=quality,
            input_quality_mask=quality[:, 0], require_reliable_input=True,
            input_quality_source="same_record_target_I")
        summaries[task] = summary
    score = correlation_score_from_summaries(summaries)
    return {"score": score, "tasks": summaries}


@torch.no_grad()
def _validate_fine(model: FineCorrector, items: list[dict],
                   data: HuaweiSevenLeadData, device: torch.device) -> dict:
    predictions = [_fine_prediction(model, item, device) for item in items]
    return _assemble_validation(predictions, [item["record"] for item in items], data)


@torch.no_grad()
def _validate_mid(model: ChestMidCorrector, items: list[dict],
                  data: HuaweiSevenLeadData, device: torch.device) -> dict:
    model.eval()
    predictions = []
    for item in items:
        correction, _ = model(item["features"][None].to(device))
        seven = item["fine7"].copy()
        seven[1:] += correction[0].cpu().numpy()
        predictions.append(seven)
    return _assemble_validation(predictions, [item["record"] for item in items], data)


def _load_fine(path: Path, scale8: np.ndarray, cnn_sha: str,
               device: torch.device) -> FineCorrector:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["cnn_checkpoint_sha256"] != cnn_sha or state["scale8_uV"] != scale8.tolist():
        raise ValueError("fine corrector is not tied to this CNN checkpoint/scales")
    model = FineCorrector(scale8)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def _train_phase(phase: str, records_train: list[Record],
                 records_valid: list[Record], data: HuaweiSevenLeadData,
                 output: Path, cnn_checkpoint: Path,
                 device: torch.device) -> None:
    phase_dir = output / ("fine_0p1" if phase == "fine" else "chest_mid")
    if phase_dir.exists() and any(phase_dir.iterdir()):
        raise FileExistsError(f"phase output already exists: {phase_dir}")
    phase_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    scale8 = data.scale
    cnn_sha = sha256(cnn_checkpoint)
    if phase == "fine":
        train = _fine_items(records_train, scale8)
        valid = _fine_items(records_valid, scale8)
        model = FineCorrector(scale8).to(device)
        validate = lambda: _validate_fine(model, valid, data, device)
        loss_fn = _fine_loss
    elif phase == "mid":
        frozen_path = output / "fine_0p1" / "best.pt"
        fine = _load_fine(frozen_path, scale8, cnn_sha, device)
        train = _mid_items(records_train, fine, scale8, device)
        valid = _mid_items(records_valid, fine, scale8, device)
        if len(valid) != len(records_valid):
            raise ValueError("mid validation must preserve every pair, even QC-masked chest leads")
        model = ChestMidCorrector(scale8).to(device)
        validate = lambda: _validate_mid(model, valid, data, device)
        loss_fn = _mid_loss
    else:
        raise ValueError(phase)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    updates = FINE_UPDATES if phase == "fine" else MID_UPDATES
    config = {"phase": phase, "updates": updates, "batch": BATCH,
              "validate_every": VALIDATE_EVERY, "seed": SEED,
              "learning_rate": 3e-4, "weight_decay": 1e-4,
              "cnn_checkpoint": str(cnn_checkpoint), "cnn_checkpoint_sha256": cnn_sha,
              "train_cache": str(output / "train_cache_step_003000"),
              "validation_cache": str(cnn_checkpoint.parent / "step_003000"),
              "train_records": len(train), "validation_pairs": len(valid),
              "selection": "main reliable-input full-120s missing-11 mean-r",
              "fine_ii_loss_weight": 3 if phase == "fine" else None,
              "fine_raw_rmse_weight_per_uV": RAW_RMSE_WEIGHT_PER_UV if phase == "fine" else None,
              "fine_raw_rmse_centering": "none" if phase == "fine" else None,
              "fine_raw_rmse_lead_normalization": "none" if phase == "fine" else None,
              "original_b7_architecture": "0.1-second all-seven curve" if phase == "fine"
                                          else "4-Hz chest-only mid curve"}
    (phase_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n",
                                           encoding="utf-8")
    best_score, best_step = -float("inf"), None
    for step in range(updates + 1):
        if step % VALIDATE_EVERY == 0:
            model.eval()
            result = validate()
            score = float(result["score"]["main_correlation"])
            (phase_dir / f"validation_{step:07d}.json").write_text(
                json.dumps({"step": step, **result}, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n", encoding="utf-8")
            print(f"[validate-{phase}] step={step} R={score:.8f}", flush=True)
            if score > best_score:
                best_score, best_step = score, step
                torch.save({"model": model.state_dict(), "step": step,
                            "score": score, "config": config,
                            "scale8_uV": scale8.tolist(),
                            "cnn_checkpoint_sha256": cnn_sha}, phase_dir / "best.pt")
        if step == updates:
            break
        model.train()
        ids = rng.choice(len(train), size=BATCH, replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss, details = loss_fn(model, [train[i] for i in ids], device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"[train-{phase}] step={step+1} loss={loss.item():.5f} "
                  f"parts={details}", flush=True)
    (phase_dir / "complete.json").write_text(json.dumps({
        "best_step": best_step, "best_R": best_score, "updates": updates}, indent=2) + "\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fine", "mid", "all"),
                        default="all")
    parser.add_argument("--data-root", default="/root/autodl-tmp/huawei_data")
    parser.add_argument("--cnn-run", default="/root/autodl-tmp/cnn_tra_runs/first_run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = HuaweiSevenLeadData(args.data_root)
    cnn_run = Path(args.cnn_run).resolve()
    checkpoint = cnn_run / "model_step_003000.pt"
    validation_dir = cnn_run / "step_003000"
    device = torch.device(args.device)
    cache = output / "train_cache_step_003000"
    if args.phase in ("prepare", "all") and not (cache / "COMPLETE").exists():
        prepare_train_cache(data, checkpoint, cache, device)
    if args.phase == "prepare":
        return
    records_train = load_train_cache(cache, checkpoint)
    records_valid = load_validation_records(data, validation_dir)
    if args.phase in ("fine", "all") and not (output / "fine_0p1" / "complete.json").exists():
        _train_phase("fine", records_train, records_valid, data, output, checkpoint, device)
    if args.phase in ("mid", "all") and not (output / "chest_mid" / "complete.json").exists():
        _train_phase("mid", records_train, records_valid, data, output, checkpoint, device)


if __name__ == "__main__":
    main()
