"""Protocol-compliant B1 training and evaluation primitives.

All public data, preprocessing, loss, split and V0 definitions come from main.
This module only supplies the B1 model loop and never mutates source arrays.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from b1.data import local_config
from b1.model import LightweightUNet
from ecg12gen.body_scale import BodyScaleVariantDataset
from ecg12gen.contracts import canonical_lead_mask
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.dataset import UnifiedECGDataset
from ecg12gen.evaluate import (
    evaluate_centered_diagnostic,
    evaluate_predictions,
    evaluate_task2_diagnostics,
    write_report,
    write_task2_diagnostics,
)
from ecg12gen.losses import (
    masked_huber_loss,
    masked_pcc_loss,
    observed_consistency_loss,
    pair_invariant_stat_loss,
    physiology_constraint_loss,
    spectral_stat_loss,
)
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.training import load_training_protocol, seed_everything

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "b1" / "official_v1"


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    task: str
    route: str
    data_variant: str
    loss_variant: str | None

    @property
    def input_channels(self) -> int:
        return 1 if self.task == "task1" else 6


@dataclass
class RawBundle:
    x: np.ndarray
    y: np.ndarray
    metadata: list[dict[str, str]]
    quality: np.ndarray | None = None


@dataclass
class ModelBundle:
    x: np.ndarray
    y: np.ndarray
    observed: np.ndarray
    raw: RawBundle
    quality: np.ndarray | None = None


def read_matrix(path: Path = ROOT / "b1" / "experiment_matrix.yaml") -> list[ExperimentSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = [ExperimentSpec(**item) for item in raw["experiments"]]
    if len(specs) != 14 or len({spec.id for spec in specs}) != 14:
        raise ValueError("B1 official matrix must contain 14 unique experiments")
    for spec in specs:
        if spec.task == "task1" and spec.route in {"C1", "C2"}:
            if spec.loss_variant is not None:
                raise ValueError("C1/C2 use the fixed pseudo loss and cannot be crossed with L0/L1")
        elif spec.loss_variant not in {"L0", "L1"}:
            raise ValueError(f"{spec.id} must select L0 or L1")
        if spec.task == "task2" and spec.route not in {"A", "B"}:
            raise ValueError("Task2 has no route C")
    return specs


def _rows_from_dataset(dataset: Iterable[Any]) -> RawBundle:
    samples = list(dataset)
    if not samples:
        raise ValueError("Dataset is empty")
    x = np.stack([sample.X_ecg for sample in samples]).astype(np.float32)
    y = np.stack([sample.Y_12lead for sample in samples]).astype(np.float32)
    metadata = [{
        "subject_id": str(sample.meta["subject_id"]),
        "window_id": str(sample.meta["window_id"]),
        "input_type": str(sample.meta.get("device_type", "d12")),
        "split": str(sample.split),
    } for sample in samples]
    return RawBundle(x=x, y=y, metadata=metadata)


def load_weak(task: str, split: str, data_variant: str) -> RawBundle:
    config = local_config()
    if task == "task1":
        return _rows_from_dataset(UnifiedECGDataset(config, "task1", split))
    canonical = list(UnifiedECGDataset(config, "task2", split))
    machine = [sample for sample in canonical if sample.meta["device_type"] == "ecg_machine_d6"]
    if data_variant == "A":
        body = [sample for sample in canonical if sample.meta["device_type"] == "body_scale_d6"]
    elif data_variant == "B":
        body = list(BodyScaleVariantDataset(config, split, "B_detrend_0p2Hz_then_window"))
    else:
        raise ValueError("Task2 data variant must be A or B")
    bundle = _rows_from_dataset([*machine, *body])
    if {row["input_type"] for row in bundle.metadata} != {"ecg_machine_d6", "body_scale_d6"}:
        raise ValueError("Every task2 data variant must include both machine and body-scale d6")
    return bundle


def _nested_data_dir(name: str) -> Path:
    outer = ROOT.parent / name
    inner = outer / name
    directory = inner if inner.is_dir() else outer
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing derived data directory: {outer}")
    return directory


def load_pseudo(route: str) -> RawBundle:
    if route not in {"C1", "C2"}:
        raise ValueError("Pseudo data route must be C1 or C2")
    name = "task1_rpeak_pseudo_output" if route == "C1" else "task1_rpeak_pseudo_output_c2"
    directory = _nested_data_dir(name)
    x = np.asarray(np.load(directory / "task1_rpeak_train_input.npy", mmap_mode="r"), dtype=np.float32)
    y = np.asarray(np.load(directory / "task1_rpeak_train_target.npy", mmap_mode="r"), dtype=np.float32)
    with (directory / "task1_rpeak_train_window_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = 18 if route == "C1" else 33
    if len(rows) != expected or len(x) != expected or len(y) != expected:
        raise ValueError(f"{route} must contain exactly {expected} accepted windows")
    if any(row.get("split") != "train" or row.get("accepted", "").lower() != "true" for row in rows):
        raise ValueError("Pseudo-pair adaptation may only read accepted train windows")
    quality = np.asarray([float(row["alignment_quality_score"]) for row in rows], dtype=np.float32)
    if not np.isfinite(quality).all() or np.any(quality <= 0):
        raise ValueError("Every pseudo window needs a positive alignment_quality_score")
    metadata = [{"subject_id": row["subject_id"], "window_id": row["source_window_id"],
                 "input_type": "watch_ecg", "split": "train"} for row in rows]
    return RawBundle(x=x, y=y, metadata=metadata, quality=quality)


def load_strict_raw(task: str) -> RawBundle:
    mode = "d12_i_pretrain" if task == "task1" else "d12_six_pretrain"
    return _rows_from_dataset(StrictD12PretrainDataset(local_config(), mode))


def fit_preprocessor(task: str, train: RawBundle, strict: RawBundle) -> ECGPreprocessor:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    signals: dict[str, np.ndarray] = {"d12": strict.y}
    if task == "task1":
        signals["watch_ecg"] = train.x
    else:
        for source in ("ecg_machine_d6", "body_scale_d6"):
            indices = [i for i, row in enumerate(train.metadata) if row["input_type"] == source]
            if not indices:
                raise ValueError(f"Task2 training data has no {source} windows")
            signals[source] = train.x[np.asarray(indices)]
    return ECGPreprocessor.fit(config, signals)


def transform_bundle(raw: RawBundle, pre: ECGPreprocessor, task: str) -> ModelBundle:
    n, channels, samples = raw.x.shape
    x = np.empty_like(raw.x, dtype=np.float32)
    if task == "task1":
        x[:] = pre.transform_batch(raw.x, "watch_ecg")[0]
    else:
        for source in ("ecg_machine_d6", "body_scale_d6"):
            indices = np.asarray([i for i, row in enumerate(raw.metadata) if row["input_type"] == source])
            if indices.size:
                x[indices] = pre.transform_batch(raw.x[indices], source)[0]
    y = pre.transform_batch(raw.y, "d12")[0]
    observed = np.zeros((n, 12, samples), dtype=np.float32)
    d12_scale = pre.scale_uV_by_source["d12"]
    for i, row in enumerate(raw.metadata):
        source_scale = pre.scale_uV_by_source[row["input_type"]]
        observed[i, :channels] = x[i] * source_scale[:, None] / d12_scale[:channels, None]
    return ModelBundle(x=x, y=y, observed=observed, raw=raw, quality=raw.quality)


def transform_strict(raw: RawBundle, pre: ECGPreprocessor, task: str) -> ModelBundle:
    y = pre.transform_batch(raw.y, "d12")[0]
    channels = 1 if task == "task1" else 6
    x = y[:, :channels].copy()
    observed = np.zeros_like(y)
    observed[:, :channels] = x
    return ModelBundle(x=x, y=y, observed=observed, raw=raw)


def _loader(bundle: ModelBundle, batch_size: int, shuffle_seed: int, include_quality: bool = False) -> DataLoader:
    tensors = [torch.from_numpy(bundle.x), torch.from_numpy(bundle.y), torch.from_numpy(bundle.observed)]
    if include_quality:
        if bundle.quality is None:
            raise ValueError("Quality-weighted loader requires quality scores")
        tensors.append(torch.from_numpy(bundle.quality))
    generator = torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True, num_workers=0,
                      pin_memory=False, generator=generator)


def _lead_mask(batch: int, channels: int, device: torch.device, missing: bool) -> torch.Tensor:
    mask = canonical_lead_mask(channels)
    if missing:
        mask = ~mask
    return torch.from_numpy(mask).to(device).unsqueeze(0).expand(batch, -1)


def strict_loss(pred: torch.Tensor, target: torch.Tensor, observed: torch.Tensor, channels: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    missing = _lead_mask(len(pred), channels, pred.device, True)
    visible = _lead_mask(len(pred), channels, pred.device, False)
    huber = masked_huber_loss(pred, target, missing)
    pcc = masked_pcc_loss(pred, target, missing)
    phys = physiology_constraint_loss(pred)
    obs = observed_consistency_loss(pred, observed, visible)
    total = huber + 0.10 * pcc + 0.05 * phys + 0.02 * obs
    return total, {"strict_huber": huber, "strict_pcc": pcc, "strict_physiology": phys, "strict_observed": obs}


def weak_loss(pred: torch.Tensor, paired_target: torch.Tensor, observed: torch.Tensor,
              reference: torch.Tensor, channels: int, variant: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visible = _lead_mask(len(pred), channels, pred.device, False)
    obs = observed_consistency_loss(pred, observed, visible)
    spectral = spectral_stat_loss(pred, reference)
    phys = physiology_constraint_loss(pred)
    pair = pair_invariant_stat_loss(pred, paired_target) if variant == "L1" else pred.new_zeros(())
    total = 0.10 * obs + 0.20 * spectral + 0.05 * phys + (0.20 * pair if variant == "L1" else 0.0)
    return total, {"weak_observed": obs, "weak_spectral": spectral, "weak_physiology": phys, "weak_pair_invariant": pair}


def pseudo_loss(pred: torch.Tensor, target: torch.Tensor, observed: torch.Tensor,
                quality: torch.Tensor, channels: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visible = _lead_mask(1, channels, pred.device, False)
    missing = _lead_mask(1, channels, pred.device, True)
    values: list[torch.Tensor] = []
    sums = {"pseudo_huber": pred.new_zeros(()), "pseudo_pcc": pred.new_zeros(()),
            "pseudo_physiology": pred.new_zeros(()), "pseudo_observed": pred.new_zeros(())}
    for i in range(len(pred)):
        huber = masked_huber_loss(pred[i:i + 1], target[i:i + 1], missing)
        pcc = masked_pcc_loss(pred[i:i + 1], target[i:i + 1], missing)
        phys = physiology_constraint_loss(pred[i:i + 1])
        obs = observed_consistency_loss(pred[i:i + 1], observed[i:i + 1], visible)
        values.append(huber + 0.10 * pcc + 0.05 * phys + 0.10 * obs)
        for key, value in (("pseudo_huber", huber), ("pseudo_pcc", pcc),
                           ("pseudo_physiology", phys), ("pseudo_observed", obs)):
            sums[key] = sums[key] + value
    weights = quality.to(device=pred.device, dtype=pred.dtype)
    total = (torch.stack(values) * weights).sum() / weights.sum().clamp_min(1e-8)
    components = {key: value / len(pred) for key, value in sums.items()}
    components["pseudo_quality_mean"] = weights.mean()
    return total, components


def _to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device) for item in batch)


def _repeat(loader: DataLoader):
    """Repeat a loader without caching all of its batches in memory."""
    while True:
        yield from loader


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in keys}


def _row(total: torch.Tensor, components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {"loss": float(total.detach()), **{key: float(value.detach()) for key, value in components.items()}}


def train_strict_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, bundle: ModelBundle,
                       task: str, batch_size: int, epoch: int, seed: int, device: torch.device,
                       max_batches: int | None = None) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for index, batch in enumerate(_loader(bundle, batch_size, seed + epoch * 1009)):
        x, y, observed = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = strict_loss(model(x), y, observed, 1 if task == "task1" else 6)
        loss.backward(); optimizer.step()
        rows.append(_row(loss, components))
        if max_batches is not None and index + 1 >= max_batches:
            break
    return _mean(rows)


def train_adaptation_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, spec: ExperimentSpec,
                           branch: ModelBundle, strict: ModelBundle, reference_bank: torch.Tensor,
                           batch_size: int, epoch: int, seed: int, device: torch.device,
                           max_batches: int | None = None) -> dict[str, float]:
    model.train()
    branch_loader = _loader(branch, batch_size, seed + epoch * 1013, spec.route in {"C1", "C2"})
    strict_loader = _loader(strict, batch_size, seed + epoch * 1019)
    strict_iter = None if spec.route == "A" else _repeat(strict_loader)
    branch_iter = _repeat(branch_loader)
    steps = len(branch_loader) if spec.route == "A" else max(len(branch_loader), len(strict_loader))
    if max_batches is not None:
        steps = min(steps, max_batches)
    rows: list[dict[str, float]] = []
    generator = torch.Generator(device=reference_bank.device).manual_seed(seed + epoch * 1021)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        components: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=device)
        if strict_iter is not None:
            sx, sy, sobs = _to_device(next(strict_iter), device)
            strict_total, strict_components = strict_loss(model(sx), sy, sobs, spec.input_channels)
            total = total + strict_total
            components.update(strict_components)
        batch = _to_device(next(branch_iter), device)
        x, y, observed = batch[:3]
        pred = model(x)
        if spec.route in {"A", "B"}:
            indices = torch.randint(0, len(reference_bank), (len(x),), generator=generator, device=reference_bank.device)
            branch_total, branch_components = weak_loss(pred, y, observed, reference_bank[indices],
                                                         spec.input_channels, str(spec.loss_variant))
        else:
            branch_total, branch_components = pseudo_loss(pred, y, observed, batch[3], spec.input_channels)
        weight = 1.0 if spec.route == "A" else 0.20
        total = total + weight * branch_total
        components.update(branch_components)
        components["adaptation_branch_weight"] = total.new_tensor(weight)
        total.backward(); optimizer.step()
        rows.append(_row(total, components))
    return _mean(rows)


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); output: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (batch,) in loader:
            output.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def validation_metrics(model: torch.nn.Module, bundle: ModelBundle, pre: ECGPreprocessor,
                       task: str, batch_size: int, device: torch.device) -> dict[str, Any]:
    pred_model = predict(model, bundle.x, batch_size, device)
    pred_raw = pred_model * pre.scale_uV_by_source["d12"][None, :, None]
    e1, _ = evaluate_predictions(pred_raw, bundle.raw.y, task)
    copied = pred_raw.copy(); copied[:, :bundle.raw.x.shape[1]] = bundle.raw.x
    e2, _ = evaluate_predictions(copied, bundle.raw.y, task)
    centered, _ = evaluate_centered_diagnostic(pred_raw, bundle.raw.y, task)
    return {"E1": e1, "E2": e2, "centered": centered, "prediction_E1": pred_raw, "prediction_E2": copied}


def write_final_evaluation(output: Path, metrics: dict[str, Any], bundle: ModelBundle, task: str) -> None:
    for key, folder in (("E1", "E1_raw"), ("E2", "E2_copy_at_eval")):
        overall, details = evaluate_predictions(metrics[f"prediction_{key}"], bundle.raw.y, task)
        paths = write_report(output / folder, overall, details, title=f"B1 {key} validation report")
        if task == "task2":
            subjects, devices = evaluate_task2_diagnostics(metrics[f"prediction_{key}"], bundle.raw.y, bundle.raw.metadata)
            write_task2_diagnostics(output / folder, subjects, devices, paths[-1])
    centered_overall, centered_details = evaluate_centered_diagnostic(metrics["prediction_E1"], bundle.raw.y, task)
    paths = write_report(output / "centered_diagnostic", centered_overall, centered_details,
                         title="Centered morphology diagnostic (not official)")
    if task == "task2":
        cp = metrics["prediction_E1"] - np.median(metrics["prediction_E1"], axis=2, keepdims=True)
        ct = bundle.raw.y - np.median(bundle.raw.y, axis=2, keepdims=True)
        subjects, devices = evaluate_task2_diagnostics(cp, ct, bundle.raw.metadata)
        write_task2_diagnostics(output / "centered_diagnostic", subjects, devices, paths[-1])
    np.save(output / "prediction_E1_raw_uV.npy", metrics["prediction_E1"])
    np.save(output / "prediction_E2_copy_at_eval_uV.npy", metrics["prediction_E2"])
    _write_json(output / "metrics.json", {key: value for key, value in metrics.items() if not key.startswith("prediction_")})


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _fingerprint(spec: ExperimentSpec) -> str:
    digest = hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode())
    for relative in ("configs/common.yaml", "configs/preprocessing.yaml", "configs/training_protocol_v1.yaml",
                     "configs/losses.yaml", "b1/experiment_matrix.yaml", "b1/model.py", "b1/official_v1.py"):
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def environment_record(device: torch.device) -> dict[str, Any]:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "device": str(device), "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(), "git_commit": _git_commit()}


def run_shared_pretrain(task: str, strict: ModelBundle, output_root: Path, epochs: int,
                        batch_size: int, seed: int, learning_rate: float, weight_decay: float,
                        device: torch.device, max_batches: int | None = None) -> Path:
    directory = output_root / ("smoke_shared_pretrain" if epochs == 1 else "shared_pretrain") / task
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "checkpoint_final.pt"; latest = directory / "checkpoint_latest.pt"
    model = LightweightUNet(1 if task == "task1" else 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []; start_epoch = 0
    if final.exists() and int(torch.load(final, map_location="cpu", weights_only=False)["epoch"]) == epochs:
        return final
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]); history = list(state.get("history", []))
    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.perf_counter()
        values = train_strict_epoch(model, optimizer, strict, task, batch_size, epoch, seed, device, max_batches)
        row = {"epoch": epoch, **values, "epoch_seconds": time.perf_counter() - started}; history.append(row)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history}, latest)
        _write_csv(directory / "training_log.csv", history)
        print(json.dumps({"stage": f"shared_pretrain_{task}", **row}), flush=True)
    os.replace(latest, final)
    _write_json(directory / "run_summary.json", {"status": "completed", "task": task, "epochs": epochs,
                                                   "environment": environment_record(device), "history": history})
    return final


def run_experiment(spec: ExperimentSpec, output_root: Path = DEFAULT_OUTPUT_ROOT, smoke: bool = False,
                   max_batches: int | None = None, device_name: str = "auto") -> Path:
    fixed = load_training_protocol(ROOT / "configs" / "training_protocol_v1.yaml")["fair_baseline_comparison"]
    epochs = 1 if smoke else int(fixed["epochs_per_stage"])
    batch_size, seed = int(fixed["batch_size"]), int(fixed["seed"])
    learning_rate, weight_decay = float(fixed["learning_rate"]), float(fixed["weight_decay"])
    if not smoke and (epochs, batch_size, seed, learning_rate, weight_decay) != (100, 16, 42, 0.001, 0.0001):
        raise ValueError("Official B1 runs must use frozen fair-baseline settings")
    seed_everything(seed, bool(fixed["deterministic"]))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run_root = output_root / ("smoke" if smoke else "experiments") / spec.id
    run_root.mkdir(parents=True, exist_ok=True); summary_path = run_root / "run_summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "completed":
        print(f"SKIP completed {spec.id}"); return run_root

    strict_raw = load_strict_raw(spec.task)
    train_raw = load_weak(spec.task, "train", spec.data_variant)
    validation_raw = load_weak(spec.task, "validation", spec.data_variant)
    pre = fit_preprocessor(spec.task, train_raw, strict_raw)
    strict = transform_strict(strict_raw, pre, spec.task)
    train = transform_bundle(train_raw, pre, spec.task)
    validation = transform_bundle(validation_raw, pre, spec.task)
    branch = transform_bundle(load_pseudo(spec.route), pre, spec.task) if spec.route in {"C1", "C2"} else train
    _write_json(run_root / "run_info.json", {
        "protocol": "b1_official_v1", "spec": asdict(spec), "fingerprint": _fingerprint(spec),
        "shared_config_commit": _git_commit(), "environment": environment_record(device),
        "training": {"epochs_per_stage": epochs, "batch_size": batch_size, "seed": seed, "optimizer": "AdamW",
                     "learning_rate": learning_rate, "weight_decay": weight_decay, "scheduler": None,
                     "gradient_clip": None, "early_stopping": False},
        "data": {"train_windows": len(train.raw.x), "validation_windows": len(validation.raw.x),
                 "strict_windows": len(strict.raw.x), "adaptation_windows": len(branch.raw.x),
                 "task2_device_counts_train": {source: sum(row["input_type"] == source for row in train.raw.metadata)
                                               for source in {row["input_type"] for row in train.raw.metadata}}},
        "preprocessing_scale_uV": {key: value.tolist() for key, value in pre.scale_uV_by_source.items()},
        "raw_prediction_baseline_policy": "zero because baseline_head is disabled"})

    model = LightweightUNet(spec.input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if spec.route in {"B", "C1", "C2"}:
        checkpoint = run_shared_pretrain(spec.task, strict, output_root, epochs, batch_size, seed,
                                         learning_rate, weight_decay, device, max_batches)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])

    latest = run_root / "checkpoint_latest.pt"; best = run_root / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []; start_epoch = 0; best_metric = -float("inf"); best_epoch = 0
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("fingerprint") != _fingerprint(spec):
            raise ValueError(f"Refusing to resume {spec.id}: configuration fingerprint changed")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        history, start_epoch = list(state["history"]), int(state["epoch"])
        best_metric, best_epoch = float(state["best_metric"]), int(state["best_epoch"])
    reference_bank = torch.from_numpy(strict.y).to(device)
    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.perf_counter()
        train_values = train_adaptation_epoch(model, optimizer, spec, branch, strict, reference_bank,
                                              batch_size, epoch, seed, device, max_batches)
        metrics = validation_metrics(model, validation, pre, spec.task, batch_size, device)
        metric_name = "task1_r1" if spec.task == "task1" else "task2_r2"
        metric = float(metrics["E1"][metric_name])
        row = {"epoch": epoch, **train_values, "validation_official_v0": metric,
               "validation_copy_at_eval": float(metrics["E2"][metric_name]),
               "validation_centered_diagnostic": float(metrics["centered"][metric_name]),
               "epoch_seconds": time.perf_counter() - started}; history.append(row)
        if metric > best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "metric": metric, "fingerprint": _fingerprint(spec)}, best)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                    "best_metric": best_metric, "best_epoch": best_epoch, "fingerprint": _fingerprint(spec)}, latest)
        _write_csv(run_root / "training_log.csv", history)
        print(json.dumps({"experiment": spec.id, **row}), flush=True)
    selected = torch.load(best, map_location=device, weights_only=False); model.load_state_dict(selected["model"])
    final_metrics = validation_metrics(model, validation, pre, spec.task, batch_size, device)
    write_final_evaluation(run_root / "evaluation", final_metrics, validation, spec.task)
    _write_json(summary_path, {"status": "completed", "experiment": spec.id, "spec": asdict(spec),
                               "best_epoch": best_epoch, "best_validation_official_v0": best_metric,
                               "best_metrics": {key: value for key, value in final_metrics.items() if not key.startswith("prediction_")},
                               "history": history, "environment": environment_record(device)})
    return run_root
