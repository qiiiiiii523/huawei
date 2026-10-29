"""Train the B3-style I -> (II,V1..V6) model for 5000 optimizer steps.

At steps 3000 and 5000, cache all seven raw-uV validation predictions and
aligned 120-second identity/target data. No correction model is fitted here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.cnn_tra_data import (HuaweiSevenLeadData, StrictTrainDataset,
                                   LEAD_INDICES, TASKS)
from ecg12gen.cnn_tra_model import CNNTransformerSevenLead, OUTPUT_LEADS


WEIGHTS = torch.tensor([3., 1., 1., 1., 1., 1., 1.], dtype=torch.float32)
VALIDATE_AT = (3000, 5000)


def loss_parts(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (7, 5000):
        raise ValueError("expected matching [batch,7,5000] predictions/targets")
    weights = WEIGHTS.to(device=pred.device, dtype=pred.dtype)[None] * mask
    if torch.any(weights.sum(dim=1) <= 0):
        raise ValueError("training sample has no reliable target lead")
    huber = F.huber_loss(pred, target, delta=1.0, reduction="none").mean(dim=-1)
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    numerator = (p * t).sum(dim=-1)
    denominator = torch.sqrt(p.square().sum(dim=-1) * t.square().sum(dim=-1)).clamp_min(1e-8)
    correlation_loss = 1.0 - numerator / denominator
    denom = weights.sum()
    huber_loss = (huber * weights).sum() / denom
    pcc_loss = (correlation_loss * weights).sum() / denom
    return huber_loss + 0.1 * pcc_loss, huber_loss, pcc_loss


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(np.float64) - float(pred.mean(dtype=np.float64))
    t = target.astype(np.float64) - float(target.mean(dtype=np.float64))
    denominator = math.sqrt(float(np.dot(p, p)) * float(np.dot(t, t)))
    return float(np.dot(p, t) / denominator) if denominator > 1e-10 else float("nan")


def _record_metrics(prediction: np.ndarray, target: np.ndarray,
                    rows: list[dict[str, str]], quality: np.ndarray,
                    task: str) -> dict:
    # Each pair is a complete 120-s record. Multiple pairs of the same physical
    # target/device are averaged before scoring, as in the main record protocol.
    pairs: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pairs.setdefault(row["pair_id"], []).append(index)
    physical: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    excluded_pairs = 0
    for indices in pairs.values():
        indices.sort(key=lambda i: int(rows[i]["window_index"]))
        if len(indices) != 12 or [int(rows[i]["window_index"]) for i in indices] != list(range(12)):
            raise ValueError("validation pair is not a complete 120-second record")
        if not np.array_equal(quality[indices], np.repeat(quality[indices[:1]], 12, axis=0)):
            raise ValueError("target QC changes inside a record")
        if not quality[indices[0], 0]:
            excluded_pairs += 1
            continue
        row = rows[indices[0]]
        key = (row["input_type"], row["target_record_id"])
        physical.setdefault(key, []).append((
            np.concatenate([prediction[i] for i in indices], axis=-1),
            np.concatenate([target[i] for i in indices], axis=-1),
            quality[indices[0], list(LEAD_INDICES)].copy()))

    by_device: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for (device, _record), copies in physical.items():
        pred = np.mean(np.stack([copy[0] for copy in copies]), axis=0)
        truth = copies[0][1]
        mask = copies[0][2]
        if any(not np.allclose(copy[1], truth, rtol=0, atol=1e-4) or
               not np.array_equal(copy[2], mask) for copy in copies[1:]):
            raise ValueError("duplicate physical target differs across pairs")
        for lead, name in enumerate(OUTPUT_LEADS):
            if not mask[lead]:
                continue
            r = _pearson(pred[lead], truth[lead])
            if not np.isfinite(r):
                continue
            rmse = float(np.sqrt(np.mean(np.square(
                pred[lead].astype(np.float64) - truth[lead].astype(np.float64)))))
            by_device.setdefault(device, {}).setdefault(name, []).append((r, rmse))
    if not by_device:
        raise ValueError(f"{task}: no eligible validation records")
    device_metrics = {}
    for device, lead_values in by_device.items():
        if set(lead_values) != set(OUTPUT_LEADS):
            raise ValueError(f"{task}/{device} has no valid record for some output lead")
        leads = {name: {"r": float(np.mean([v[0] for v in values])),
                        "rmse_uV": float(np.mean([v[1] for v in values])),
                        "physical_records": len(values)}
                 for name, values in lead_values.items()}
        device_metrics[device] = {
            "r_7lead": float(np.mean([leads[name]["r"] for name in OUTPUT_LEADS])),
            "rmse_7lead_uV": float(np.mean([leads[name]["rmse_uV"] for name in OUTPUT_LEADS])),
            "leads": leads,
        }
    if task == "task2" and set(device_metrics) != {"ecg_machine_d6", "body_scale_d6"}:
        raise ValueError("Task2 validation must contain both machine and body-scale device types")
    return {
        "task": task, "score_scope": "provisional_7lead_120s_not_final_11lead",
        "r_7lead": float(np.mean([d["r_7lead"] for d in device_metrics.values()])),
        "rmse_7lead_uV": float(np.mean([d["rmse_7lead_uV"] for d in device_metrics.values()])),
        "devices": device_metrics, "eligible_physical_device_records": len(physical),
        "excluded_input_qc_pairs": excluded_pairs,
    }


@torch.no_grad()
def validate_and_cache(model: CNNTransformerSevenLead, data: HuaweiSevenLeadData,
                       device: torch.device, output_dir: Path, step: int,
                       batch_size: int, checkpoint_name: str) -> dict:
    model.eval()
    step_dir = output_dir / f"step_{step:06d}"
    if step_dir.exists():
        raise FileExistsError(f"validation cache already exists: {step_dir}")
    step_dir.mkdir(parents=True)
    task_metrics = {}
    for task in TASKS:
        task_dir = step_dir / task
        task_dir.mkdir()
        source = data.validation_arrays[task]
        rows = data.validation_rows[task]
        quality = data.validation_masks[task]
        count = len(source)
        pred = np.lib.format.open_memmap(task_dir / "prediction_7_uV.npy", mode="w+",
                                         dtype="float32", shape=(count, 7, 5000))
        target = np.lib.format.open_memmap(task_dir / "target_7_uV.npy", mode="w+",
                                           dtype="float32", shape=(count, 7, 5000))
        anchor = np.lib.format.open_memmap(task_dir / "input_i_uV.npy", mode="w+",
                                           dtype="float32", shape=(count, 1, 5000))
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            raw = np.asarray(source[start:stop], dtype=np.float32)
            x = torch.from_numpy(np.array(raw[:, 0:1] / data.scale[0], copy=True)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(x)
            pred[start:stop] = output.float().cpu().numpy() * data.scale[None, 1:, None]
            target[start:stop] = raw[:, list(LEAD_INDICES)]
            anchor[start:stop] = raw[:, 0:1]
        pred.flush(); target.flush(); anchor.flush()
        np.save(task_dir / "target_quality_mask_7.npy", quality[:, list(LEAD_INDICES)])
        np.save(task_dir / "input_i_quality_mask.npy", quality[:, 0])
        with (task_dir / "window_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        task_metrics[task] = _record_metrics(pred, target, rows, quality, task)
        (task_dir / "metrics.json").write_text(json.dumps(task_metrics[task], indent=2,
                                                   allow_nan=False) + "\n", encoding="utf-8")
        del pred, target, anchor
    summary = {
        "step": step, "selection_metric": "mean_task_7lead_120s_r",
        "r_7lead": float(np.mean([task_metrics[t]["r_7lead"] for t in TASKS])),
        "rmse_7lead_uV": float(np.mean([task_metrics[t]["rmse_7lead_uV"] for t in TASKS])),
        "tasks": task_metrics,
    }
    (step_dir / "metrics.json").write_text(json.dumps(summary, indent=2,
                                               allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"cache_schema": "cnn_tra_7lead_validation_v1", "step": step,
                "checkpoint": checkpoint_name, "output_leads": OUTPUT_LEADS,
                "samples_per_window": 5000, "windows_per_record": 12,
                "sampling_rate_hz": 500, "unit": "uV", "data_root": str(data.root),
                "notes": "Seven direct predictions only; no formula or correction output."}
    (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (step_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary


def _checkpoint(path: Path, model: CNNTransformerSevenLead, optimizer: torch.optim.Optimizer,
                step: int, data: HuaweiSevenLeadData, args: argparse.Namespace) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "scale_uV": data.scale.tolist(), "output_leads": OUTPUT_LEADS,
                "configuration": vars(args), "architecture": "B3_CNN_Transformer_FPN_7lead_v1"}, path)


def run(args: argparse.Namespace) -> None:
    if args.max_steps != 5000 and not args.smoke_test:
        raise ValueError("production schedule is fixed to 5000 optimizer steps")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data = HuaweiSevenLeadData(args.data_root)
    model = CNNTransformerSevenLead(args.transformer_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(StrictTrainDataset(data), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=False)
    batches = iter(loader)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(json.dumps({"train_windows": len(data.train_items),
                      "scale_uV": data.scale.tolist(),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "device": str(device), "validation_steps": VALIDATE_AT}), flush=True)
    max_steps = args.max_steps
    if args.smoke_test:
        max_steps = 1
    else:
        output_dir = Path(args.output_dir).resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n",
                                                     encoding="utf-8")
        (output_dir / "scale_uV.json").write_text(json.dumps({
            "leads": ["I", *OUTPUT_LEADS], "scale_uV": data.scale.tolist()}, indent=2) + "\n",
            encoding="utf-8")
    best = None
    for step in range(1, max_steps + 1):
        try:
            anchor, target, mask = next(batches)
        except StopIteration:
            batches = iter(loader)
            anchor, target, mask = next(batches)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        anchor = anchor.to(device); target = target.to(device); mask = mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            prediction = model(anchor)
            total, huber, pcc = loss_parts(prediction.float(), target, mask)
        scaler.scale(total).backward()
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % 100 == 0:
            print(f"step={step} loss={total.item():.6f} huber={huber.item():.6f} "
                  f"pcc={pcc.item():.6f}", flush=True)
        if not args.smoke_test and step in VALIDATE_AT:
            ckpt_name = f"model_step_{step:06d}.pt"
            _checkpoint(output_dir / ckpt_name, model, optimizer, step, data, args)
            metrics = validate_and_cache(model, data, device, output_dir, step,
                                         args.validation_batch_size, ckpt_name)
            print(f"VALIDATION step={step} r7={metrics['r_7lead']:.6f} "
                  f"rmse7_uV={metrics['rmse_7lead_uV']:.3f}", flush=True)
            candidate = (metrics["r_7lead"], -metrics["rmse_7lead_uV"])
            if best is None or candidate > best[0]:
                best = (candidate, step)
                shutil.copy2(output_dir / ckpt_name, output_dir / "best_7lead.pt")
                (output_dir / "best_7lead.json").write_text(json.dumps({
                    "step": step, "r_7lead": metrics["r_7lead"],
                    "rmse_7lead_uV": metrics["rmse_7lead_uV"],
                    "scope": "provisional_7lead_not_final_11lead"}, indent=2) + "\n",
                    encoding="utf-8")
    if args.smoke_test:
        print("SMOKE_TEST_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/root/autodl-tmp/huawei_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true",
                        help="one real train update, without writing checkpoints/caches")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
