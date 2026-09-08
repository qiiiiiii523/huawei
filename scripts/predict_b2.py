"""B2 inference: explicit machine-I anchor only; target is never an input."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg12gen.b2_data import b2_collate, build_joint_dataset
from ecg12gen.b2_model import B2JointAnchorPatchTransformer, B2ModelConfig
from ecg12gen.b2_train import P0_STAGE, _forward
from ecg12gen.contracts import ContractError, WINDOW_SAMPLES, canonical_lead_mask
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def _preprocessor(run_dir: Path) -> ECGPreprocessor:
    cfg = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    scales = json.loads((run_dir / "preprocessing_scales.json").read_text(encoding="utf-8"))
    return ECGPreprocessor(cfg, {k: np.asarray(v, dtype=np.float32) for k, v in scales.items()})


def _model(path: Path, device: str) -> tuple[B2JointAnchorPatchTransformer, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != B2JointAnchorPatchTransformer.checkpoint_schema: raise ContractError("incompatible B2 checkpoint schema")
    model = B2JointAnchorPatchTransformer(B2ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval(); return model, checkpoint


def _d6_model(values: np.ndarray, source: str, pre: ECGPreprocessor, indices: tuple[int, ...] | None) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3 or values.shape[-1] != WINDOW_SAMPLES: raise ContractError("d6 context must be [N,C,5000]")
    chosen = indices or tuple(range(6))
    if values.shape[1] != len(chosen): raise ContractError("d6 channels disagree with --context-channel-indices")
    raw = np.zeros((len(values), 6, WINDOW_SAMPLES), np.float32); raw[:, list(chosen)] = values
    model, _, _ = pre.transform_batch(raw, source)
    mask = np.zeros((len(values), 6), bool); mask[:, list(chosen)] = True
    return model, mask


@torch.no_grad()
def _explicit(model: B2JointAnchorPatchTransformer, checkpoint: dict[str, object], pre: ECGPreprocessor, task_id: str,
              anchor: np.ndarray, watch: np.ndarray | None, machine: np.ndarray | None, body: np.ndarray | None,
              indices: tuple[int, ...] | None, device: str) -> tuple[np.ndarray, np.ndarray]:
    if anchor.ndim != 3 or anchor.shape[1:] != (1, WINDOW_SAMPLES): raise ContractError("--anchor-npy must be [N,1,5000]")
    n = len(anchor)
    if task_id == "task1" and (machine is not None or body is not None): raise ContractError("task1 accepts watch context only")
    if task_id == "task2" and watch is not None: raise ContractError("task2 accepts machine/body d6 context only")
    if task_id == "task1" and watch is not None and watch.shape != (n, 1, WINDOW_SAMPLES): raise ContractError("watch context must be [N,1,5000]")
    if task_id == "task2" and ((machine is not None and len(machine) != n) or (body is not None and len(body) != n)): raise ContractError("all explicit inputs must have the same N")
    if checkpoint["stage"] != P0_STAGE and ((task_id == "task1" and watch is None) or (task_id == "task2" and machine is None and body is None)):
        raise ContractError("P1 inference requires explicit context as well as explicit machine-I anchor")
    anchor_model, _, _ = pre.transform_batch(anchor, "ecg_machine_i")
    zeros_watch, zeros_d6 = np.zeros((n, 1, 5000), np.float32), np.zeros((n, 6, 5000), np.float32)
    watch_model = zeros_watch if watch is None else pre.transform_batch(watch, "watch_ecg")[0]
    machine_model, machine_mask = (zeros_d6, np.zeros((n, 6), bool)) if machine is None else _d6_model(machine, "ecg_machine_d6", pre, indices)
    body_model, body_mask = (zeros_d6, np.zeros((n, 6), bool)) if body is None else _d6_model(body, "body_scale_d6", pre, indices)
    batch = {"anchor_model": torch.from_numpy(anchor_model).to(device), "anchor_lead_mask": torch.from_numpy(np.broadcast_to(canonical_lead_mask(1), (n,12)).copy()).to(device),
             "watch_context_model": torch.from_numpy(watch_model).to(device), "watch_available": torch.full((n,), watch is not None, device=device),
             "machine_d6_model": torch.from_numpy(machine_model).to(device), "machine_d6_mask": torch.from_numpy(machine_mask).to(device), "machine_available": torch.full((n,), machine is not None, device=device),
             "body_d6_model": torch.from_numpy(body_model).to(device), "body_d6_mask": torch.from_numpy(body_mask).to(device), "body_available": torch.full((n,), body is not None, device=device)}
    raw = _forward(model, batch, task_id).cpu().numpy() * pre.scale_uV_by_source["d12"][None, :, None]
    submit = raw.copy(); submit[:, :1] = anchor
    return raw.astype(np.float32), submit.astype(np.float32)


def _metadata(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["array_index", "subject_id", "input_type", "split"]); writer.writeheader()
        for i, row in enumerate(rows): writer.writerow({"array_index": i, "subject_id": row.get("subject_id", ""), "input_type": row.get("input_type", ""), "split": row.get("split", "validation")})


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", required=True); p.add_argument("--run-dir")
    p.add_argument("--task-id", choices=("task1", "task2"), required=True); p.add_argument("--output-dir", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--validation", action="store_true"); p.add_argument("--context-view", default="auto")
    p.add_argument("--body-scale-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    p.add_argument("--context-channel-indices", nargs="+", type=int); p.add_argument("--anchor-npy")
    p.add_argument("--watch-context-npy"); p.add_argument("--machine-d6-context-npy"); p.add_argument("--body-d6-context-npy"); p.add_argument("--centered-diagnostic", action="store_true")
    a = p.parse_args(); indices = tuple(a.context_channel_indices) if a.context_channel_indices else None
    if not a.validation and not a.anchor_npy: raise SystemExit("inference requires --anchor-npy; --target is intentionally unsupported")
    path = Path(a.checkpoint).resolve(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    pre, (model, checkpoint) = _preprocessor(Path(a.run_dir).resolve() if a.run_dir else path.parent), _model(path, a.device)
    if not a.validation:
        raw, submit = _explicit(model, checkpoint, pre, a.task_id, np.load(a.anchor_npy, mmap_mode="r"),
            np.load(a.watch_context_npy, mmap_mode="r") if a.watch_context_npy else None,
            np.load(a.machine_d6_context_npy, mmap_mode="r") if a.machine_d6_context_npy else None,
            np.load(a.body_d6_context_npy, mmap_mode="r") if a.body_d6_context_npy else None, indices, a.device)
        np.save(out / "prediction_raw.npy", raw); np.save(out / "prediction_submit.npy", submit)
        (out / "inference_contract.json").write_text(json.dumps({"anchor_required": True, "target_accepted": False, "schema": checkpoint["schema"], "stage": checkpoint["stage"]}, indent=2), encoding="utf-8"); return
    data = build_joint_dataset(ROOT / "configs" / "common.yaml", a.task_id, "validation", pre, a.body_scale_variant, indices, a.context_view)
    raw, target, anchor, meta = [], [], [], []
    for batch in DataLoader(data, batch_size=16, shuffle=False, collate_fn=b2_collate):
        moved = {k: v.to(a.device) if torch.is_tensor(v) else v for k,v in batch.items()}; raw.append(_forward(model, moved, a.task_id).cpu().numpy() * pre.scale_uV_by_source["d12"][None,:,None])
        target.append(batch["raw_target_uV"].numpy()); anchor.append(batch["raw_anchor_i_uV"].numpy()); meta.extend(batch["meta"])
    raw_a, target_a, anchor_a = np.concatenate(raw), np.concatenate(target), np.concatenate(anchor); np.save(out / "prediction_raw.npy", raw_a); np.save(out / "prediction_submit.npy", np.concatenate((anchor_a, raw_a[:,1:]),1)); np.save(out / "validation_target.npy", target_a); np.save(out / "validation_anchor_i.npy", anchor_a)
    metadata = out / "validation_metadata.csv"; _metadata(metadata, meta)
    command = [sys.executable, "-m", "ecg12gen.evaluate", "--prediction", str(out / "prediction_raw.npy"), "--anchor", str(out / "validation_anchor_i.npy"), "--target", str(out / "validation_target.npy"), "--metadata", str(metadata), "--task-id", a.task_id, "--output-dir", str(out)]
    if a.centered_diagnostic: command.append("--write-centered-diagnostic")
    subprocess.run(command, cwd=ROOT, check=True)

if __name__ == "__main__": main()
