"""B2 inference with explicit target-time machine-I anchor input."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.b2_data import b2_collate, build_joint_dataset
from ecg12gen.b2_model import B2MaskedPatchTransformer, B2ModelConfig
from ecg12gen.b2_train import P0_STAGE, _forward
from ecg12gen.contracts import ContractError, WINDOW_SAMPLES, canonical_lead_mask, prepare_joint_anchor_inference
from ecg12gen.evaluate import evaluate_joint_anchor_predictions, evaluate_task2_diagnostics
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def _load_preprocessor(run_dir: Path) -> ECGPreprocessor:
    config = PreprocessingConfig.from_yaml(ROOT / "configs" / "preprocessing.yaml")
    scales = json.loads((run_dir / "preprocessing_scales.json").read_text(encoding="utf-8"))
    return ECGPreprocessor(config, {key: np.asarray(value, dtype=np.float32) for key, value in scales.items()})


def _load_model(path: Path, device: str) -> tuple[B2MaskedPatchTransformer, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = B2ModelConfig(**checkpoint.get("model_config", {}))
    model = B2MaskedPatchTransformer(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def _context_model(values: np.ndarray, source: str, preprocessor: ECGPreprocessor,
                   indices: tuple[int, ...] | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != WINDOW_SAMPLES:
        raise ContractError("context must have shape [N,C,5000]")
    chosen = (0,) if source == "watch_ecg" else (indices or tuple(range(6)))
    if values.shape[1] != len(chosen):
        raise ContractError("context channels disagree with selected channel indices")
    if source == "watch_ecg":
        return preprocessor.transform_batch(values, source)[0], np.ones((len(values), 1), dtype=bool)
    full = np.zeros((len(values), 6, WINDOW_SAMPLES), dtype=np.float32)
    full[:, list(chosen)] = values
    transformed = preprocessor.transform_batch(full, source)[0][:, list(chosen)]
    mask = np.zeros((len(values), 6), dtype=bool); mask[:, list(chosen)] = True
    return transformed, mask


@torch.no_grad()
def _predict_explicit(model: B2MaskedPatchTransformer, checkpoint: dict[str, object], preprocessor: ECGPreprocessor,
                      task_id: str, anchor_raw: np.ndarray, context_raw: np.ndarray | None,
                      context_source: str | None, indices: tuple[int, ...] | None, device: str) -> tuple[np.ndarray, np.ndarray]:
    anchor_raw = np.asarray(anchor_raw, dtype=np.float32)
    if anchor_raw.ndim != 3 or anchor_raw.shape[1:] != (1, WINDOW_SAMPLES):
        raise ContractError("--anchor-npy must be [N,1,5000]")
    prepare_joint_anchor_inference(anchor_raw[0], context_source_type=context_source or ("watch_ecg" if task_id == "task1" else "ecg_machine_d6"), task_id=task_id, anchor_i_ecg=anchor_raw[0], context_channel_indices=indices)
    anchor_model = preprocessor.transform_batch(anchor_raw, "ecg_machine_i")[0]
    anchor_mask = np.broadcast_to(canonical_lead_mask(1), (len(anchor_raw), 12)).copy()
    stage = str(checkpoint.get("stage", P0_STAGE))
    if stage == P0_STAGE:
        output = model.forward_anchor(torch.from_numpy(anchor_model).to(device), torch.from_numpy(anchor_mask).to(device))
    else:
        if context_raw is None or context_source is None:
            raise ContractError("P1 inference requires context and explicit machine-I anchor")
        context_model, context_mask = _context_model(context_raw, context_source, preprocessor, indices)
        output = model(torch.from_numpy(context_model).to(device), torch.from_numpy(anchor_model).to(device),
                       torch.from_numpy(context_mask).to(device), torch.from_numpy(anchor_mask).to(device))
    raw = output.cpu().numpy() * preprocessor.scale_uV_by_source["d12"][None, :, None]
    submit = raw.copy(); submit[:, :1] = anchor_raw
    return raw.astype(np.float32), submit.astype(np.float32)


def _metadata(path: Path, rows: list[dict[str, object]]) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["array_index", "subject_id", "input_type", "split"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"array_index": index, "subject_id": row.get("subject_id", ""),
                             "input_type": row.get("input_type", ""), "split": "validation"})


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--run-dir", default=None)
    parser.add_argument("--task-id", choices=("task1", "task2"), required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--validation", action="store_true")
    parser.add_argument("--anchor-npy", default=None); parser.add_argument("--context-npy", default=None)
    parser.add_argument("--context-source", choices=("watch_ecg", "ecg_machine_d6", "body_scale_d6"), default=None)
    parser.add_argument("--body-scale-variant", choices=("A_raw_window", "B_detrend_0p2Hz_then_window"), default="A_raw_window")
    parser.add_argument("--context-view", choices=("auto", "machine", "body", "watch"), default="auto")
    parser.add_argument("--context-channel-indices", nargs="+", type=int, default=None)
    parser.add_argument("--centered-diagnostic", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    preprocessor = _load_preprocessor(Path(args.run_dir).resolve() if args.run_dir else checkpoint_path.parent)
    model, checkpoint = _load_model(checkpoint_path, args.device)
    indices = tuple(args.context_channel_indices) if args.context_channel_indices else None

    if not args.validation:
        if not args.anchor_npy:
            parser.error("inference requires --anchor-npy; target input is unsupported")
        context = np.load(args.context_npy, mmap_mode="r") if args.context_npy else None
        source = args.context_source or ("watch_ecg" if args.task_id == "task1" else "ecg_machine_d6")
        raw, submit = _predict_explicit(model, checkpoint, preprocessor, args.task_id,
                                        np.load(args.anchor_npy, mmap_mode="r"), context, source, indices, args.device)
        np.save(output / "prediction_raw.npy", raw); np.save(output / "prediction_submit.npy", submit)
        (output / "inference_contract.json").write_text(json.dumps({
            "anchor_required": True, "target_accepted": False, "context_target_relation": "same_subject_cross_time",
            "anchor_target_relation": "same_record_same_window", "stage": checkpoint.get("stage"),
        }, indent=2), encoding="utf-8")
        return

    if args.task_id == "task2" and args.context_view == "auto":
        raise ContractError("validation task2 requires one context view: machine or body")
    data = build_joint_dataset(ROOT / "configs" / "common.yaml", args.task_id, "validation", preprocessor,
                               args.body_scale_variant, indices,
                               args.context_view if args.context_view != "auto" else "watch")
    raw_parts: list[np.ndarray] = []; target_parts: list[np.ndarray] = []; anchor_parts: list[np.ndarray] = []; metadata: list[dict[str, object]] = []
    for batch in DataLoader(data, batch_size=16, shuffle=False, collate_fn=b2_collate):
        moved = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        raw_parts.append(_forward(model, moved, str(checkpoint.get("stage", P0_STAGE))).cpu().numpy() * preprocessor.scale_uV_by_source["d12"][None, :, None])
        target_parts.append(batch["raw_target_uV"].numpy()); anchor_parts.append(batch["raw_anchor_i_uV"].numpy()); metadata.extend(batch["meta"])
    raw = np.concatenate(raw_parts).astype(np.float32); target = np.concatenate(target_parts); anchor = np.concatenate(anchor_parts)
    summary, _, _, submit = evaluate_joint_anchor_predictions(raw, target, anchor, args.task_id)
    np.save(output / "prediction_raw.npy", raw); np.save(output / "prediction_submit.npy", submit)
    np.save(output / "validation_target.npy", target); np.save(output / "validation_anchor_i.npy", anchor)
    _metadata(output / "validation_metadata.csv", metadata)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if args.task_id == "task2":
        subjects, devices = evaluate_task2_diagnostics(submit, target, [{"subject_id": str(x.get("subject_id", "")), "input_type": str(x.get("input_type", ""))} for x in metadata])
        (output / "task2_diagnostics.json").write_text(json.dumps({"subject_macro": subjects, "device": devices}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
