"""Export 11 corrected missing leads for the frozen step-3000 validation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.cnn_tra_data import HuaweiSevenLeadData, TASKS
from ecg12gen.cnn_tra_corrector import (FineCorrector, ChestMidCorrector,
                                         compose_d12, make_features, mid_features)
from ecg12gen.cnn_tra_corrector_data import load_validation_records, sha256
from ecg12gen.evaluate import (evaluate_record_predictions,
                                correlation_score_from_summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/root/autodl-tmp/huawei_data")
    parser.add_argument("--cnn-run", default="/root/autodl-tmp/cnn_tra_runs/first_run")
    parser.add_argument("--corrector-run", default="/root/autodl-tmp/cnn_tra_runs/corrector_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    data = HuaweiSevenLeadData(args.data_root)
    cnn_run, run = Path(args.cnn_run), Path(args.corrector_run)
    fine_path, mid_path = run / "fine_0p1/best.pt", run / "chest_mid/best.pt"
    cnn_path = cnn_run / "model_step_003000.pt"
    if not (run / "fine_0p1/complete.json").exists() or not (run / "chest_mid/complete.json").exists():
        raise ValueError("both corrector training phases must be complete")
    fine_state = torch.load(fine_path, map_location="cpu", weights_only=False)
    mid_state = torch.load(mid_path, map_location="cpu", weights_only=False)
    cnn_sha = sha256(cnn_path)
    if fine_state["cnn_checkpoint_sha256"] != cnn_sha or mid_state["cnn_checkpoint_sha256"] != cnn_sha:
        raise ValueError("corrector checkpoint does not match frozen CNN")
    scale8 = np.asarray(fine_state["scale8_uV"], dtype=np.float32)
    if mid_state["scale8_uV"] != fine_state["scale8_uV"]:
        raise ValueError("corrector scales disagree")
    device = torch.device(args.device)
    fine = FineCorrector(scale8).to(device).eval()
    fine.load_state_dict(fine_state["model"], strict=True)
    mid = ChestMidCorrector(scale8).to(device).eval()
    mid.load_state_dict(mid_state["model"], strict=True)
    records = load_validation_records(data, cnn_run / "step_003000")
    output = run / "final_validation"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    outputs = {}
    filled = {}
    for task in TASKS:
        folder = output / task
        folder.mkdir()
        outputs[task] = np.lib.format.open_memmap(folder / "prediction_11_uV.npy", mode="w+",
            dtype=np.float32, shape=(len(data.validation_rows[task]), 11, 5000))
        filled[task] = np.zeros(len(data.validation_rows[task]), dtype=bool)
    with torch.no_grad():
        for n, record in enumerate(records, 1):
            feature = make_features(record.anchor, record.base7, scale8, 0.1)
            fine_delta, _ = fine(feature[None].to(device))
            fine7 = record.base7 + fine_delta[0].cpu().numpy()
            feature_mid = mid_features(record.anchor, record.base7, fine7, scale8)
            mid_delta, _ = mid(feature_mid[None].to(device))
            final7 = fine7.copy()
            final7[1:] += mid_delta[0].cpu().numpy()
            full = compose_d12(torch.from_numpy(record.anchor[None]),
                               torch.from_numpy(final7[None]))[0].numpy()
            missing = full[1:].reshape(11, 12, 5000).transpose(1, 0, 2)
            task, indices = record.meta["task"], record.meta["validation_array_indices"]
            outputs[task][indices] = missing
            filled[task][indices] = True
            if n % 20 == 0:
                print(f"[export] {n}/{len(records)} records", flush=True)
    if any(not filled[task].all() for task in TASKS):
        raise ValueError("validation output misses windows")
    summaries = {}
    for task in TASKS:
        outputs[task].flush()
        full = np.concatenate((data.validation_arrays[task][:, :1], outputs[task]), axis=1)
        quality = data.validation_masks[task]
        summary, _ = evaluate_record_predictions(
            full, data.validation_arrays[task], task, data.validation_rows[task],
            target_quality_mask=quality, input_quality_mask=quality[:, 0],
            require_reliable_input=True, input_quality_source="same_record_target_I")
        summaries[task] = summary
    score = correlation_score_from_summaries(summaries)
    manifest = {"schema": "cnn_tra_corrected_validation_11_v1", "R": score,
                "tasks": summaries, "cnn_step": 3000,
                "cnn_checkpoint_sha256": cnn_sha,
                "fine_checkpoint_sha256": sha256(fine_path),
                "mid_checkpoint_sha256": sha256(mid_path),
                "unit": "uV", "sampling_rate_hz": 500,
                "lead_order": ["II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                "row_order": "same as frozen step-3000 validation cache and source validation index"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"[export] validation score={json.dumps(score)}; {output}", flush=True)


if __name__ == "__main__":
    main()
