"""Apply trained 120-second fine and chest-mid correctors to CNN+TRA output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg12gen.cnn_tra_corrector import (FineCorrector, ChestMidCorrector,
                                         compose_d12, make_features, mid_features)
from ecg12gen.cnn_tra_corrector_data import sha256


@torch.no_grad()
def apply(anchor: np.ndarray, base7: np.ndarray,
          fine_checkpoint: str | Path, mid_checkpoint: str | Path,
          cnn_checkpoint: str | Path, device: torch.device) -> np.ndarray:
    anchor = np.array(anchor, dtype=np.float32, copy=True).reshape(1, 60000)
    base7 = np.array(base7, dtype=np.float32, copy=True)
    if base7.shape != (7, 60000) or not np.isfinite(anchor).all() or not np.isfinite(base7).all():
        raise ValueError("expected finite I [1,60000] and seven predictions [7,60000]")
    fine_state = torch.load(fine_checkpoint, map_location="cpu", weights_only=False)
    mid_state = torch.load(mid_checkpoint, map_location="cpu", weights_only=False)
    scale8 = np.asarray(fine_state["scale8_uV"], dtype=np.float32)
    if scale8.shape != (8,) or mid_state["scale8_uV"] != fine_state["scale8_uV"]:
        raise ValueError("corrector scale metadata mismatch")
    if mid_state["cnn_checkpoint_sha256"] != fine_state["cnn_checkpoint_sha256"]:
        raise ValueError("correctors were trained on different CNN checkpoints")
    if sha256(cnn_checkpoint) != fine_state["cnn_checkpoint_sha256"]:
        raise ValueError("base CNN checkpoint does not match corrector training")
    fine = FineCorrector(scale8).to(device).eval()
    fine.load_state_dict(fine_state["model"], strict=True)
    mid = ChestMidCorrector(scale8).to(device).eval()
    mid.load_state_dict(mid_state["model"], strict=True)
    fine_feature = make_features(anchor, base7, scale8, 0.1)
    fine_correction, _ = fine(fine_feature[None].to(device))
    fine7 = base7 + fine_correction[0].cpu().numpy()
    mid_feature = mid_features(anchor, base7, fine7, scale8)
    mid_correction, _ = mid(mid_feature[None].to(device))
    corrected7 = fine7.copy()
    corrected7[1:] += mid_correction[0].cpu().numpy()
    d12 = compose_d12(torch.from_numpy(anchor[None]),
                      torch.from_numpy(corrected7[None]))[0].numpy()
    missing11 = np.asarray(d12[1:], dtype=np.float32)
    if missing11.shape != (11, 60000) or not np.isfinite(missing11).all():
        raise RuntimeError("invalid corrected 11-lead output")
    return missing11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-i", required=True, help="raw-uV 120-s I .npy")
    parser.add_argument("--base-seven", required=True,
                        help="raw-uV CNN output .npy, order II,V1..V6")
    parser.add_argument("--fine-checkpoint", required=True)
    parser.add_argument("--mid-checkpoint", required=True)
    parser.add_argument("--cnn-checkpoint", required=True,
                        help="frozen CNN+TRA step-3000 checkpoint")
    parser.add_argument("--output", required=True, help="new [11,60000] .npy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = apply(np.load(args.input_i, allow_pickle=False),
                   np.load(args.base_seven, allow_pickle=False),
                   args.fine_checkpoint, args.mid_checkpoint,
                   args.cnn_checkpoint, torch.device(args.device))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)
    meta = {"schema": "cnn_tra_corrected_missing11_v1", "unit": "uV",
            "sampling_rate_hz": 500, "samples": 60000,
            "lead_order": ["II", "III", "aVR", "aVL", "aVF",
                           "V1", "V2", "V3", "V4", "V5", "V6"],
            "fine_checkpoint_sha256": sha256(args.fine_checkpoint),
            "mid_checkpoint_sha256": sha256(args.mid_checkpoint),
            "cnn_checkpoint_sha256": sha256(args.cnn_checkpoint),
            "input_i": str(Path(args.input_i).resolve()),
            "base_seven": str(Path(args.base_seven).resolve())}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
