"""Apply a trained B5-final TCN correction without changing lead I or lead means."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .model import OUTPUT_INDICES, TrendCorrector, compose_d12, make_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, help="B5-final [12,60000] .npy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.prediction), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    prediction = np.load(source, allow_pickle=False).astype("float32", copy=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    scales = np.asarray(ckpt["config"]["scales_uV"], dtype="float32")
    model = TrendCorrector(scales); model.load_state_dict(ckpt["model"], strict=True); model.eval()
    with torch.no_grad(): correction, _ = model(make_features(prediction, scales)[None])
    generated = torch.from_numpy(prediction[list(OUTPUT_INDICES)].copy())[None] + correction
    result = compose_d12(torch.from_numpy(prediction[:1].copy())[None], generated)[0].numpy()
    if not np.array_equal(result[0], prediction[0]) or not np.allclose(result[list(OUTPUT_INDICES)].mean(-1), prediction[list(OUTPUT_INDICES)].mean(-1), atol=2e-5):
        raise AssertionError("TCN changed visible I or II/V1..V6 120-second means")
    output.parent.mkdir(parents=True, exist_ok=True); np.save(output, result, allow_pickle=False)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps({"source_prediction": str(source), "checkpoint": str(args.checkpoint), "target_used": False, "zero_mean_per_generated_lead": True}, indent=2))


if __name__ == "__main__": main()
