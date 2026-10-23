"""New-record MACHINE inference: visible I + legal historical d6, no target caches."""
import argparse
from pathlib import Path
import numpy as np
import torch
from baselines.B5.runtime import device_from_name
from baselines.B5.metadata import encode_demographics
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.slow import load_slow, predict_native
from baselines.B5_best.common import read, write
from baselines.B5_slow120.common import load_study as load_slow_study, load_correction
from baselines.B5_slow120.model import visible_inputs, inputs
from baselines.B5_slow120.evaluate import array_batch
from .data import load_study, FIELDS
from .common import load_candidate
from .evaluate import generate
from .sampling import SAMPLING


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "checkpoint", "anchor-npy", "record-key", "output"): p.add_argument("--" + k, required=True)
    p.add_argument("--metadata-json"); p.add_argument("--history-npy"); p.add_argument("--history-mask-npy")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if bool(a.history_npy) != bool(a.history_mask_npy): p.error("History needs both raw-uV and validity mask")
    if Path(a.output).exists(): raise FileExistsError("Use a new prediction output")
    study = load_study(a.run, verify=False); device = device_from_name(a.device)
    model, cp = load_candidate(a.checkpoint, study, a.run, device)
    anchor = np.asarray(np.load(a.anchor_npy, allow_pickle=False), np.float32).reshape(1, -1)
    if anchor.shape != (1, 60000) or not np.isfinite(anchor).all(): raise ValueError("Need finite raw-uV I, 500Hz, real120s")
    scale = np.asarray(study["scale_uV"], np.float32)
    meta = encode_demographics(read(a.metadata_json) if a.metadata_json else {})
    slow_root = Path(study["slow_run"])
    ss = load_slow_study(slow_root, verify_records=False)
    g, _ = load_correction(study["slow_checkpoint"], ss, slow_root / "study.json", device)
    bundle = load_bundle(study["stable_bundle"])
    local = load_slow(bundle["slow"], device, bundle["provenance"]["slow_variant"])
    g.eval()
    with torch.no_grad():
        fields = {k: torch.from_numpy(np.asarray(meta[k])[None].copy()).to(device) for k in FIELDS}
        native = predict_native(local, dict(anchor=torch.from_numpy(
            (anchor.reshape(1, 6000, 10).mean(-1) / scale[0])[None]).to(device), **fields))[0].cpu().numpy()
        glob = g(**inputs(array_batch(visible_inputs(anchor, scale[0], meta), device)))[0].cpu().numpy()
        hist = np.load(a.history_npy, allow_pickle=False) if a.history_npy else np.zeros((6, 1), np.float32)
        mask = np.load(a.history_mask_npy, allow_pickle=False) if a.history_mask_npy else np.zeros((6, 1), bool)
        record = dict(anchor_uV=anchor, target_uV=np.zeros((12, 60000), np.float32), mask=np.ones(12, bool),
                      history_uV=hist, history_mask=mask,
                      fixed_slow=(native.astype(float) + glob.astype(float)) * scale[1:, None], **meta)
        prediction, status = generate(model, record, dict(target_record_id=a.record_key, input_record_id="provided_history"),
                                      scale, device, cp["arm"] == "context_public")
    out = Path(a.output); out.mkdir(parents=True)
    np.save(out / "prediction_uV.npy", prediction)
    write(out / "inference.json", dict(sampling=SAMPLING, arm=cp["arm"], target_used_at_inference=False,
                                      shape=list(prediction.shape), route="ecg_machine_d6", context=status))


if __name__ == "__main__": main()
