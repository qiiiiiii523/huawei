"""Visible 120s I -> frozen B5 FM -> full-record fast_missing11; no slow model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from baselines.B5.metadata import encode_demographics
from . import MISSING_LEADS, PROTOCOL
from .components import WaveComponent, compose, extract_fast
from .package import frozen_model, load
from .profile import file_sha, load_profile, read, resolve_checkpoint, write
from .sampling import sample_fast_fm


class Predictor:
    """Immutable-in-use original model, independent of slow/history assets."""

    def __init__(self, checkpoint=None, device="cpu"):
        self.profile = load_profile()
        self.checkpoint = resolve_checkpoint(self.profile, checkpoint)
        self.payload = load(self.checkpoint, self.profile)
        self.device = torch.device(device)
        self.fm = frozen_model(self.payload["fm"], self.device)
        self.scale = np.asarray(self.payload["fm"]["scales"]["d12"], np.float32)

    @torch.no_grad()
    def predict_encoded(self, anchor_uV, demographics, record_key, batch_size=1):
        raw = np.asarray(anchor_uV, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (1, 60000) or not np.isfinite(raw).all():
            raise ValueError("Visible I must be finite [1,60000], 500Hz, uV")
        if not isinstance(record_key, str) or not record_key or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Stable record key and positive integer batch size required")
        if self.fm.training or any(p.requires_grad for p in self.fm.parameters()):
            raise RuntimeError("The B5 fast model must remain frozen in eval mode")
        meta = {k: np.asarray(demographics[k]) for k in ("numeric", "sex", "field_mask", "age_topcoded")}
        full = np.empty((12, 60000), np.float32)
        starts = list(range(0, 60000, 5000))
        for offset in range(0, len(starts), batch_size):
            group = starts[offset:offset + batch_size]
            batch = {"anchor": torch.from_numpy(np.stack([raw[:, s:s + 5000] / self.scale[0] for s in group])).to(self.device)}
            batch.update({k: torch.from_numpy(np.stack([v for _ in group])).to(self.device) for k, v in meta.items()})
            generated = sample_fast_fm(self.fm, batch, [record_key + ":" + str(s) for s in group])
            restored = generated.float().cpu().numpy() * self.scale[None, :, None]
            for i, start in enumerate(group):
                full[:, start:start + 5000] = restored[i]
        return WaveComponent(record_key, extract_fast(full), "fast", PROTOCOL)

    def predict(self, anchor_uV, metadata, record_key, batch_size=1):
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must contain only visible demographics")
        return self.predict_encoded(anchor_uV, encode_demographics(metadata), record_key, batch_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", help="Standalone FM-only package; default models/b5_fast.pt")
    p.add_argument("--anchor-npy", required=True)
    p.add_argument("--metadata-json")
    p.add_argument("--record-key", required=True)
    p.add_argument("--output", required=True, help="NEW output directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Use a NEW output directory")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    engine = Predictor(args.checkpoint, device)
    anchor = np.load(args.anchor_npy, allow_pickle=False).astype(np.float32)
    if anchor.ndim == 1:
        anchor = anchor[None]
    result = engine.predict(anchor, read(args.metadata_json) if args.metadata_json else {}, args.record_key, args.batch_size)
    out.mkdir(parents=True)
    # Retain float64 fast residual so future addition rounds just once, like legacy combine.
    np.save(out / "fast_missing11_uV.npy", result.values_uV, allow_pickle=False)
    np.save(out / "observed_I_uV.npy", anchor, allow_pickle=False)
    np.save(out / "fast_only_d12_uV.npy", compose(anchor, result), allow_pickle=False)
    write(out / "inference.json", {"protocol": PROTOCOL, "output_kind": "fast_only_not_complete_reconstruction",
        "record_key": args.record_key, "missing_lead_order": list(MISSING_LEADS), "shape": [11, 60000],
        "unit": "uV", "sampling_rate_hz": 500, "start_sample": 0,
        "checkpoint_sha256": file_sha(engine.checkpoint), "fm_state_sha256": engine.profile["fm_state_sha256"],
        "sampling": engine.profile["sampling"], "decomposition": engine.profile["decomposition"],
        "fast_array_sha256": file_sha(out / "fast_missing11_uV.npy"),
        "observed_I_sha256": file_sha(out / "observed_I_uV.npy"),
        "slow_model_loaded": False, "context_enabled": False, "optimizer_updates": 0,
        "target_used_at_inference": False})
    print({"output": str(out.resolve()), "shape": [11, 60000], "slow_model_loaded": False})


if __name__ == "__main__":
    main()
