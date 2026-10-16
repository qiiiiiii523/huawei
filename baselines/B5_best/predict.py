"""One target-free entry point for the two embedded networks and full-record combination."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5.flow import sample
from baselines.B5.metadata import encode_demographics
from baselines.B5.runtime import device_from_name, seed_all
from . import LEADS
from .bundle import load_bundle, models, bundle_slow_beta
from .combine import combine, upsample_native
from .common import read, write, sha
from .slow import predict_native


class Predictor:
    def __init__(self, bundle, device):
        self.bundle, self.device = bundle, device
        self.fm, self.slow = models(bundle, device)
        self.slow_beta = bundle_slow_beta(bundle)
        self.scale = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)

    @torch.no_grad()
    def predict_encoded(self, anchor_uV, demographics, record_key, batch_size=8):
        raw = np.asarray(anchor_uV, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (1, 60000) or not np.isfinite(raw).all():
            raise ValueError("Input must be a finite 120-second I [1,60000], 500Hz, uV")
        if not isinstance(record_key, str) or not record_key or batch_size < 1:
            raise ValueError("A stable record key and positive batch size are required")
        meta = {k: np.asarray(demographics[k]) for k in ("numeric", "sex", "field_mask", "age_topcoded")}
        fields = {k: torch.from_numpy(v[None].copy()).to(self.device) for k, v in meta.items()}
        # Same block-mean raw I -> normalization -> 50s contexts as the original probe.
        anchor50 = raw.reshape(1, 6000, 10).mean(-1) / self.scale[0]
        native = predict_native(self.slow, {"anchor": torch.from_numpy(anchor50[None]).to(self.device), **fields})[0].cpu().numpy()
        # The probe restores uV in float64 before full-record interpolation.
        learned = upsample_native(native.astype(np.float64) * self.scale[1:, None].astype(np.float64))
        original = np.empty((12, 60000), np.float32)
        starts = list(range(0, 60000, 5000))
        for offset in range(0, 12, batch_size):
            group = starts[offset:offset + batch_size]
            inputs = {"anchor": torch.from_numpy(np.stack([raw[:, s:s + 5000] / self.scale[0] for s in group])).to(self.device)}
            inputs.update({k: torch.from_numpy(np.stack([v for _ in group])).to(self.device) for k, v in meta.items()})
            generated = sample(self.fm, inputs, [record_key + ":" + str(s) for s in group], **self.bundle["sampling"])
            restored = generated.float().cpu().numpy() * self.scale[None, :, None]
            for i, start in enumerate(group):
                original[:, start:start + 5000] = restored[i]
        return combine(original, learned, slow_beta=self.slow_beta)

    def predict(self, anchor_uV, metadata, record_key, batch_size=8):
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be an object containing visible demographic fields")
        return self.predict_encoded(anchor_uV, encode_demographics(metadata), record_key, batch_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Portable B5-best bundle, not the original FM alone")
    p.add_argument("--anchor-npy", required=True)
    p.add_argument("--metadata-json")
    p.add_argument("--record-key", required=True)
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty inference output directory")
    bundle = load_bundle(args.checkpoint)
    seed_all(bundle["sampling"]["seed"], True)
    engine = Predictor(bundle, device_from_name(args.device))
    prediction = engine.predict(np.load(args.anchor_npy, allow_pickle=False),
                                read(args.metadata_json) if args.metadata_json else {}, args.record_key, args.batch_size)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prediction_uV.npy", prediction, allow_pickle=False)
    write(out / "inference.json", {"checkpoint_sha256": sha(args.checkpoint), "shape": list(prediction.shape),
          "lead_order": list(LEADS), "unit": "uV", "sampling_rate_hz": 500, "sampling": bundle["sampling"],
          "slow_beta": bundle_slow_beta(bundle),
          "record_key": args.record_key, "target_used_at_inference": False,
          "prepared_data_required": False, "slow_variant": bundle["provenance"]["slow_variant"]})
    print({"prediction": str(out / "prediction_uV.npy"), "shape": list(prediction.shape)}, flush=True)


if __name__ == "__main__":
    main()
