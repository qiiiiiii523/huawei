"""Frozen-B8 120-second cached-prediction TCN correction stage.

The correction never receives an anchor, context, or target at inference.  Its only
signal input is B8's cached seven-lead prediction.  It predicts a 5-Hz correction
curve and explicitly removes each full-record lead mean before addition.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.B7.data import load_record, read_manifest
from baselines.B7.evaluate import limb_diagnostics, pearson
from baselines.B7.leads import compose_d12
from .checkpoint import load as load_b8_checkpoint, model_from_checkpoint
from .evaluate import aggregate
from .inference import predict_generated_records

PROTOCOL = "B8-frozen-cache-tcn120-demeaned-v1"
GENERATED = (1, 6, 7, 8, 9, 10, 11)
SAMPLES, CURVE_HZ, CURVE_STEPS = 60000, 5, 600

def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def _name(key): return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npy"

class TCNBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.one = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.two = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
    def forward(self, x): return x + self.two(F.silu(self.one(F.silu(self.norm(x)))))

class CachedPredictionTCN120(nn.Module):
    """Non-causal 120s TCN. Zero head makes epoch zero exactly equal frozen B8."""
    def __init__(self, width=32, dilations=(1, 2, 4, 8, 16, 32, 64, 128)):
        super().__init__()
        if width < 8 or width % 8 or not dilations or any(d < 1 for d in dilations): raise ValueError("invalid TCN configuration")
        self.width, self.dilations = width, tuple(dilations)
        self.input = nn.Conv1d(7, width, 1)
        self.blocks = nn.ModuleList(TCNBlock(width, d) for d in self.dilations)
        self.head = nn.Conv1d(width, 7, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, prediction):
        if prediction.ndim != 3 or prediction.shape[1:] != (7, SAMPLES): raise ValueError("TCN input must be cached [B,7,60000]")
        # Fixed average pooling makes the correction a 5-Hz full-record curve.
        x = F.avg_pool1d(prediction, kernel_size=100, stride=100)
        x = self.input(x)
        for block in self.blocks: x = block(x)
        curve = self.head(F.silu(x))
        if curve.shape[-1] != CURVE_STEPS: raise RuntimeError("unexpected curve length")
        full = F.interpolate(curve, size=SAMPLES, mode="linear", align_corners=False)
        return full - full.mean(-1, keepdim=True)

def _cache_manifest(cache_dir):
    return json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))

@torch.no_grad()
def build_cache(checkpoint_path, manifest_path, cache_dir, device="cpu"):
    """Write only frozen B8 predictions, never target arrays, to a 120s cache."""
    checkpoint_path, manifest_path, cache_dir = map(Path, (checkpoint_path, manifest_path, cache_dir))
    checkpoint, manifest = load_b8_checkpoint(checkpoint_path), read_manifest(manifest_path)
    if cache_dir.exists(): raise FileExistsError("use a new B8 prediction-cache directory")
    model = model_from_checkpoint(checkpoint, device).eval()
    cache_dir.mkdir(parents=True); rows, seen = [], {}
    for row in manifest["records"]:
        if not row["input_reliable"]: continue
        anchor, target, mask, _ = load_record(row)
        if anchor.shape != (1, SAMPLES) or target.shape != (12, SAMPLES): raise ValueError("TCN120 cache requires real complete 120-second records")
        key = row["target_record_id"]
        fingerprint = (row["subject_id"], hashlib.sha256(anchor.tobytes()).hexdigest(), mask.tobytes())
        if key in seen:
            if seen[key] != fingerprint: raise ValueError("same target cache key has inconsistent visible input/QC")
            continue
        predicted = predict_generated_records(model, [anchor], checkpoint["scales_uV"])[0].cpu().numpy().astype("float32")
        file = _name(key); np.save(cache_dir / file, predicted, allow_pickle=False)
        rows.append({"target_record_id": key, "subject_id": row["subject_id"], "split": row["split"], "file": file,
                     "sha256": _sha(cache_dir / file), "lead_mask": mask[list(GENERATED)].astype(int).tolist()})
        seen[key] = fingerprint
    value = {"protocol": PROTOCOL, "base_checkpoint": str(checkpoint_path.resolve()), "base_checkpoint_sha256": _sha(checkpoint_path),
             "source_manifest": str(manifest_path.resolve()), "source_manifest_sha256": _sha(manifest_path),
             "scales_uV": checkpoint["scales_uV"], "samples": SAMPLES, "prediction_channels": 7,
             "target_stored": False, "records": rows}
    (cache_dir / "manifest.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value

def _load_cache(cache_dir, manifest_path):
    cache = _cache_manifest(cache_dir); manifest = read_manifest(manifest_path)
    if cache.get("protocol") != PROTOCOL or cache.get("samples") != SAMPLES or cache.get("prediction_channels") != 7 or cache.get("target_stored") is not False:
        raise ValueError("invalid B8 TCN prediction cache")
    if cache["source_manifest_sha256"] != _sha(manifest_path) or cache["scales_uV"] != manifest["scales_uV"]: raise ValueError("cache/manifest identity mismatch")
    entries = {r["target_record_id"]: r for r in cache["records"]}
    source = {}
    for row in manifest["records"]:
        if row["input_reliable"] and row["target_record_id"] in entries: source.setdefault(row["target_record_id"], row)
    if set(entries) != set(source): raise ValueError("cache must cover exactly reliable manifest targets")
    return cache, manifest, entries, source

def _target(row):
    anchor, target, mask, _ = load_record(row)
    if anchor.shape != (1, SAMPLES) or target.shape != (12, SAMPLES): raise ValueError("TCN120 only accepts complete real120 targets")
    return anchor, target, mask

def _prediction(cache_dir, entry):
    value = np.load(Path(cache_dir) / entry["file"], allow_pickle=False)
    if value.shape != (7, SAMPLES) or value.dtype != np.float32 or _sha(Path(cache_dir) / entry["file"]) != entry["sha256"]: raise ValueError("corrupt cached B8 prediction")
    return value

def load_correction(path, device="cpu"):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL or value.get("config", {}).get("demeaned") is not True:
        raise ValueError("expected a demeaned B8 TCN120 correction checkpoint")
    config = value["config"]; model = CachedPredictionTCN120(config["width"], tuple(config["dilations"]))
    model.load_state_dict(value["model"], strict=True)
    return model.to(device).eval(), value

def correct_generated(model, prediction_uV, scales, device="cpu"):
    """Apply the cache-only TCN to one raw-uV, 7×120s B8 prediction."""
    prediction_uV = np.asarray(prediction_uV, dtype="float32")
    if prediction_uV.shape != (7, SAMPLES) or not np.isfinite(prediction_uV).all(): raise ValueError("need finite cached B8 [7,60000] prediction")
    x = torch.from_numpy(prediction_uV[None] / np.asarray(scales)[list(GENERATED), None]).to(device=device, dtype=torch.float32)
    with torch.no_grad(): delta = model(x).cpu().numpy()[0] * np.asarray(scales)[list(GENERATED), None]
    # Assertion protects the central purpose of this candidate: it cannot inject a DC correction.
    if not np.allclose(delta.mean(-1), 0., atol=2e-5): raise AssertionError("full120 correction must be per-lead demeaned")
    return prediction_uV + delta.astype("float32"), delta.astype("float32")

def _correct(model, prediction_uV, scales, device):
    return correct_generated(model, prediction_uV, scales, device)

def evaluate_cache(model, cache_dir, manifest_path, device="cpu"):
    cache, manifest, entries, source = _load_cache(cache_dir, manifest_path); rows = []
    for key, entry in entries.items():
        row = source[key]
        if row["split"] != "validation": continue
        anchor, target, mask = _target(row); predicted, _ = _correct(model, _prediction(cache_dir, entry), cache["scales_uV"], device)
        d12 = compose_d12(torch.from_numpy(anchor[None]), torch.from_numpy(predicted[None]))[0].numpy()
        leads = {str(j): {"r": (r if np.isfinite(r := pearson(d12[j], target[j])) else None),
                           "rmse_uV": float(np.sqrt(np.mean((d12[j] - target[j]) ** 2)))} for j in range(1, 12) if mask[j]}
        if leads: rows.append({"record_id": row["record_id"], "target_record_id": key, "group": row["group"], "leads": leads,
                               "limb_diagnostics": limb_diagnostics(anchor, d12, target, mask)})
    return aggregate(rows, "huawei40"), rows

def train(cache_dir, manifest_path, output, *, device="cpu", epochs=10, batch_size=2, learning_rate=2e-4):
    cache_dir, manifest_path, output = map(Path, (cache_dir, manifest_path, output))
    if output.exists(): raise FileExistsError("use a new TCN120 output directory")
    cache, manifest, entries, source = _load_cache(cache_dir, manifest_path)
    train_ids = [key for key, row in source.items() if row["split"] == "train"]
    if not train_ids: raise ValueError("TCN120 needs training records")
    torch.manual_seed(42); np.random.seed(42); torch.use_deterministic_algorithms(True); device = torch.device(device)
    model = CachedPredictionTCN120().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    output.mkdir(parents=True); best = -float("inf"); history = []
    scales = np.asarray(cache["scales_uV"], dtype="float32")[list(GENERATED), None]
    for epoch in range(0, epochs + 1):
        if epoch:
            model.train(); order = np.random.permutation(train_ids); total = 0.; steps = 0
            for start in range(0, len(order), batch_size):
                ids = order[start:start + batch_size]; pred = np.stack([_prediction(cache_dir, entries[k]) for k in ids]) / scales[None]
                label, mask = [], []
                for key in ids:
                    _, target, quality = _target(source[key]); label.append(target[list(GENERATED)] / scales); mask.append(quality[list(GENERATED)])
                x, y = torch.from_numpy(pred).to(device), torch.from_numpy(np.stack(label)).to(device)
                quality = torch.from_numpy(np.stack(mask)).to(device).bool(); delta = model(x); error = delta - (y - x)
                valid = quality[:, :, None]
                loss = torch.where(valid, error.square(), torch.zeros_like(error)).sum() / valid.sum().clamp_min(1) / SAMPLES
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                total += float(loss.detach()); steps += 1
        metrics, rows = evaluate_cache(model.eval(), cache_dir, manifest_path, device); row = {"epoch": epoch, "train_mse": None if not epoch else total / max(1, steps), "validation": metrics}; history.append(row)
        payload = {"protocol": PROTOCOL, "model": model.state_dict(), "config": {"width": model.width, "dilations": model.dilations, "curve_hz": CURVE_HZ, "demeaned": True},
                   "base_checkpoint_sha256": cache["base_checkpoint_sha256"], "cache_manifest_sha256": _sha(cache_dir / "manifest.json"), "epoch": epoch, "best_r": best}
        if metrics["r"] > best: best = metrics["r"]; payload["best_r"] = best; torch.save(payload, output / "best_r.pt")
        torch.save(payload, output / "last.pt"); (output / f"validation_{epoch:03d}.json").write_text(json.dumps({"metrics": metrics, "records": rows}, ensure_ascii=False), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "completed.json").write_text(json.dumps({"completed": True, "best_r": best, "epochs": epochs, "protocol": PROTOCOL}), encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(required=True)
    c = sub.add_parser("cache"); c.add_argument("--checkpoint", required=True); c.add_argument("--manifest", required=True); c.add_argument("--output", required=True); c.add_argument("--device", default="cpu")
    t = sub.add_parser("train"); t.add_argument("--cache", required=True); t.add_argument("--manifest", required=True); t.add_argument("--output", required=True); t.add_argument("--device", default="cpu"); t.add_argument("--epochs", type=int, default=10); t.add_argument("--batch-size", type=int, default=2); t.add_argument("--learning-rate", type=float, default=2e-4)
    args = p.parse_args()
    if args.cmd == "cache": build_cache(args.checkpoint, args.manifest, args.output, args.device)
    else: train(args.cache, args.manifest, args.output, device=args.device, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
if __name__ == "__main__": main()
