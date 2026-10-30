"""Train the zero-mean 120-second B5-final trend corrector from a frozen cache."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import OUTPUT_INDICES, TrendCorrector, compose_d12, make_features


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scales(base_bundle: Path) -> np.ndarray:
    from baselines.B5_best.bundle import load_bundle
    scales = np.asarray(load_bundle(base_bundle)["fm"]["scales"]["d12"], dtype=np.float32)
    if scales.shape != (12,):
        raise ValueError("Expected 12 scales_uV")
    return scales


def _load_items(cache: Path, baseline: Path, scales: np.ndarray):
    index = _read_json(cache / "index.json")
    if index.get("record_shape") != [12, 60000]:
        raise ValueError("Not a B5-final 120-second cache")
    source = {}
    for task in ("task1", "task2"):
        folder = baseline / task
        target = np.load(folder / "target_uV.npy", mmap_mode="r", allow_pickle=False)
        quality = np.load(folder / "target_quality_mask.npy", mmap_mode="r", allow_pickle=False)
        if target.ndim != 3 or target.shape[1:] != (12, 5000) or quality.shape != target.shape[:2]:
            raise ValueError(f"Invalid frozen targets for {task}")
        source[task] = (target, quality)
    result = {"train": [], "validation": []}
    for row in index["records"]:
        split, task, indices = row["split"], row["task_id"], row["window_indices"]
        if split not in result or task not in source or len(indices) != 12:
            raise ValueError("Malformed cache index")
        pred = np.load(cache / row["prediction_file"], allow_pickle=False)
        target, quality = source[task]
        y = np.concatenate([target[i] for i in indices], axis=-1).astype("float32", copy=False)
        mask = np.all(quality[indices], axis=0)
        # Lead I is the fixed synthesis anchor, not a corrected/supervised
        # output.  A record with an invalid reference I label can still train
        # the II/V1--V6 residuals from its valid direct-lead labels.
        if pred.shape != (12, 60000) or y.shape != pred.shape or not mask[1:].any():
            raise ValueError(f"Bad cached record {row['pair_id']}")
        result[split].append({"prediction": torch.from_numpy(pred.copy()), "target": torch.from_numpy(y.copy()),
                              "mask": torch.from_numpy(mask[1:].copy()),
                              "output_mask": torch.from_numpy(mask[list(OUTPUT_INDICES)].copy()),
                              "features": make_features(pred, scales)})
    if not result["train"] or not result["validation"]:
        raise ValueError("Cache must contain train and validation records")
    return result


def _loss(model, batch, device):
    features = torch.stack([x["features"] for x in batch]).to(device)
    prediction = torch.stack([x["prediction"] for x in batch]).to(device)
    target = torch.stack([x["target"] for x in batch]).to(device)
    mask = torch.stack([x["mask"] for x in batch]).to(device)
    output_mask = torch.stack([x["output_mask"] for x in batch]).to(device)
    correction, controls = model(features)
    generated = prediction[:, list(OUTPUT_INDICES)] + correction
    corrected = compose_d12(prediction[:, :1], generated)
    pc = corrected[:, 1:] - corrected[:, 1:].mean(-1, keepdim=True)
    yc = target[:, 1:] - target[:, 1:].mean(-1, keepdim=True)
    correlation = (pc * yc).sum(-1) / (pc.square().sum(-1).clamp_min(1) * yc.square().sum(-1).clamp_min(1)).sqrt()
    correlation_loss = ((1 - correlation) * mask).sum() / mask.sum().clamp_min(1)
    residual = (target[:, list(OUTPUT_INDICES)] - prediction[:, list(OUTPUT_INDICES)]).reshape(len(batch), 7, 480, 125).median(-1).values
    residual = residual - residual.mean(-1, keepdim=True)
    trend_loss = (F.smooth_l1_loss(controls, residual / model.scales[None, :, None], reduction="none") * output_mask[:, :, None]).sum()
    trend_loss = trend_loss / (output_mask.sum().clamp_min(1) * 480)
    curvature = (controls[..., 2:] - 2 * controls[..., 1:-1] + controls[..., :-2]).square().mean()
    return correlation_loss + .2 * trend_loss + .01 * curvature, {"correlation": correlation_loss.item(), "trend": trend_loss.item()}


@torch.no_grad()
def _score(model, items, device):
    model.eval(); values = []
    for item in items:
        correction, _ = model(item["features"][None].to(device))
        p = compose_d12(item["prediction"][:1][None].to(device), item["prediction"][list(OUTPUT_INDICES)][None].to(device) + correction)[0, 1:]
        y = item["target"][1:].to(device)
        p = p - p.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True)
        r = (p * y).sum(-1) / (p.square().sum(-1).clamp_min(1) * y.square().sum(-1).clamp_min(1)).sqrt()
        values.extend(r[item["mask"].to(device)].cpu().tolist())
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-bundle", required=True, help="the SHA-pinned B5-final FM+P bundle; supplies immutable d12 scales")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache, baseline, output = Path(args.cache), Path(args.baseline_evaluation), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    scales = _load_scales(Path(args.base_bundle))
    if scales.shape != (12,): raise ValueError("--base-bundle contains invalid d12 scales")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    items = _load_items(cache, baseline, scales)
    output.mkdir(parents=True)
    config = {"protocol": "B5-final-120s-zero-mean-tcn-v1", "cache": str(cache.resolve()),
              "baseline_evaluation": str(baseline.resolve()), "updates": args.updates, "batch_size": args.batch_size,
              "validate_every": args.validate_every, "seed": args.seed, "base_bundle": str(Path(args.base_bundle).resolve()), "scales_uV": scales.tolist(),
              "loss": "1-r(centered) + .2*zero-mean 4Hz residual + .01*curvature"}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    model = TrendCorrector(scales).to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng, best = np.random.default_rng(args.seed), -float("inf")
    for step in range(args.updates + 1):
        if step % args.validate_every == 0:
            score = _score(model, items["validation"], args.device)
            (output / f"validation_{step:07d}.json").write_text(json.dumps({"step": step, "centered_r": score}, indent=2))
            print(f"[validate] step={step} centered_r={score:.8f}", flush=True)
            if score > best:
                best = score; torch.save({"model": model.state_dict(), "step": step, "score": score, "config": config}, output / "best.pt")
        if step == args.updates: break
        model.train(); chosen = rng.choice(len(items["train"]), size=min(args.batch_size, len(items["train"])), replace=False)
        optimizer.zero_grad(set_to_none=True); loss, detail = _loss(model, [items["train"][i] for i in chosen], args.device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if (step + 1) % 50 == 0: print(f"[train] step={step+1} loss={loss.item():.5f} r={detail['correlation']:.5f} trend={detail['trend']:.5f}", flush=True)
    (output / "complete.json").write_text(json.dumps({"completed": True, "best_centered_r": best, "updates": args.updates}, indent=2))


if __name__ == "__main__": main()
