"""Train only a small II corrector on immutable B5-best predictions, with matched controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from baselines.B5_best.common import write, sha
from .body_features import BODY_PROTOCOL, encode_windows, restore_ii
from .body_prepare import load_cache


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else None


def target_macro(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["r_II"] is not None:
            grouped[row["target_record_id"]].append(row["r_II"])
    if not grouped:
        raise ValueError("No defined QC-eligible validation II correlations")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def run(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from baselines.B5.runtime import device_from_name, seed_all
    from baselines.B5_best.common import state_sha
    from .body_model import BodyIIAdapter, BodyAdapter
    root, output = Path(args.cache).resolve(), Path(args.output).resolve()
    cache = load_cache(root)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Use positive epochs/batch-size/learning-rate and a uint32 training seed")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new group output, or --resume; tee logs must be adjacent")
    rows = {s: [r for r in cache["records"] if r["split"] == s and r["I_II_eligible"]]
            for s in ("train", "validation")}
    arrays = {}
    for row in rows["train"] + rows["validation"]:
        with np.load(root / row["file"], allow_pickle=False) as record:
            arrays[row["file"]] = {k: record[k].copy() for k in record.files}
    class Windows(Dataset):
        def __init__(self):
            self.windows = []
            # Repeated pairs share a target's total training weight equally.
            counts = defaultdict(int)
            for r in rows["train"]:
                counts[r["target_record_id"]] += 1
            for row in rows["train"]:
                a = arrays[row["file"]]
                signals, bases, scales = encode_windows(a["anchor"], a["base_ii"])
                for j in range(12):
                    if scales[j] == 0:
                        continue
                    truth = a["target_ii"].reshape(12, 5000)[j]
                    if np.std(truth.astype(float)) == 0:
                        continue
                    self.windows.append((signals[j], bases[j],
                        ((truth - np.median(truth)) / scales[j]).astype(np.float32),
                        a["beat"], a["stats"], np.float32(a["available"] and args.group == "context"),
                        np.float32(1 / counts[row["target_record_id"]])))
        def __len__(self):
            return len(self.windows)
        def __getitem__(self, i):
            return self.windows[i]
    data = Windows()
    if not len(data) or not rows["validation"]:
        raise ValueError("No eligible training windows or validation records")
    seed_all(args.seed, True)
    device = device_from_name("cpu" if args.check_only else args.device)
    model = BodyIIAdapter().to(device)
    initial_hash = state_sha(model.state_dict())
    output.mkdir(parents=True, exist_ok=True)
    config = {"protocol": BODY_PROTOCOL, "group": args.group, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "cache_sha256": sha(root / "cache.json"), "initial_weights_sha256": initial_hash,
        "base_checkpoint_sha256": cache["frozen"]["base_checkpoint_sha256"],
        "selection": "full120s_II_target_macro_QC_r",
        "loss": "window_PCC + 0.01 * normalized_II_residual_squared",
        "FM_frozen": True, "slow_CNN_frozen": True,
        "training_windows": len(data), "train_pairs": len(rows["train"]), "validation_pairs": len(rows["validation"])}
    if args.resume:
        from baselines.B5_best.common import read
        if read(output / "config.json") != config:
            raise ValueError("Resume changed paired training settings/cache; start a new run")
    else:
        write(output / "config.json", config)
    if args.check_only:
        x = next(iter(DataLoader(data, batch_size=2, shuffle=False)))
        with torch.no_grad():
            value = model(x[0], x[1], x[3], x[4], x[5])
        if not torch.equal(value, x[1]):
            raise AssertionError("Zero-initialized II correction must exactly preserve B5-best")
        adapter = BodyAdapter.__new__(BodyAdapter)
        adapter.model, adapter.device, adapter.mode = model, device, "correct"
        adapter.payload = {"group": args.group}
        first = arrays[rows["validation"][0]["file"]]
        restored, _, _ = adapter.predict_ii_features(first["anchor"], first["base_ii"],
            first["beat"], first["stats"], first["available"])
        if not np.array_equal(restored, first["base_ii"]):
            raise AssertionError("Visible-cache inference must preserve base II at zero initialization")
        # Exercise nonzero correction, gradients, and the hard residual bound on CPU.
        with torch.no_grad():
            model.head.bias.fill_(1.)
        value = model(x[0], x[1], x[3], x[4], x[5])
        if (value - x[1]).abs().max().item() > .35001:
            raise AssertionError("II residual bound violated")
        value.mean().backward()
        if model.head.bias.grad is None or not torch.isfinite(model.head.bias.grad).all():
            raise AssertionError("Missing/invalid adapter gradient")
        write(output / "preflight.json", {**config, "ready": True, "training_started": False,
            "optimizer_updates": 0, "zero_init_exact": True, "residual_bound_checked": True,
            "CPU_backward_checked": True, "visible_feature_inference_checked": True})
        print({"ready": True, "training_started": False, "initial_weights_sha256": initial_hash}, flush=True)
        return
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)

    @torch.no_grad()
    def assess():
        model.eval()
        details = []
        for row in rows["validation"]:
            a = arrays[row["file"]]
            signals, bases, scales = encode_windows(a["anchor"], a["base_ii"])
            args_ = (signals, bases, np.repeat(a["beat"][None], 12, axis=0),
                np.repeat(a["stats"][None], 12, axis=0),
                np.full(12, a["available"] and args.group == "context", np.float32))
            output_ = model(*(torch.from_numpy(np.asarray(x, np.float32).copy()).to(device) for x in args_))
            predicted = restore_ii(a["base_ii"], bases, output_.cpu().numpy(), scales)
            details.append({"pair_id": row["pair_id"], "target_record_id": row["target_record_id"],
                "r_II": pearson(predicted, a["target_ii"])})
        return target_macro(details), details

    start, updates, best, best_epoch = 0, 0, -float("inf"), 0
    def payload(epoch):
        return {"protocol": BODY_PROTOCOL, "model": model.state_dict(), "group": args.group,
            "training_seed": args.seed,
            "epoch": epoch, "updates": updates, "best_validation_II_r": best, "best_epoch": best_epoch,
            "base_checkpoint_sha256": cache["frozen"]["base_checkpoint_sha256"],
            "sampling": cache["frozen"]["sampling"], "slow_beta": cache["frozen"]["slow_beta"],
            "cache_sha256": config["cache_sha256"], "initial_weights_sha256": initial_hash,
            "optimizer": optimizer.state_dict(), "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
    def save_checkpoint(name, epoch):
        dest = output / name
        temp = dest.with_suffix(".pending")
        torch.save(payload(epoch), temp)
        temp.replace(dest)
    if args.resume and (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if (saved.get("protocol") != BODY_PROTOCOL or saved.get("group") != args.group
                or saved.get("training_seed", 42) != args.seed
                or saved.get("cache_sha256") != config["cache_sha256"]
                or saved.get("initial_weights_sha256") != initial_hash):
            raise ValueError("Resume checkpoint does not belong to this paired body run")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["loader_rng"]); torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, updates, best, best_epoch = saved["epoch"], saved["updates"], saved["best_validation_II_r"], saved["best_epoch"]
    else:
        if args.resume and (output / "validation").exists():
            raise ValueError("Resume needs last.pt after training has begun; do not restart over prior results")
        best, _ = assess()  # Retain the unchanged epoch0 model if continuation is harmful.
        save_checkpoint("best.pt", 0)
        save_checkpoint("last.pt", 0)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for lead_i, base, truth, beat, stats, available, weight in loader:
            x = [a.to(device) for a in (lead_i, base, truth, beat, stats, available, weight)]
            prediction = model(x[0], x[1], x[3], x[4], x[5])
            p, t = prediction - prediction.mean(1, keepdim=True), x[2] - x[2].mean(1, keepdim=True)
            r = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-7)
            loss = ((1 - r + .01 * (prediction - x[1]).square().mean(1)) * x[6]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite body loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1
            total += float(loss.detach()) * len(lead_i); count += len(lead_i)
        metric, details = assess()
        improved = metric > best
        if improved:
            best, best_epoch = metric, epoch
            save_checkpoint("best.pt", epoch)
        save_checkpoint("last.pt", epoch)
        row = {"epoch": epoch, "group": args.group, "train_loss": total / count,
            "validation_full120s_II_r": metric, "best_II_r": best, "best_epoch": best_epoch, "updates": updates}
        write(output / f"validation/epoch_{epoch:03d}.json", {**row, "pairs": details})
        print(row, flush=True)
    write(output / "completed.json", {"completed": True, "group": args.group, "epochs": args.epochs,
        "training_seed": args.seed,
        "best_epoch": best_epoch, "best_II_r": best, "FM_updates": 0, "slow_CNN_updates": 0,
        "adapter_updates": updates, "best_checkpoint_sha256": sha(output / "best.pt")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group", required=True, choices=("no_context", "context"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42, help="II adapter initialization/shuffle seed; B5-best inference noise stays seed42")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true", help="CPU forward/backward only; no optimizer updates")
    args = p.parse_args()
    if args.check_only and args.resume:
        p.error("Use a separate empty output for --check-only")
    run(args)


if __name__ == "__main__":
    main()
