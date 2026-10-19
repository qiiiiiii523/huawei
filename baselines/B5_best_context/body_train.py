"""Train only a small II corrector on immutable B5-best predictions, with matched controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from baselines.B5_best.common import write, sha
from .body_features import BODY_PROTOCOL, encode_windows, restore_ii, feature_encoding, LEGACY_ENCODING
from .body_prepare import load_cache
from .body_history import history_policy


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
    from .body_gate_features import (GATE_PROTOCOL, visible_record_features, fit_normalization, normalize_features)
    root, output = Path(args.cache).resolve(), Path(args.output).resolve()
    cache = load_cache(root)
    encoding = feature_encoding(cache)
    policy = history_policy(cache)
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
    gate_features, normalization = {}, None
    gated = args.record_gate is not None
    adapt_checkpoint = getattr(args, "adapt_body_checkpoint", None)
    if adapt_checkpoint and (gated or args.init_body_checkpoint or args.group != "context"):
        raise ValueError("Body adaptation is separate from record-gate/no-context experiments")
    if gated:
        if encoding != LEGACY_ENCODING:
            raise ValueError("Record-gate C/G uses the original legacy cache; do not combine experiments")
        if args.group != "context" or not args.init_body_checkpoint:
            raise ValueError("Record C/G experiment requires group=context and --init-body-checkpoint")
        for row in rows["train"] + rows["validation"]:
            a = arrays[row["file"]]
            gate_features[row["file"]] = visible_record_features(a["anchor"], a["base_ii"], a["beat"], a["stats"], a["available"])
        # One sample per physical training target, never validation or hidden II.
        training_features = {}
        for row in rows["train"]:
            training_features.setdefault(row["target_record_id"], gate_features[row["file"]])
        normalization = fit_normalization(list(training_features.values()))
        gate_features = {key: normalize_features(value, normalization) for key, value in gate_features.items()}
    elif args.init_body_checkpoint:
        raise ValueError("Use --record-gate C/G with this paired continuation entry")
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
                    values = (signals[j], bases[j],
                        ((truth - np.median(truth)) / scales[j]).astype(np.float32),
                        a["beat"], a["stats"], np.float32(a["available"] and args.group == "context"),
                        np.float32(1 / counts[row["target_record_id"]]))
                    self.windows.append(values + (gate_features[row["file"]],) if gated else values)
        def __len__(self):
            return len(self.windows)
        def __getitem__(self, i):
            return self.windows[i]
    data = Windows()
    if not len(data) or not rows["validation"]:
        raise ValueError("No eligible training windows or validation records")
    seed_all(args.seed, True)
    device = device_from_name("cpu" if args.check_only else args.device)
    source_sha = source_state_sha = None
    if gated:
        from .body_model import load_body_checkpoint
        from .body_gate_model import RecordGatedIIAdapter
        source_model, source_payload = load_body_checkpoint(args.init_body_checkpoint,
            cache["frozen"]["base_checkpoint_sha256"], device, expected_group="context")
        if source_payload.get("protocol") != BODY_PROTOCOL or source_payload.get("cache_sha256") != sha(root / "cache.json"):
            raise ValueError("Initialize C/G from an original body context checkpoint aligned to this cache")
        model = RecordGatedIIAdapter(args.record_gate).to(device)
        model.body.load_state_dict(source_model.state_dict(), strict=True)
        source_sha, source_state_sha = sha(args.init_body_checkpoint), state_sha(source_model.state_dict())
    else:
        model = BodyIIAdapter().to(device)
    adaptation = None
    if adapt_checkpoint:
        from .body_model import load_body_checkpoint
        from .body_adapt_prepare import check_source
        from .selected_study import SELECTED_B43_SHA256
        if sha(adapt_checkpoint) != SELECTED_B43_SHA256:
            raise ValueError("Initialize adaptation from selected B43 seed43 best15")
        source_model, source_payload = load_body_checkpoint(adapt_checkpoint,
            cache["body_adaptation"]["source_base_sha256"], device, expected_group="context")
        adaptation = check_source(source_payload, cache)
        model.load_state_dict(source_model.state_dict(), strict=True)
        source_sha, source_state_sha = sha(adapt_checkpoint), state_sha(source_model.state_dict())
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
    if encoding != LEGACY_ENCODING:
        config["feature_encoding"] = encoding
    if "history_preprocessing" in cache:
        config["history_preprocessing"] = policy
    if adaptation is not None:
        config["body_adaptation"] = adaptation
    if gated:
        config.update(protocol=GATE_PROTOCOL, record_gate=args.record_gate,
            init_body_checkpoint_sha256=source_sha, initial_body_weights_sha256=source_state_sha,
            gate_normalization=normalization, gate_initial_alpha=1., gate_shared_across_12_windows=True)
    if args.resume:
        from baselines.B5_best.common import read
        if read(output / "config.json") != config:
            raise ValueError("Resume changed paired training settings/cache; start a new run")
    else:
        write(output / "config.json", config)
    if args.check_only:
        x = next(iter(DataLoader(data, batch_size=2, shuffle=False)))
        if adaptation is not None:
            with torch.no_grad():
                value = model(x[0], x[1], x[3], x[4], x[5])
                original = source_model(x[0], x[1], x[3], x[4], x[5])
            if not torch.equal(value, original) or state_sha(model.state_dict()) != source_state_sha:
                raise AssertionError("Adaptation initialization changed selected B43 output/weights")
            value = model(x[0], x[1], x[3], x[4], x[5])
            if (value-x[1]).abs().max().item() > .35001:
                raise AssertionError("Loaded body residual bound violated")
            (value-x[2]).square().mean().backward()
            if model.head.bias.grad is None or not torch.isfinite(model.head.bias.grad).all():
                raise AssertionError("Missing/nonfinite adaptation backward gradient")
            write(output/"preflight.json", {**config,"ready":True,"training_started":False,"optimizer_updates":0,
                "source_output_exactly_preserved":True,"CPU_backward_checked":True,"residual_bound_checked":True})
            print({"ready":True,"body_adaptation":True,"optimizer_updates":0},flush=True)
            return
        if gated:
            with torch.no_grad():
                original = model.body(x[0], x[1], x[3], x[4], x[5])
                value = model(x[0], x[1], x[3], x[4], x[5], x[7])
                alpha = model.alpha(x[7])
            if not torch.equal(value, original) or not torch.equal(alpha, torch.ones_like(alpha)):
                raise AssertionError("C/G must exactly preserve the loaded body output at initialization")
            value = model(x[0], x[1], x[3], x[4], x[5], x[7])
            ((value - x[1]).square().mean() + .01 * model.alpha(x[7]).mean()).backward()
            if args.record_gate == "G":
                gradient = model.gate[-1].bias.grad
                if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum():
                    raise AssertionError("Gate boundary initialization has no attenuation gradient")
            else:
                if any(p.requires_grad or p.grad is not None for p in model.gate.parameters()):
                    raise AssertionError("Control gate must remain fixed")
            repeated = torch.from_numpy(np.repeat(x[7][:1].numpy(), 12, axis=0))
            with torch.no_grad():
                values = model.alpha(repeated)
            if values.min() < 0 or values.max() > 1 or not torch.equal(values, values[:1].expand_as(values)):
                raise AssertionError("A record must have one bounded alpha shared by twelve windows")
            write(output / "preflight.json", {**config, "ready": True, "training_started": False,
                "optimizer_updates": 0, "source_output_exactly_preserved": True,
                "gate_backward_checked": args.record_gate == "G", "one_alpha_per_record_checked": True})
            print({"ready": True, "record_gate": args.record_gate, "optimizer_updates": 0}, flush=True)
            return
        with torch.no_grad():
            value = model(x[0], x[1], x[3], x[4], x[5])
        if not torch.equal(value, x[1]):
            raise AssertionError("Zero-initialized II correction must exactly preserve B5-best")
        adapter = BodyAdapter.__new__(BodyAdapter)
        adapter.model, adapter.device, adapter.mode = model, device, "correct"
        adapter.payload = {"group": args.group, "feature_encoding": encoding}
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
            if gated:
                args_ += (np.repeat(gate_features[row["file"]][None], 12, axis=0),)
            output_ = model(*(torch.from_numpy(np.asarray(x, np.float32).copy()).to(device) for x in args_))
            predicted = restore_ii(a["base_ii"], bases, output_.cpu().numpy(), scales)
            details.append({"pair_id": row["pair_id"], "target_record_id": row["target_record_id"],
                "r_II": pearson(predicted, a["target_ii"])})
        return target_macro(details), details

    start, updates, best, best_epoch = 0, 0, -float("inf"), 0
    def payload(epoch):
        value = {"protocol": GATE_PROTOCOL if gated else BODY_PROTOCOL, "model": model.state_dict(), "group": args.group,
            "training_seed": args.seed,
            "epoch": epoch, "updates": updates, "best_validation_II_r": best, "best_epoch": best_epoch,
            "base_checkpoint_sha256": cache["frozen"]["base_checkpoint_sha256"],
            "sampling": cache["frozen"]["sampling"], "slow_beta": cache["frozen"]["slow_beta"],
            "cache_sha256": config["cache_sha256"], "initial_weights_sha256": initial_hash,
            "optimizer": optimizer.state_dict(), "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        if encoding != LEGACY_ENCODING:
            value["feature_encoding"] = encoding
        if "history_preprocessing" in cache:
            value["history_preprocessing"] = policy
        if adaptation is not None:
            value["body_adaptation"] = adaptation
        if gated:
            value.update(record_gate=args.record_gate, gate_normalization=normalization,
                init_body_checkpoint_sha256=source_sha, initial_body_weights_sha256=source_state_sha)
        return value
    def save_checkpoint(name, epoch):
        dest = output / name
        temp = dest.with_suffix(".pending")
        torch.save(payload(epoch), temp)
        temp.replace(dest)
    if args.resume and (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if (saved.get("protocol") != config["protocol"] or saved.get("group") != args.group
                or saved.get("record_gate") != args.record_gate
                or saved.get("training_seed", 42) != args.seed
                or saved.get("cache_sha256") != config["cache_sha256"]
                or saved.get("initial_weights_sha256") != initial_hash):
            raise ValueError("Resume checkpoint does not belong to this paired body run")
        if feature_encoding(saved) != encoding:
            raise ValueError("Resume checkpoint uses a different feature encoding")
        if saved.get("body_adaptation") != adaptation:
            raise ValueError("Resume changed adaptation source/base/optimizer initialization")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["loader_rng"]); torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, updates, best, best_epoch = saved["epoch"], saved["updates"], saved["best_validation_II_r"], saved["best_epoch"]
    else:
        if args.resume and (output / "validation").exists():
            raise ValueError("Resume needs last.pt after training has begun; do not restart over prior results")
        best, initial_details = assess()  # Retain the unchanged epoch0 model if continuation is harmful.
        if adaptation is not None:
            write(output/"validation/epoch_000.json", {"epoch":0,"group":args.group,"updates":0,
                "train_loss":None,"validation_full120s_II_r":best,"best_II_r":best,"best_epoch":0,
                "pairs":initial_details,"source_checkpoint_sha256":source_sha})
        save_checkpoint("best.pt", 0)
        save_checkpoint("last.pt", 0)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in loader:
            x = [a.to(device) for a in batch]
            forward = (x[0], x[1], x[3], x[4], x[5]) + ((x[7],) if gated else ())
            prediction = model(*forward)
            p, t = prediction - prediction.mean(1, keepdim=True), x[2] - x[2].mean(1, keepdim=True)
            r = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-7)
            loss = ((1 - r + .01 * (prediction - x[1]).square().mean(1)) * x[6]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite body loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1
            total += float(loss.detach()) * len(x[0]); count += len(x[0])
        metric, details = assess()
        improved = metric > best
        if improved:
            best, best_epoch = metric, epoch
            save_checkpoint("best.pt", epoch)
        save_checkpoint("last.pt", epoch)
        row = {"epoch": epoch, "group": args.group, "train_loss": total / count,
            "validation_full120s_II_r": metric, "best_II_r": best, "best_epoch": best_epoch, "updates": updates}
        if gated:
            row["record_gate"] = args.record_gate
        write(output / f"validation/epoch_{epoch:03d}.json", {**row, "pairs": details})
        print(row, flush=True)
    write(output / "completed.json", {"completed": True, "group": args.group, "epochs": args.epochs,
        "record_gate": args.record_gate,
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
    p.add_argument("--record-gate", choices=("C", "G"), help="Optional paired continuation: C=fixed alpha1, G=learned record attenuation")
    p.add_argument("--init-body-checkpoint", help="Same existing context/best.pt for both C and G; never a B5 FM checkpoint")
    p.add_argument("--adapt-body-checkpoint", help="Explicit selected B43-best15 warm start on a new-base B cache; resets optimizer and retains epoch0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true", help="CPU forward/backward only; no optimizer updates")
    args = p.parse_args()
    if args.check_only and args.resume:
        p.error("Use a separate empty output for --check-only")
    run(args)


if __name__ == "__main__":
    main()
