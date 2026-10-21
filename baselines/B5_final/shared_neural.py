"""Matched nonlinear N1/N2/N3 screen on original B5-final; only chest corrections train."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.B5_best.common import read, write, csv_write, sha

PROTOCOL = "B5-final-shared-private-neural-screen-v1"
DILATIONS = (1, 2, 4, 8, 16, 29)
EPOCHS = 10


class SameConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.radius = dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, x):
        r = self.radius
        return self.conv(torch.cat((x[..., :1].expand(-1, -1, r), x,
                                   x[..., -1:].expand(-1, -1, r)), dim=-1))


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(16, 2 * width)
        self.one, self.two = SameConv(width, dilation), SameConv(width, dilation)

    def forward(self, x, metadata):
        a, b = self.film(metadata).chunk(2, dim=-1)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = h * (1 + .1 * torch.tanh(a)[..., None]) + .1 * b[..., None]
        return x + self.two(F.silu(self.one(F.silu(h))))


class Correction(nn.Module):
    def __init__(self, rank, metadata_dim, basis, width=24):
        super().__init__()
        if rank not in (0, 2, 3):
            raise ValueError("N1/N2/N3 use ranks 0/2/3")
        self.rank, self.width = rank, width
        self.metadata = nn.Sequential(nn.Linear(metadata_dim, 16), nn.SiLU(), nn.Linear(16, 16))
        self.input = nn.Conv1d(3, width, 1)
        self.blocks = nn.ModuleList(Block(width, d) for d in DILATIONS)
        # Six small lead-specific heads (two hidden channels per lead), same in all groups.
        self.private_in = nn.Conv1d(width * 6, 12, 1, groups=6)
        self.private_out = nn.Conv1d(12, 6, 1, groups=6)
        nn.init.zeros_(self.private_out.weight); nn.init.zeros_(self.private_out.bias)
        if rank:
            self.shared_out = nn.Conv1d(width, rank, 1)
            nn.init.zeros_(self.shared_out.weight); nn.init.zeros_(self.shared_out.bias)
            self.mix = nn.Parameter(torch.as_tensor(basis[:, :rank], dtype=torch.float32).clone())
        else:
            self.shared_out = None

    def forward(self, signal, metadata):
        m = self.metadata(metadata)
        x = self.input(signal)
        for block in self.blocks:
            x = block(x, m)
        x = F.silu(x)
        private = self.private_out(F.silu(self.private_in(x.repeat(1, 6, 1))))
        shared = (torch.einsum("lk,bkt->blt", self.mix, self.shared_out(x))
                  if self.rank else torch.zeros_like(private))
        return shared + private, shared, private


def objective(delta, shared, private, error, predicted_slow, target_slow, mask, weights):
    p = predicted_slow + delta
    p = p - p.mean(-1, keepdim=True)
    t = target_slow - target_slow.mean(-1, keepdim=True)
    r = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1)).clamp_min(1e-7)
    valid = mask & (t.square().sum(-1) > 1e-10)
    w = valid.to(delta.dtype) * weights[:, None]
    per_lead = ((delta - error).square().mean(-1) + .1 * (1 - r)
        + .01 * delta.square().mean(-1)
        + .001 * (shared.square().mean(-1) + private.square().mean(-1)))
    return (per_lead * w).sum() / w.sum().clamp_min(1e-7)


def load_data(screen):
    cache = screen / "slow_cache"
    manifest = read(cache / "cache.json")
    if not manifest.get("completed"):
        raise ValueError("Complete the independent direction4 screen first")
    records, arrays = manifest["records"], {}
    train_subjects = {r["subject_id"] for r in records if r["split"] == "train"}
    if train_subjects & {r["subject_id"] for r in records if r["split"] == "validation"}:
        raise ValueError("Patient split leakage")
    scale = np.asarray(manifest["scale_chest"], np.float64)[:, None]
    for row in records:
        if sha(cache / row["file"]) != row["sha256"]:
            raise ValueError("Frozen slow cache changed")
        with np.load(cache / row["file"], allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
        # Center lag of existing I-only lag bank: mean, RMS, MA501; NOT all lagged features.
        signal = a["features"][:, 36:39].T.astype(np.float64)
        metadata = a["features"][0, 75:-1].astype(np.float32)
        target = (a["target_slow"] / scale).reshape(6, 600, 10).mean(-1)
        predicted = (.75 * a["predicted_slow"] / scale).reshape(6, 600, 10).mean(-1)
        error = (a["error"] / scale).reshape(6, 600, 10).mean(-1)
        error -= error.mean(axis=-1, keepdims=True)
        arrays[row["target_record_id"]] = {"signal": signal, "metadata": metadata,
            "target": target.astype(np.float32), "predicted": predicted.astype(np.float32),
            "error": error.astype(np.float32), "mask": a["quality"][6:].astype(bool)}
    train = [r for r in records if r["split"] == "train" and r["valid_chest_leads"] > 0]
    counts = Counter(r["subject_id"] for r in train)
    weights = np.array([1 / counts[r["subject_id"]] for r in train])
    mean = sum(w * arrays[r["target_record_id"]]["signal"].mean(1)
               for r, w in zip(train, weights)) / weights.sum()
    variance = sum(w * ((arrays[r["target_record_id"]]["signal"] - mean[:, None]) ** 2).mean(1)
                   for r, w in zip(train, weights)) / weights.sum()
    std = np.sqrt(np.maximum(variance, 1e-8))
    for a in arrays.values():
        a["signal"] = ((a["signal"] - mean[:, None]) / std[:, None]).astype(np.float32)
    return manifest, arrays, train, counts, mean, std


class Evaluator:
    def __init__(self, manifest, arrays, device):
        from baselines.B5_best_context.evaluate import load_baseline
        source = Path(manifest["source_run"])
        self.loaded = load_baseline(source / "base_inference")
        self.baseline = {"task1": self.loaded[1]["task1"]["prediction"],
            "task2": np.load(source / "study/C0_no_projection_II/task2/prediction_uV.npy", mmap_mode="r")}
        self.manifest, self.arrays, self.device = manifest, arrays, device
        self.expected = next(r["mean_r"] for r in read(source / "study/final.json")["summary"]
                             if r["group"] == "C0_no_projection_II")

    def close(self):
        self.baseline["task2"]._mmap.close()
        for item in self.loaded[1].values():
            for key in ("prediction", "target", "quality"):
                item[key]._mmap.close()

    def run(self, model, output, save_arrays=False):
        from baselines.B5_best_context.evaluate import _score, _record_metrics
        from baselines.B5.scoring import scores_from_summaries
        from ecg12gen.evaluate import write_report
        model.eval()
        rows = [r for r in self.manifest["records"] if r["split"] == "validation"]
        corrections = {}
        shared_energy = private_energy = cross = 0.
        with torch.no_grad():
            for start in range(0, len(rows), 8):
                chunk = rows[start:start + 8]
                x = torch.from_numpy(np.stack([self.arrays[r["target_record_id"]]["signal"] for r in chunk])).to(self.device)
                m = torch.from_numpy(np.stack([self.arrays[r["target_record_id"]]["metadata"] for r in chunk])).to(self.device)
                delta, shared, private = model(x, m)
                d, s, u = [v.cpu().numpy().astype(np.float64) for v in (delta, shared, private)]
                shared_energy += float((s * s).sum()); private_energy += float((u * u).sum())
                cross += float((2 * s * u).sum())
                for i, row in enumerate(chunk):
                    corrections[row["target_record_id"]] = d[i] * np.asarray(self.manifest["scale_chest"])[:, None]
        tasks, metrics = {}, []
        for task, item in self.loaded[1].items():
            prediction = np.array(self.baseline[task], copy=True)
            for ids in item["pairs"]:
                row = item["rows"][ids[0]]
                delta5 = corrections.get(row["target_record_id"])
                if delta5 is not None:
                    delta = np.stack([np.interp(np.arange(60000), np.arange(600) * 100 + 49.5, v) for v in delta5])
                    for j, index in enumerate(ids):
                        prediction[index, 6:] = (self.baseline[task][index, 6:].astype(float)
                            + delta[:, j * 5000:(j + 1) * 5000]).astype(np.float32)
                result = np.concatenate([prediction[i] for i in ids], axis=-1)
                truth = np.concatenate([item["target"][i] for i in ids], axis=-1)
                metrics.append(_record_metrics(result, truth, item["quality"][ids[0]], row, task))
            if not np.array_equal(prediction[:, :6], self.baseline[task][:, :6]):
                raise AssertionError("Chest experiment changed I/limb/context outputs")
            overall, details = _score(prediction, item, task)
            tasks[task] = overall
            write_report(output / task, overall, details, title="Independent direction4 nonlinear correction")
            if save_arrays:
                np.save(output / task / "prediction_uV.npy", prediction, allow_pickle=False)
        score = scores_from_summaries(tasks)
        result = {"task1_r": score["task1_r_missing11"], "task2_r": score["task2_r_missing11"],
            "machine_r": tasks["task2"]["ecg_machine_d6_r_missing11"],
            "body_r": tasks["task2"]["body_scale_d6_r_missing11"], "mean_r": score["main_correlation"],
            "delta_mean_r": score["main_correlation"] - self.expected,
            "shared_energy_fraction": shared_energy / max(shared_energy + private_energy, 1e-30),
            "shared_normalized_energy": shared_energy, "private_normalized_energy": private_energy,
            "cross_term_normalized_energy": cross}
        csv_write(output / "pair_metrics.csv", metrics)
        write(output / "score.json", result)
        return result


def train(args):
    from torch.utils.data import TensorDataset, DataLoader
    from baselines.B5.runtime import seed_all, device_from_name
    screen, out = Path(args.screen_run).resolve(), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Use a new neural training output")
    manifest, arrays, train_rows, counts, mean, std = load_data(screen)
    basis = np.asarray(read(screen / "pca.json")["basis_error"], np.float32)
    values = []
    for r in train_rows:
        a = arrays[r["target_record_id"]]
        values.append((a["signal"], a["metadata"], a["error"], a["predicted"], a["target"],
                       a["mask"], np.float32(1 / counts[r["subject_id"]])))
    data = TensorDataset(*[torch.from_numpy(np.stack([v[k] for v in values]).copy()) for k in range(7)])
    seed_all(42, True)
    device = device_from_name(args.device)
    model = Correction(args.rank, len(values[0][1]), basis).to(device)
    loader = DataLoader(data, batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(42), num_workers=0)
    out.mkdir(parents=True)
    config = {"protocol": PROTOCOL, "rank": args.rank, "seed": 42, "epochs": EPOCHS,
        "learning_rate": 2e-4, "batch_size": 8, "width": 24, "private_hidden_channels_per_lead": 2,
        "dilations": list(DILATIONS), "receptive_field_samples_5Hz": 241,
        "input_range_including_statistics_seconds_at_most": 49.2,
        "parameters": sum(p.numel() for p in model.parameters()),
        "signal_mean_train_only": mean.tolist(), "signal_std_train_only": std.tolist(),
        "metadata_dim": len(values[0][1]), "chest_scales": manifest["scale_chest"],
        "slow_cache_sha256": sha(screen / "slow_cache/cache.json"), "pca_sha256": sha(screen / "pca.json"),
        "baseline": "Original B5-final C0; direction2 NOT applied", "FM_updates": 0, "P_updates": 0,
        "selection": "full120s mean_r including unchanged epoch0; same budget for N1/N2/N3",
        "loss": "QC masked normalized slow-error MSE + .1 slow PCC + .01 correction L2 + .001 branch L2",
        "update_rule": "Add final-wave chest correction directly; do not multiply correction by beta again",
        "screen_run": str(screen)}
    write(out / "config.json", config)
    evaluator = Evaluator(manifest, arrays, device)
    try:
        initial = evaluator.run(model, out / "validation/epoch_000")
        if abs(initial["delta_mean_r"]) > 1e-10:
            raise ValueError("Zero initialization failed to reproduce original B5-final")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
        best, best_epoch, updates = initial["mean_r"], 0, 0
        def save(name, epoch):
            path = out / name
            temp = path.with_suffix(".pending")
            torch.save({"protocol": PROTOCOL, "config": config, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "updates": updates,
                "best_epoch": best_epoch, "best_mean_r": best}, temp)
            temp.replace(path)
        save("best.pt", 0)
        curve = [{"epoch": 0, "updates": 0, "train_loss": None, **initial,
            "best_mean_r": best, "best_epoch": 0}]
        print({"phase": "neural", "rank": args.rank, **curve[0]}, flush=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total, count = 0., 0
            for batch in loader:
                x = [v.to(device) for v in batch]
                delta, shared, private = model(x[0], x[1])
                loss = objective(delta, shared, private, x[2], x[3], x[4], x[5], x[6])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite correction loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                total += float(loss.detach()) * len(delta); count += len(delta)
            result = evaluator.run(model, out / f"validation/epoch_{epoch:03d}")
            if result["mean_r"] > best:
                best, best_epoch = result["mean_r"], epoch
                save("best.pt", epoch)
            save("last.pt", epoch)
            row = {"epoch": epoch, "updates": updates, "train_loss": total / count, **result,
                "best_mean_r": best, "best_epoch": best_epoch}
            curve.append(row); csv_write(out / "learning_curve.csv", curve)
            print({"phase": "neural", "rank": args.rank, **row}, flush=True)
        for name in ("best", "last"):
            payload = torch.load(out / (name + ".pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            evaluator.run(model, out / "final" / name, save_arrays=True)
        write(out / "completed.json", {"completed": True, "rank": args.rank,
            "best_epoch": best_epoch, "best_mean_r": best, "last_mean_r": curve[-1]["mean_r"],
            "baseline_mean_r": initial["mean_r"], "best_gain": best - initial["mean_r"],
            "last_gain": curve[-1]["delta_mean_r"], "parameters": config["parameters"],
            "updates": updates, "FM_updates": 0, "P_updates": 0,
            "best_checkpoint_sha256": sha(out / "best.pt")})
    finally:
        evaluator.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, choices=(0, 2, 3))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.rank is not None:
        train(args)
    else:
        out = Path(args.output).resolve()
        if out.exists():
            raise FileExistsError("Use new N1/N2/N3 pipeline output")
        out.mkdir(parents=True)
        groups = (("N1_independent", 0), ("N2_shared2_private", 2), ("N3_shared3_private", 3))
        for name, rank in groups:
            subprocess.run([sys.executable, "-u", "-m", "baselines.B5_final.shared_neural",
                "--screen-run", args.screen_run, "--output", str(out / name),
                "--rank", str(rank), "--device", args.device], check=True)
        results = [{"group": name, **read(out / name / "completed.json")} for name, _ in groups]
        csv_write(out / "summary.csv", [{k: r[k] for k in ("group", "rank", "parameters", "best_epoch",
            "baseline_mean_r", "best_mean_r", "best_gain", "last_mean_r", "last_gain")} for r in results])
        write(out / "completed.json", {"completed": True, "groups": results,
            "baseline": "Original B5-final, independent from direction2", "phase": "small-budget candidate screening"})


if __name__ == "__main__":
    main()
