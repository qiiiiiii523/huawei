"""Train an external body-scale relation adapter on frozen B5 lead-II predictions.

Train-time target II is a label only. Inference inputs are target I, historical
body-scale six leads, frozen B5 II, and the frozen B5's usual metadata. The body
record is never passed through B5. Runs both matched-context and no-context
trainings with identical architecture and compares disabled/shuffled context.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from baselines.B5.context_body_lead_relation import extract_lead_pair_beats
from baselines.B5.objective import file_sha256
from scripts.train_b5_body_context import BodySubset, ROOT


BASE = ROOT / "results/context_body/diagnostics"
B5 = ROOT / "results/B5/context_start/best.pt"
OUTPUT = ROOT / "results/context_body/body_relation_b5_adapter_v1"
SEEDS = (101, 202, 303)


def body_mask(sample):
    return (sample.context.time_mask & sample.context.window_mask[:, None, None] &
            sample.input_quality_mask[None, :, None])


def normalize_i(lead_i, valid):
    if valid.sum() < 1000:
        raise ValueError("Insufficient lead-I coverage")
    centered = np.where(valid, lead_i - np.median(lead_i[valid]), 0).astype(np.float32)
    scale = float(np.sqrt(np.mean(np.square(centered[valid]))))
    if not np.isfinite(scale) or scale < 1e-4:
        raise ValueError("Degenerate lead-I amplitude")
    return np.clip(centered / scale, -8, 8), scale


def pearson_per_record(predicted, truth, indices):
    values = []
    for group in indices:
        p = predicted[group].reshape(-1).astype(np.float64)
        t = truth[group].reshape(-1).astype(np.float64)
        p, t = p - p.mean(), t - t.mean()
        values.append(float(np.dot(p, t) / max(np.linalg.norm(p) * np.linalg.norm(t), 1e-9)))
    return np.asarray(values)


def relation_features(item):
    raw = np.asarray(item.context.raw_uV, dtype=np.float64)
    mask = body_mask(item)
    beats, present = extract_lead_pair_beats(raw, mask)
    if present.sum() < 4:
        return np.zeros((2, 300), np.float32), np.zeros(27, np.float32), False
    beat = np.median(beats[present, :2], axis=0).astype(np.float32)
    beat -= np.median(beat, axis=1, keepdims=True)
    vectors = []
    for window in range(len(raw)):
        valid = mask[window].all(axis=0)
        if valid.sum() < 1000:
            continue
        x = raw[window][:, valid]
        x = x - np.median(x, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(x[0] ** 2))
        if scale < 1e-4:
            continue
        x = np.clip(x / scale, -8, 8)
        cov = (x @ x.T) / len(x[0])
        rms = np.sqrt(np.diag(cov) + 1e-8)
        corr = cov / np.outer(rms, rms)
        vectors.append(np.r_[corr[np.triu_indices(6)], np.log(np.clip(rms, 1e-4, 1e4))])
    if not vectors:
        return np.zeros((2, 300), np.float32), np.zeros(27, np.float32), False
    return beat, np.median(vectors, axis=0).astype(np.float32), True


class FixedCache(Dataset):
    def __init__(self, split):
        self.source = BodySubset(split, "A_raw_window")
        folder = BASE / ("base_train" if split == "train" else "base")
        with (folder / "run_config.json").open(encoding="utf-8") as f:
            self.config = json.load(f)
        if self.config["source_checkpoint_sha256"] != file_sha256(B5):
            raise ValueError("Frozen B5 cache checkpoint mismatch")
        self.pred = np.load(folder / "prediction_uV.npy", mmap_mode="r")
        self.truth = np.load(folder / "target_uV.npy", mmap_mode="r")
        with (folder / "window_metadata.csv").open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if len(self.source) != len(self.rows) or self.pred.shape != self.truth.shape:
            raise ValueError("B5 cache roster mismatch")
        if self.pred.shape != (len(self.rows), 12, 5000):
            raise ValueError("Unexpected B5 cache shape")
        self.support = {}
        for index in range(len(self.source)):
            item = self.source[index]
            row = self.rows[index]
            if (item.pair_id != row["pair_id"] or
                item.target_record_id != row["target_record_id"] or
                str(item.meta["start_sample_500hz"]) != row["start_sample_500hz"] or
                not np.allclose(item.Y_12lead[1], self.truth[index, 1], rtol=1e-5, atol=1e-3)):
                raise ValueError(f"B5 cache/sample mismatch at {index}")
            if item.pair_id not in self.support:
                self.support[item.pair_id] = relation_features(item)
        self.ids = sorted(self.support)
        if split == "validation" and (len(self.ids) != 25 or len(self.rows) != 300):
            raise ValueError("Validation roster is not the fixed 25-patient set")
        self.id_index = {key: idx for idx, key in enumerate(self.ids)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.source[index]
        lead_i, scale = normalize_i(item.anchor_i_ecg[0], np.ones(5000, bool))
        base = np.asarray(self.pred[index, 1], dtype=np.float32)
        truth = np.asarray(self.truth[index, 1], dtype=np.float32)
        beat, stats, available = self.support[item.pair_id]
        return (torch.from_numpy(lead_i.copy()),
                torch.from_numpy(np.clip((base - np.median(base)) / scale, -12, 12).copy()),
                torch.from_numpy(((truth - np.median(truth)) / scale).copy()),
                torch.from_numpy(beat.copy()), torch.from_numpy(stats.copy()),
                torch.tensor(float(available)), self.id_index[item.pair_id])

    def close(self):
        self.source.close()


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.beat = nn.Sequential(nn.Conv1d(2, 24, 9, padding=4), nn.SiLU(),
                                  nn.Conv1d(24, 24, 9, stride=2, padding=4), nn.SiLU(),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.context = nn.Sequential(nn.Linear(24 + 27, 64), nn.SiLU(), nn.Linear(64, 64))
        self.signal = nn.Sequential(nn.Conv1d(2, 32, 15, padding=7), nn.SiLU(),
                                    nn.Conv1d(32, 32, 9, padding=4), nn.SiLU(),
                                    nn.Conv1d(32, 32, 9, padding=8, dilation=2), nn.SiLU())
        self.film = nn.Linear(64, 64)
        self.head = nn.Conv1d(32, 1, 7, padding=3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, lead_i, base, beat, stats, enabled):
        features = self.signal(torch.stack((lead_i, base), dim=1))
        context = self.context(torch.cat((self.beat(beat), stats), dim=1))
        context = context * enabled[:, None]
        gamma, beta = self.film(context).chunk(2, dim=1)
        modified = features * (1 + 0.15 * torch.tanh(gamma)[..., None]) + 0.15 * beta[..., None]
        correction = 0.35 * torch.tanh(self.head(torch.nn.functional.silu(modified))[:, 0])
        return base + correction


def correlation(predicted, truth):
    p = predicted - predicted.mean(dim=1, keepdim=True)
    t = truth - truth.mean(dim=1, keepdim=True)
    return (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-7)


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        indices = rng.permutation(n)
        if np.all(indices != np.arange(n)):
            return indices
    raise ValueError("Could not derange patients")


@torch.no_grad()
def assess(model, data, device, mode, seed=101):
    model.eval()
    mapping = derangement(len(data.ids), seed) if mode == "shuffled" else np.arange(len(data.ids))
    predictions = []
    for batch in DataLoader(data, batch_size=16, shuffle=False, num_workers=0):
        lead_i, base, _, beat, stats, available, ids = batch
        if mode == "shuffled":
            donor = [data.support[data.ids[mapping[int(i)]]] for i in ids]
            beat = torch.from_numpy(np.stack([entry[0] for entry in donor]))
            stats = torch.from_numpy(np.stack([entry[1] for entry in donor]))
            available = torch.tensor([float(entry[2]) for entry in donor])
        if mode == "disabled":
            available = torch.zeros_like(available)
        prediction = model(lead_i.to(device), base.to(device), beat.to(device),
                           stats.to(device), available.to(device))
        predictions.append(prediction.cpu().numpy())
    normalized = np.concatenate(predictions)
    # Correlation is invariant to per-record positive I scaling, but each window's
    # centering/scale differs. Restore microvolts before concatenating 12 windows.
    microvolts = []
    for index, output in enumerate(normalized):
        item = data.source[index]
        _, scale = normalize_i(item.anchor_i_ecg[0], np.ones(5000, bool))
        base = np.asarray(data.pred[index, 1], dtype=np.float32)
        microvolts.append(base + scale * (output - np.clip((base - np.median(base)) / scale, -12, 12)))
    groups = {}
    for index, row in enumerate(data.rows):
        groups.setdefault(row["pair_id"], []).append(index)
    if any(len(indices) != 12 for indices in groups.values()):
        raise ValueError("Expected 12 windows per record")
    truth = np.asarray(data.truth[:, 1])
    scores = pearson_per_record(np.asarray(microvolts), truth, list(groups.values()))
    baseline = pearson_per_record(np.asarray(data.pred[:, 1]), truth, list(groups.values()))
    return {"mean_r": float(scores.mean()), "delta_vs_B5": float((scores - baseline).mean()),
            "patients_improved": int((scores > baseline).sum()),
            "baseline_mean_r": float(baseline.mean()), "per_patient": dict(zip(groups, scores.tolist()))}


def run(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    train, val = FixedCache("train"), FixedCache("validation")
    try:
        if set(train.ids) & set(val.ids):
            raise ValueError("Patient leakage between splits")
        probe = train[0]
        model = Adapter().to(device)
        out = model(*(x[None].to(device) for x in (probe[0], probe[1], probe[3], probe[4], probe[5])))
        if out.shape != (1, 5000) or not torch.isfinite(out).all():
            raise ValueError("Adapter preflight failed")
        print(json.dumps({"preflight": "passed", "train_windows": len(train),
                          "validation_windows": len(val), "train_patients": len(train.ids),
                          "validation_patients": len(val.ids)}), flush=True)
        if args.preflight:
            return
        OUTPUT.mkdir(parents=True, exist_ok=False)
        (OUTPUT / "config.json").write_text(json.dumps({"epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.lr, "seed": 42,
            "selection": "validation mean r II", "B5_sha256": file_sha256(B5),
            "target_II_role": "train label, validation score only",
            "test_inputs": "target I + historical body six leads + frozen B5 output"}, indent=2))
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        summary = {}
        for kind in ("no_context", "context"):
            torch.manual_seed(42)
            model = Adapter().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            best = -float("inf")
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for lead_i, base, truth, beat, stats, available, _ in loader:
                    lead_i, base, truth, beat, stats, available = (
                        x.to(device) for x in (lead_i, base, truth, beat, stats, available))
                    if kind == "no_context":
                        available = torch.zeros_like(available)
                    prediction = model(lead_i, base, beat, stats, available)
                    correction = prediction - base
                    loss = (1 - correlation(prediction, truth)).mean() + 0.01 * correction.square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                metric = assess(model, val, device, "correct" if kind == "context" else "disabled")
                row = {"group": kind, "epoch": epoch, "train_loss": float(np.mean(losses)),
                       "validation_mean_r_II": metric["mean_r"]}
                with (OUTPUT / "history.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if metric["mean_r"] > best:
                    best = metric["mean_r"]
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "B5_sha256": file_sha256(B5)}, OUTPUT / f"{kind}_best.pt")
            saved = torch.load(OUTPUT / f"{kind}_best.pt", map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            summary[kind] = {"best_epoch": saved["epoch"],
                             "correct": assess(model, val, device,
                                               "correct" if kind == "context" else "disabled")}
            if kind == "context":
                summary[kind]["disabled"] = assess(model, val, device, "disabled")
                summary[kind]["shuffled"] = {str(seed): assess(model, val, device, "shuffled", seed)
                                               for seed in SEEDS}
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"final_summary": summary}), flush=True)
    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())
