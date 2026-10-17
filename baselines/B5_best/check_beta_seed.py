"""Paired inference-only beta check; one FM draw ensemble per target and seed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import checkpoint_preprocessor
from baselines.B5.config import load_config, ModelConfig
from baselines.B5.data import preprocessing_config
from baselines.B5.flow import sample
from baselines.B5.runtime import build_validation_datasets, device_from_name, seed_all
from ecg12gen.evaluate import evaluate_record_predictions, write_report
from .bundle import DEFAULT_PROFILE, export_profile
from .common import read, write, csv_write
from .combine import combine, upsample_native
from .predict import Predictor
from .scoring import diagnostic_summary, scores_from_summaries, write_score_report
from .slow import predict_native
from .validate import pair_indices, verify_reference_inputs


class ComponentPredictor(Predictor):
    @torch.no_grad()
    def components(self, anchor, meta, key, seed, batch_size):
        raw = np.asarray(anchor, dtype=np.float32)
        if raw.shape != (1, 60000) or not np.isfinite(raw).all():
            raise ValueError("Expected finite full-record visible I [1,60000]")
        fields = {k: torch.from_numpy(np.asarray(v)[None].copy()).to(self.device)
                  for k, v in meta.items()}
        anchor50 = raw.reshape(1, 6000, 10).mean(-1) / self.scale[0]
        native = predict_native(self.slow, {
            "anchor": torch.from_numpy(anchor50[None]).to(self.device), **fields
        })[0].cpu().numpy()
        learned = upsample_native(native.astype(np.float64) *
                                  self.scale[1:, None].astype(np.float64))
        original = np.empty((12, 60000), np.float32)
        settings = dict(self.bundle["sampling"], seed=seed)
        for offset in range(0, 12, batch_size):
            starts = list(range(offset * 5000, min(offset + batch_size, 12) * 5000, 5000))
            inputs = {"anchor": torch.from_numpy(np.stack([
                raw[:, s:s + 5000] / self.scale[0] for s in starts])).to(self.device)}
            inputs.update({k: torch.from_numpy(np.stack([v] * len(starts))).to(self.device)
                           for k, v in meta.items()})
            generated = sample(self.fm, inputs, [key + ":" + str(s) for s in starts], **settings)
            restored = generated.float().cpu().numpy() * self.scale[None, :, None]
            for i, start in enumerate(starts):
                original[:, start:start + 5000] = restored[i]
        return original, learned


def mix(original, learned, beta):
    return combine(original, learned, slow_beta=beta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", default="/home/qht/huawei")
    p.add_argument("--config")
    p.add_argument("--profile", default=str(DEFAULT_PROFILE), help="Default promoted pair; legacy profile is explicit")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not 0 <= args.seed < 2**32 or args.batch_size < 1:
        p.error("Need a valid seed and positive batch size")
    root, output = Path(args.repository_root), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    bundle = export_profile(args.profile, output / "components.pt")
    config_path = args.config or root / "configs/experiments/b5_main_v2_public_finetune_meta.yaml"
    cfg = load_config(config_path, data_only=True)
    verify_reference_inputs(bundle, cfg)
    if ModelConfig.from_dict(cfg["model"]).fingerprint != bundle["fm"]["architecture_hash"]:
        raise ValueError("Config and FM architecture disagree")
    preprocessor = checkpoint_preprocessor(bundle["fm"], preprocessing_config(cfg))
    datasets = build_validation_datasets(cfg, preprocessor)
    for task, ds in datasets.items():
        expected = bundle["provenance"]["source_manifests"].get(task)
        if expected and expected != ds.manifest_digest:
            raise ValueError("Validation population changed: " + task)
    seed_all(args.seed, True)
    engine = ComponentPredictor(bundle, device_from_name(args.device))
    betas = (1.0, 0.75)
    summaries = {b: {} for b in betas}
    cache = {}
    sampling = dict(bundle["sampling"], seed=args.seed)
    write(output / "run.json", {"completed": False, "training_started": False,
          "sampling": sampling, "betas": betas, "provenance": bundle["provenance"]})
    for task, ds in datasets.items():
        if task not in ("task1", "task2"):
            raise ValueError("Unexpected task")
        shape = (len(ds), 12, 5000)
        predictions = {b: np.empty(shape, np.float32) for b in betas}
        targets, quality = np.empty(shape, np.float32), np.empty((len(ds), 12), bool)
        metadata = [None] * len(ds)
        for indices in pair_indices(ds.rows):
            first = ds.rows[indices[0]]
            anchor = np.concatenate([ds.base[i].anchor_i_ecg for i in indices], axis=-1).astype(np.float32)
            meta = ds.demographics.get(first["subject_id"])
            digest = hashlib.sha256(anchor.tobytes())
            for k, v in sorted(meta.items()):
                digest.update(k.encode()); digest.update(np.asarray(v).tobytes())
            key = first["target_record_id"]
            if key not in cache:
                original, learned = engine.components(anchor, meta, key, args.seed, args.batch_size)
                cache[key] = (digest.hexdigest(), {b: mix(original, learned, b) for b in betas})
            previous, combined = cache[key]
            if previous != digest.hexdigest():
                raise ValueError("Repeated target has different visible I/demographics")
            for pos, index in enumerate(indices):
                item = ds[index]
                targets[index], quality[index] = item["target_uV"], item["quality_mask"]
                metadata[index] = item["evaluation_metadata"]
                for b in betas:
                    predictions[b][index] = combined[b][:, pos * 5000:(pos + 1) * 5000]
            print(f"seed={args.seed} {task}: {key} complete (both betas)", flush=True)
        for b in betas:
            folder = output / f"beta_{b:g}" / task
            folder.mkdir(parents=True, exist_ok=True)
            overall, details = evaluate_record_predictions(predictions[b], targets, task, metadata,
                                                            target_quality_mask=quality)
            overall = diagnostic_summary(overall)
            overall.update(sampling=sampling, beta=b, target_used_at_inference=False)
            write_report(folder, overall, details, title="Paired seed/beta full-record QC r")
            csv_write(folder / "window_metadata.csv", metadata)
            summaries[b][task] = overall
        del predictions, targets, quality
    rows = []
    for b in betas:
        score = scores_from_summaries(summaries[b])
        write_score_report(output / f"beta_{b:g}", score)
        rows.append({"seed": args.seed, "beta": b, **score})
    csv_write(output / "summary.csv", rows)
    result = {"completed": True, "training_started": False, "sampling": sampling,
              "scores": rows, "delta_mean_r_beta075_minus_beta1":
              rows[1]["main_correlation"] - rows[0]["main_correlation"],
              "unique_generated_targets": len(cache), "same_FM_predictions_for_both_betas": True,
              "deployment_model_changed": False, "provenance": bundle["provenance"]}
    write(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
