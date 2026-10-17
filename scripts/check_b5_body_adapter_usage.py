"""Read-only A checkpoint audit: patient embeddings and FiLM signal strength.

One target window per body-scale validation pair is inspected. The same state,
time, anchor and weights are used for true-context and disabled-context velocity
predictions; no optimizer or sampling loop is run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from baselines.B5.checkpoint import checkpoint_preprocessor, load_checkpoint
from baselines.B5.metadata import DemographicsTable
from ecg12gen.preprocessing import PreprocessingConfig
from scripts.evaluate_b5_body_diagnostics import ROOT, SOURCE, body_pairs, check_existing_roster, load_model
from scripts.prepare_task2_context_scales import with_context_scales
from scripts.train_b5_body_context import BatchAdapter, BodySubset, on_device


def describe(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu().flatten().numpy()
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


@torch.no_grad()
def inspect(device: torch.device, checkpoint_kind: str = "A") -> dict:
    source = load_checkpoint(SOURCE)
    base = checkpoint_preprocessor(source, PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml"))
    preprocessor = with_context_scales(
        base, ROOT / "results/B5/context_params/body_A_scales.npz",
        "body_scale_d6", "A_raw_window")
    dataset = BodySubset("validation", "A_raw_window")
    try:
        pairs = body_pairs(dataset)
        check_existing_roster(pairs)
        adapter = BatchAdapter(preprocessor, DemographicsTable(ROOT / "data/userinfobean.csv"),
                               0.0, seed=42)
        model = load_model("A_true", source, device, checkpoint_kind=checkpoint_kind)
        original = model._condition_bottleneck
        feature_rms, modulation_ratio = [], []

        def tracked(self, features, condition):
            output = original(features, condition)
            if condition.context_available.any():
                denominator = features.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-12)
                numerator = (output - features).square().mean(dim=(1, 2)).sqrt()
                feature_rms.extend(denominator.detach().cpu())
                modulation_ratio.extend((numerator / denominator).detach().cpu())
            return output

        model._condition_bottleneck = MethodType(tracked, model)
        embeddings, gammas, betas, velocity_ratios = [], [], [], []
        for item in pairs.values():
            batch = on_device(adapter([item]), device)
            condition = model.encode_body_conditions(
                batch["anchor"], batch["numeric"], batch["sex"],
                batch["field_mask"], batch["age_topcoded"],
                batch["context"], batch["context_time_mask"],
                batch["context_window_weights"], batch["context_available"])
            if not condition.context_available.all():
                raise ValueError("A validation context unexpectedly unavailable")
            embedding = condition.body_embedding
            gamma, beta = model.body_film.projection(embedding).chunk(2, dim=-1)
            state = torch.zeros((1, 11, batch["anchor"].shape[-1]), device=device)
            time = torch.full((1,), 0.5, device=device)
            velocity_on = model.velocity(state, time, condition)
            velocity_off = model.velocity(
                state, time, replace(condition, context_available=torch.zeros_like(condition.context_available)))
            ratio = ((velocity_on - velocity_off).square().mean().sqrt()
                     / velocity_off.square().mean().sqrt().clamp_min(1e-12))
            embeddings.append(embedding[0].detach().cpu())
            gammas.append(gamma[0].detach().cpu())
            betas.append(beta[0].detach().cpu())
            velocity_ratios.append(ratio.detach().cpu())
        vectors = torch.stack(embeddings)
        gamma_values, beta_values = torch.stack(gammas), torch.stack(betas)
        cosine = F.normalize(vectors, dim=1) @ F.normalize(vectors, dim=1).T
        off_diagonal = cosine[~torch.eye(len(vectors), dtype=torch.bool)]
        return {
            "checkpoint": {"A": "A/epoch_0015/best.pt", "frozen": "A_frozen_adapter_v1/best.pt",
                           "frozen_meanstd": "A_frozen_meanstd_v1/best.pt",
                           "frozen_attention": "A_frozen_attention_v1/best.pt",
                           "frozen_lead_relation": "A_frozen_lead_relation_v1/best.pt"}[checkpoint_kind],
            "n_pairs": len(pairs),
            "embedding_norm": describe(vectors.norm(dim=1)),
            "embedding_mean_dimension_std": float(vectors.std(dim=0).mean()),
            "embedding_pairwise_cosine": describe(off_diagonal),
            "gamma_rms": describe(gamma_values.square().mean(dim=1).sqrt()),
            "beta_rms": describe(beta_values.square().mean(dim=1).sqrt()),
            "gamma_mean_dimension_std": float(gamma_values.std(dim=0).mean()),
            "beta_mean_dimension_std": float(beta_values.std(dim=0).mean()),
            "bottleneck_feature_rms": describe(torch.stack(feature_rms)),
            "bottleneck_modulation_relative_rms": describe(torch.stack(modulation_ratio)),
            "velocity_true_vs_disabled_relative_rms": describe(torch.stack(velocity_ratios)),
            "diagnostic_state": "zeros", "flow_time": 0.5,
        }
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-kind", choices=("A", "frozen", "frozen_meanstd", "frozen_attention",
                                                      "frozen_lead_relation"), default="A")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --device cpu")
    output = args.output or (ROOT / "results/context_body" /
                             {"A": "diagnostics", "frozen": "A_frozen_adapter_v1/diagnostics",
                              "frozen_meanstd": "A_frozen_meanstd_v1/diagnostics",
                              "frozen_attention": "A_frozen_attention_v1/diagnostics",
                              "frozen_lead_relation": "A_frozen_lead_relation_v1/diagnostics"}[args.checkpoint_kind]
                             / "adapter_usage.json")
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    report = inspect(torch.device(args.device), checkpoint_kind=args.checkpoint_kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
