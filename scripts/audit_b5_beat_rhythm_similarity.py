"""Read-only cross-patient cosine audit for the three beat/RR checkpoints."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from baselines.B5.checkpoint import checkpoint_preprocessor, load_checkpoint
from baselines.B5.context_body_beat_rhythm import MAX_BEATS
from baselines.B5.context_body_attention import BodyContextEncoderAttention
from baselines.B5.metadata import DemographicsTable
from ecg12gen.preprocessing import PreprocessingConfig
from scripts.evaluate_b5_body_diagnostics import ROOT, SOURCE, body_pairs, check_existing_roster, load_model
from scripts.prepare_task2_context_scales import with_context_scales
from scripts.train_b5_body_context import BatchAdapter, BodySubset, on_device


def pairwise(vectors: list[torch.Tensor]) -> dict:
    array = torch.stack(vectors).float()
    if not torch.isfinite(array).all():
        raise ValueError("Nonfinite feature vector")
    norms = array.norm(dim=1)
    if (norms <= 1e-10).any():
        raise ValueError("Zero feature vector")
    cos = F.normalize(array, dim=1) @ F.normalize(array, dim=1).T
    values = cos[torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)].numpy()
    return {"n_pairs": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "q10": float(np.quantile(values, 0.1)),
            "q90": float(np.quantile(values, 0.9)), "min": float(values.min()),
            "max": float(values.max()), "mean_norm": float(norms.mean())}


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = load_checkpoint(SOURCE)
    base = checkpoint_preprocessor(source, PreprocessingConfig.from_yaml(ROOT / "configs/preprocessing.yaml"))
    preprocessor = with_context_scales(base, ROOT / "results/B5/context_params/body_A_scales.npz",
                                       "body_scale_d6", "A_raw_window")
    dataset = BodySubset("validation", "A_raw_window")
    try:
        pairs = body_pairs(dataset)
        check_existing_roster(pairs)
        adapter = BatchAdapter(preprocessor, DemographicsTable(ROOT / "data/userinfobean.csv"), 0.0, seed=42)
        batches = [on_device(adapter([item]), device) for item in pairs.values()]
        result = {"n_patients": len(pairs), "validation_windows": len(dataset),
                  "metric": "cross-patient upper-triangle cosine similarity; one context record per patient"}
        for kind in ("beat", "rhythm", "beat_rhythm"):
            model = load_model("A_true", source, device, checkpoint_kind=f"frozen_{kind}")
            model.eval()
            encoder = model.body_encoder
            vectors = {"attention_base": [], "final_embedding": [],
                       "beat_branch": [], "rhythm_branch": []}
            beat_counts, rhythm_ready = [], []
            for batch in batches:
                context, mask = batch["context"], batch["context_time_mask"]
                weights, available = batch["context_window_weights"], batch["context_available"]
                base_vector = BodyContextEncoderAttention.forward(encoder, context, mask, weights, available)[0]
                final_vector = encoder(context, mask, weights, available)[0]
                beat, present, rhythm, rhythm_available = encoder._features(context, mask)
                count = present.sum(dim=1)
                beat_encoded = encoder.beat_branch(beat.flatten(0, 1)).reshape(len(beat), MAX_BEATS, -1)
                beat_vector = ((beat_encoded * present.unsqueeze(-1)).sum(dim=1)
                               / count.clamp_min(1).unsqueeze(-1))[0]
                rhythm_vector = encoder.rhythm_branch(rhythm)[0]
                for name, value in (("attention_base", base_vector), ("final_embedding", final_vector),
                                    ("beat_branch", beat_vector), ("rhythm_branch", rhythm_vector)):
                    vectors[name].append(value.detach().cpu())
                beat_counts.append(int(count.item()))
                rhythm_ready.append(bool(rhythm_available.item()))
            active = {"attention_base", "final_embedding"}
            if kind in ("beat", "beat_rhythm"):
                active.add("beat_branch")
            if kind in ("rhythm", "beat_rhythm"):
                active.add("rhythm_branch")
            result[kind] = {name: pairwise(vectors[name]) for name in sorted(active)}
            result[kind]["beat_count"] = {"min": min(beat_counts), "median": float(np.median(beat_counts)),
                                          "max": max(beat_counts), "at_least_3": sum(x >= 3 for x in beat_counts)}
            result[kind]["rhythm_available"] = sum(rhythm_ready)
            result[kind]["gates"] = {"beat": float(torch.tanh(encoder.beat_gate).item()),
                                     "rhythm": float(torch.tanh(encoder.rhythm_gate).item())}
            del model
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
