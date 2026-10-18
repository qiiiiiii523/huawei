"""Export the frozen control100 FM + updated slow CNN, with scoped new reference."""
import argparse
import copy
from pathlib import Path

import torch

from baselines.B5.checkpoint import atomic_save
from baselines.B5_best.bundle import validate_bundle
from baselines.B5_best.common import state_sha, write, sha
from .common import load_checkpoint


def export(study_path, checkpoint_path, output):
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output bundle")
    cp, study, original, model = load_checkpoint(checkpoint_path, study_path, torch.device("cpu"))
    bundle = copy.deepcopy(original)
    bundle["slow"]["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bundle["slow_state_sha256"] = state_sha(bundle["slow"]["model"])
    bundle["provenance"].update(stable_reference=False, experiment="Control100PCC-reliable-I",
              experiment_group=cp["group"], source_updated_slow_epoch=cp["epoch"],
              original_slow_sha256=original["provenance"]["slow_sha256"],
              slow_sha256=sha(checkpoint_path), study_sha256=sha(study_path),
              FM_frozen=True, final_beta=study["beta"])
    score = cp.get("validation_score")
    bundle["reference_qc_r"] = ({k: score[k] for k in ("task1_r_missing11", "task2_r_missing11")}
                                if score is not None else None)
    bundle['reference_score_protocol'] = study['score_protocol']
    bundle['provenance']['validation_FM'] = 'control100_for_all_arms'
    bundle['provenance']['training_FM_partner'] = cp['group']
    bundle['provenance']['slow_component'] = f"Control100PCC-{cp['group']}-epoch{cp['epoch']}"
    bundle['provenance']['multiscale_training_weight'] = cp.get('multiscale', {}).get('weight', 0.)
    validate_bundle(bundle)
    if state_sha(bundle["fm"]["ema"]) != original["fm_state_sha256"]:
        raise ValueError("Export unexpectedly changed FM")
    atomic_save(output, bundle)
    write(output.with_suffix(".json"), {"bundle_sha256": sha(output), "group": cp["group"],
          "epoch": cp["epoch"], "FM_unchanged": True, "sampling": study["sampling"], "beta": study["beta"],
          "reference_qc_r": bundle["reference_qc_r"], "target_required_at_inference": False,
          "reference_score_protocol": study['score_protocol'],
          "training_cache_required_at_inference": False})
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export(args.study, args.checkpoint, args.output)
    print({"portable_bundle": args.output, "FM_training_started": False}, flush=True)


if __name__ == "__main__":
    main()
