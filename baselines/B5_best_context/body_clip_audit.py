"""Read original body history; quantify feature clipping without modifying cache/model/QC."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import write, csv_write
from .body_prepare import load_cache
from .body_features import relation_features
from .adapters import history_from_sample, _visible_digest
from .evaluate import pairs


def run(args):
    root, output = Path(args.cache).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty audit directory; tee logs must be adjacent")
    cache = load_cache(root)
    from baselines.B5.config import load_config
    from baselines.B5.data import common_config
    from ecg12gen.dataset import JointAnchorDataset
    cfg = common_config(load_config(args.config, data_only=True))
    lookup = {(r["split"], r["pair_id"]): r for r in cache["records"]}
    results, examples, visited = [], [], set()
    for split in ("train", "validation"):
        dataset = JointAnchorDataset(cfg, "task2", split)
        try:
            rows = [dataset._rows[i] for i in dataset._indices]
            for indices in pairs(rows):
                first = rows[indices[0]]
                if first["input_type"] != "body_scale_d6":
                    continue
                key = split, first["pair_id"]
                if key not in lookup:
                    raise ValueError("Body audit and original cache have different pair populations")
                record = lookup[key]
                history = history_from_sample(dataset[indices[0]])
                if _visible_digest(history) != record["history_visible_sha256"]:
                    raise ValueError("Original body history/QC changed since preparation")
                audit = {}
                beat, stats, available = relation_features(history, audit=audit)
                with np.load(root / record["file"], allow_pickle=False) as a:
                    if (not np.array_equal(beat, a["beat"]) or not np.array_equal(stats, a["stats"])
                            or bool(a["available"]) != available):
                        raise ValueError("Audit altered original features or cache is not aligned")
                uncut, clipped = audit.pop("_unclipped_template", None), audit.pop("_clipped_template", None)
                row = {"split": split, "subject_id": record["subject_id"], "pair_id": record["pair_id"],
                    "target_record_id": record["target_record_id"], "I_II_eligible": record["I_II_eligible"],
                    "history_available": available, **audit}
                results.append(row); visited.add(key)
                if split == "validation" and record["subject_id"] in args.subjects and uncut is not None:
                    name = record["pair_id"] + "_templates.csv"
                    csv_write(output / name, [{"time_relative_to_history_event_s": (i - 100) / 500,
                        "I_before_feature_clip": float(uncut[0, i]), "II_before_feature_clip": float(uncut[1, i]),
                        "I_after_feature_clip": float(clipped[0, i]), "II_after_feature_clip": float(clipped[1, i])} for i in range(300)])
                    examples.append({"subject_id": record["subject_id"], "file": name})
                print(f"{split} {record['subject_id']}: II clipping={audit['II_beat_clipped_fraction']}", flush=True)
        finally:
            dataset.close()
    if visited != set(lookup):
        raise ValueError("Audit missed cached body pairs")
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "clipping_records.csv", results)
    summary = []
    for split in ("train", "validation"):
        selected = [r for r in results if r["split"] == split and r["I_II_eligible"] and r["history_available"]]
        fractions = [r["II_beat_clipped_fraction"] for r in selected]
        summary.append({"split": split, "eligible_available_pairs": len(selected),
            "pairs_with_any_II_feature_clipping": sum(v > 0 for v in fractions),
            "median_II_clipped_fraction": float(np.median(fractions)) if fractions else None,
            "max_II_clipped_fraction": max(fractions) if fractions else None})
    write(output / "audit.json", {"completed": True, "training_started": False, "B5_inference_executed": False,
        "cache_unchanged_and_features_exactly_reproduced": True, "summary": summary, "examples": examples,
        "note": "Counts refer to normalization+clipping of selected historical I/II beats. They do not identify raw acquisition saturation or prove why r decreased. No QC/feature/model changes were made."})
    print({"completed": True, "summary": summary}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--subjects", nargs="+", default=["HNU11135", "HNU11141", "HNU11119", "HNU11104"])
    run(p.parse_args())


if __name__ == "__main__":
    main()
