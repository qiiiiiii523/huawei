"""Rebuild the pinned plain FM+P cache from observed I and verified window indices."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from baselines.B5_best.common import sha, write
from baselines.B5_best.validate import evaluate, pair_indices
from baselines.B5_best.bundle import load_bundle
from baselines.B5.metadata import DemographicsTable
from baselines.B5.runtime import device_from_name
from baselines.B5_final.profile import BASE_SHA, resolve_assets


class IndexedValidation:
    """Expose only observed I + demographics to Predictor; hidden leads score only."""
    def __init__(self, target, metadata, demographics):
        self.targets = np.load(target, mmap_mode="r", allow_pickle=False)
        with Path(metadata).open(encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if self.targets.shape != (len(self.rows), 12, 5000) or self.targets.dtype != np.float32:
            raise ValueError("Target/index shape or dtype mismatch")
        pair_indices(self.rows)
        self.quality = np.asarray([[int(x) for x in r["target_lead_mask"].split("|")]
                                   for r in self.rows], dtype=bool)
        if self.quality.shape != (len(self.rows), 12):
            raise ValueError("Invalid QC shape")
        self.demographics = demographics
        self.base = self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return IndexedItem(self, i)


class IndexedItem:
    def __init__(self, dataset, index):
        self.dataset, self.index = dataset, index

    @property
    def anchor_i_ecg(self):
        return np.asarray(self.dataset.targets[self.index, :1]).copy()

    def __getitem__(self, key):
        ds, i = self.dataset, self.index
        return {"target_uV": ds.targets[i], "quality_mask": ds.quality[i],
                "evaluation_metadata": ds.rows[i]}[key]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--existing-project", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    incoming, old, run = map(lambda x: Path(x).resolve(), (args.input_dir, args.existing_project, args.run))
    if not run.is_relative_to(root / "results"):
        raise ValueError("Run must be inside this checkout's results directory")
    if run.exists():
        raise FileExistsError("Use a new run directory")
    reference = json.loads((incoming / "evaluation.json").read_text(encoding="utf-8-sig"))
    if reference["checkpoint_sha256"] != BASE_SHA:
        raise ValueError("Wrong baseline reference")
    assets = resolve_assets(root, checkpoint="models/base.pt", body_checkpoint="models/body.pt")
    checks = reference["provenance"]["slow_source_checks"]
    sources = {"demographics_csv": old / "huawei/data/userinfobean.csv",
        "subject_split_csv": root / "metadata/subject_split.csv",
        "device_interpretation_qc_csv": root / "metadata/device_interpretation_qc.csv",
        "task1_pair_manifest": root / "metadata/pair_manifest_task1.csv",
        "task2_pair_manifest": root / "metadata/pair_manifest_task2.csv",
        "task2_window_metadata": old / "task2_record_context_v3/task2_window_metadata.csv",
        "task2_validation_target": old / "task2_record_context_v3/task2_validation_target.npy"}
    for key, source in sources.items():
        if sha(source) != checks[key]["sha256"]:
            raise ValueError("Existing data version mismatch: " + key)
    if sha(incoming / "task1_validation_target.npy") != reference["population"]["task1"]["target_sha256"]:
        raise ValueError("Wrong Task1 validation target")
    indices = {}
    for task in ("task1", "task2"):
        expected = reference["population"][task]["metadata_sha256"]
        matches = [f for f in incoming.glob("*.csv") if sha(f) == expected]
        if len(matches) != 1:
            raise ValueError("Missing exact evaluation index: " + task)
        indices[task] = matches[0]
    run.mkdir(parents=True)
    local = run / "inputs"
    local.mkdir()
    # Copy only metadata; all original project's arrays are read-only mmap sources.
    for key in ("demographics_csv", "subject_split_csv", "device_interpretation_qc_csv",
                "task1_pair_manifest", "task2_pair_manifest"):
        shutil.copyfile(sources[key], local / sources[key].name)
    import yaml
    common = yaml.safe_load((root / "configs/common.yaml").read_text())
    common["paths"].update({"data_root": str(old),
        "task2_output": str(old / "task2_record_context_v3"),
        "subject_split_csv": str(local / "subject_split.csv"),
        "device_interpretation_qc_csv": str(local / "device_interpretation_qc.csv"),
        "task1_pair_manifest_csv": str(local / "pair_manifest_task1.csv"),
        "task2_pair_manifest_csv": str(local / "pair_manifest_task2.csv")})
    write(local / "common.json", common)
    cfg = yaml.safe_load((root / "configs/experiments/b5_base.yaml").read_text())
    cfg["paths"].update({"common_config": str(local / "common.json"),
        "huawei_data_root": str(old), "data_root": str(local),
        "demographics": str(local / "userinfobean.csv")})
    write(local / "validation.json", cfg)
    demographics = DemographicsTable(local / "userinfobean.csv")
    datasets = {task: IndexedValidation(
        incoming / "task1_validation_target.npy" if task == "task1" else sources["task2_validation_target"],
        indices[task], demographics) for task in ("task1", "task2")}
    # Test provider compatibility and QC hashes BEFORE expensive FM generation.
    from baselines.B5_best_context.evaluate import HistoryProvider
    from baselines.B5_best.common import csv_write
    for task, ds in datasets.items():
        np.save(local / (task + "_QC.npy"), ds.quality, allow_pickle=False)
        if sha(local / (task + "_QC.npy")) != reference["population"][task]["QC_sha256"]:
            raise ValueError("QC reconstruction differs: " + task)
        csv_write(local / (task + "_index.csv"), ds.rows)
        if sha(local / (task + "_index.csv")) != reference["population"][task]["metadata_sha256"]:
            raise ValueError("Index serialization differs: " + task)
    provider = HistoryProvider(local / "validation.json", {"rows": datasets["task2"].rows,
        "target": datasets["task2"].targets, "quality": datasets["task2"].quality})
    provider.close()
    write(run / "preflight.json", {"passed": True, **assets.report(),
        "task_windows": {t: len(ds) for t, ds in datasets.items()},
        "demographics": demographics.audit(), "source_checks": {k: str(v) for k, v in sources.items()}})
    print("PREFLIGHT PASSED: exact target/index/QC and live Task2 history matched", flush=True)
    if args.preflight_only:
        return
    bundle = load_bundle(assets.base)
    evaluate(bundle, datasets, device_from_name(args.device), run / "base_inference", BASE_SHA,
             batch_size=8, verify_reference=True, reference_report=reference["score"])
    subprocess.run([sys.executable, "-u", "-m", "baselines.B5_final.body_scope",
        "--repository-root", str(root), "--checkpoint", str(assets.base),
        "--body-checkpoint", str(assets.body), "--config", str(local / "validation.json"),
        "--baseline-evaluation", str(run / "base_inference"), "--output", str(run / "study"),
        "--device", args.device, "--verify-reference"], check=True, cwd=root)


if __name__ == "__main__":
    main()
