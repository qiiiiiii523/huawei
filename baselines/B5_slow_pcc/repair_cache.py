"""Repair ONLY the known pre-training 0e1cd7f cache failure; preserve every FM record."""
import argparse
import copy
import hashlib
from pathlib import Path

from baselines.B5_best import SAMPLING, SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle, bundle_slow_beta
from baselines.B5_best.common import read, sha
from . import PROTOCOL, CACHE_PROTOCOL
from .common import atomic_json, code_digest, load_cache

OLD_CODE_SHA256 = "db955b3b5052099971d7f79c040407741f0d0e195574e7e8f6897aa76fa6f9f9"
OLD_CACHE_PROTOCOL = "B5-best-SlowPCC-fixed-FM-records-v1"
UNCHANGED_MODEL_CODE = {
    "baselines/B5_slow_pcc/core.py": "05b52a8f2237991f9179507af4f17cc523450dcc627d80b08b87c25109475a94",
    "baselines/B5_slow_pcc/losses.py": "c1a4d0b9bd8dd884c76731d35fb76737676c11f79e2186a926f37bcf9429f495",
    "baselines/B5/flow.py": "4dc30f8b3bc75504ff777dbdee7b504d6179453b280343013b501c571f5848f0",
    "baselines/B5/model.py": "e2d6a316cea60e7b301de12bfe7843b8e1779b62ec955a4458d4ba8ba3d3e63f",
    "baselines/B5_best/slow.py": "e439779a7d2d46f8963729e058680f4f8fecc49a8014a0ab3ed7d5f54a2ee1ab",
    "baselines/B5_slow_pcc/prepare.py": "86cbbd0616a0e285cf9baa49d43cba93cc0df5916df47a6d5f58e0b93a460f1f",
    "baselines/B5_slow_pcc/train.py": "c02fa0200ebc1bf22718b7f732733de1b2230fe8ad3dd913198440cf1c1a1b4a",
    "baselines/B5_slow_pcc/export.py": "e73bc4c290b68b85f080b88957ba2a8298c89f25192c68b3f960c725e25f11cc",
}


def backup(path, destination):
    if destination.exists():
        if path.read_bytes() != destination.read_bytes():
            raise ValueError("Existing backup differs; keep both files and inspect before repair")
    else:
        destination.write_bytes(path.read_bytes())


def repair(study_path, cache_root, finish=False):
    study_path, cache_root = Path(study_path).resolve(), Path(cache_root).resolve()
    run_root = study_path.parent.parent
    for group in ("control", "pcc"):
        folder = run_root / group
        if folder.exists() and (any(folder.glob("*.pt")) or (folder / "history.jsonl").exists()):
            raise ValueError("Training already started: do not rewrite its study/cache identity")
    code_root = Path(__file__).resolve().parents[2]
    for relative, expected in UNCHANGED_MODEL_CODE.items():
        actual = hashlib.sha256((code_root / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Repair is restricted to cache-only changes; model/loss/training changed: " + relative)
    current = read(study_path)
    current_code = code_digest()
    original_study_backup = study_path.with_name("study.before_pair_cache_fix.json")
    progress_path = cache_root / "progress.json"
    original_progress_backup = cache_root / "progress.before_pair_cache_fix.json"
    if current.get("code_sha256") == OLD_CODE_SHA256:
        original = current
        original_sha = sha(study_path)
        if not progress_path.is_file():
            raise FileNotFoundError("No saved cache progress; cannot reuse generated records")
        progress = read(progress_path)
        if progress.get("study_sha256") != original_sha or progress.get("protocol") != OLD_CACHE_PROTOCOL:
            raise ValueError("Original cache progress/study identity mismatch")
        backup(study_path, original_study_backup)
        backup(progress_path, original_progress_backup)
    elif current.get("code_sha256") == current_code and current.get("cache_pair_repair"):
        original = read(original_study_backup)
        original_sha = sha(original_study_backup)
        if original.get("code_sha256") != OLD_CODE_SHA256:
            raise ValueError("Backup is not the known source version")
        progress = read(progress_path)
        if progress.get("study_sha256") not in (original_sha, sha(study_path)):
            raise ValueError("Interrupted repair has an unrelated progress identity")
    else:
        raise ValueError("This repair only accepts the exact released 0e1cd7f study, not an arbitrary checksum bypass")
    if original.get("protocol") != PROTOCOL or original.get("sampling") != SAMPLING or original.get("score_protocol") != SCORE_PROTOCOL:
        raise ValueError("Source experiment/sampling/score protocol changed")
    bundle_path = study_path.parent / "components.pt"
    if sha(bundle_path) != original["bundle_sha256"]:
        raise ValueError("Fixed weight bundle changed")
    bundle = load_bundle(bundle_path)
    if bundle_slow_beta(bundle) != original["beta"]:
        raise ValueError("Source study/bundle slow coefficient mismatch")
    for key, item in original["source_checks"].items():
        print("Checking recovery source: " + key, flush=True)
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Source data/QC/demographics changed: " + key)
    for path, expected in original["source_evaluation_checks"].items():
        if sha(path) != expected:
            raise ValueError("Original FM validation predictions changed: " + path)
    train_records = [item for key, item in progress["records"].items() if key.startswith("train:")]
    if len(train_records) != original["train_records"]:
        raise ValueError("Recovery requires all training FM records to be complete; finish missing records with the source version first")
    for index, item in enumerate(progress["records"].values(), 1):
        path = (cache_root / item["file"]).resolve()
        if not path.is_relative_to(cache_root) or sha(path) != item["sha256"]:
            raise ValueError("A saved cache record changed: " + item["file"])
        if index % 25 == 0 or index == len(progress["records"]):
            print(f"Verified existing cache files: {index}/{len(progress['records'])}", flush=True)
    updated = copy.deepcopy(original)
    updated["code_sha256"] = current_code
    updated["cache_pair_repair"] = {"from_commit": "0e1cd7f", "from_study_sha256": original_sha,
          "from_code_sha256": OLD_CODE_SHA256, "model_loss_sampling_changed": False,
          "training_FM_records_preserved": len(train_records)}
    # Save source files first; replacement is atomic and interrupted repairs can be resumed.
    atomic_json(study_path, updated)
    progress["protocol"], progress["study_sha256"] = CACHE_PROTOCOL, sha(study_path)
    atomic_json(progress_path, progress)
    report = {"repaired": True, "train_records_reused": len(train_records), "training_started": False,
              "new_FM_inference_allowed": False, "study_sha256": sha(study_path),
              "backups": [str(original_study_backup), str(original_progress_backup)]}
    atomic_json(cache_root / "repair_report.json", report)
    if finish:
        from .cache import build
        build(study_path, cache_root, "cpu", resume=True, reuse_only=True)
        report["cache_completed"] = load_cache(cache_root, study_path)["completed"]
        atomic_json(cache_root / "repair_report.json", report)
    print(report, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--finish-cache", action="store_true", help="Complete pair-specific validation on CPU; never regenerate FM")
    args = p.parse_args()
    repair(args.study, args.cache, args.finish_cache)


if __name__ == "__main__":
    main()
