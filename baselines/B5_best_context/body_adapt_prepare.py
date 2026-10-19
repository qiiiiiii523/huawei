"""Rebase unchanged selected B features onto new-FM training/validation predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.B5_best.common import read, write, sha
from .body_prepare import load_cache, write_ledger
from .body_features import BODY_PROTOCOL, LEGACY_ENCODING, BEAT16_ENCODING, feature_encoding
from .body_history import B_POLICY, history_policy
from .selected_study import SELECTED_B43_SHA256
from .four_routes import SAMPLING

PROTOCOL = "B5-body-new-base-adaptation-v1"


def check_source(payload, cache):
    provenance = cache.get("body_adaptation", {})
    if (provenance.get("protocol") != PROTOCOL
            or provenance.get("source_checkpoint_sha256") != SELECTED_B43_SHA256
            or payload.get("protocol") != BODY_PROTOCOL or payload.get("group") != "context"
            or payload.get("record_gate") is not None or payload.get("training_seed") != 43 or payload.get("epoch") != 15
            or feature_encoding(payload) != BEAT16_ENCODING or feature_encoding(cache) != BEAT16_ENCODING
            or history_policy(payload) != B_POLICY or history_policy(cache) != B_POLICY
            or payload.get("cache_sha256") != provenance.get("source_cache_sha256")
            or payload.get("base_checkpoint_sha256") != provenance.get("source_base_sha256")
            or payload.get("sampling") != SAMPLING or payload.get("slow_beta") != .75
            or cache["frozen"].get("sampling") != SAMPLING or cache["frozen"].get("slow_beta") != .75):
        raise ValueError("Adaptation must initialize selected frozen B43-best15 with matching B features/settings")
    return {**provenance, "target_base_sha256": cache["frozen"]["base_checkpoint_sha256"],
            "optimizer_reset": True, "source_epoch": 15, "epoch_numbering": "additional epochs; zero retains source weights"}


def prepare(args):
    old_root, source_root, output = map(lambda p: Path(p).resolve(), (args.old_body_cache, args.cache, args.output))
    for root in (old_root, source_root):
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Adaptation output must be separate from both input caches")
    if sha(args.source_checkpoint) != SELECTED_B43_SHA256:
        raise ValueError("Use selected B43 seed43 best15 as adaptation source")
    old, source = load_cache(old_root), load_cache(source_root)
    if (feature_encoding(old) != BEAT16_ENCODING or history_policy(old) != B_POLICY
            or feature_encoding(source) != LEGACY_ENCODING or "history_preprocessing" in source):
        raise ValueError("Use original selected B cache plus fresh new-FM raw-history prediction cache")
    for task in ("task1", "task2"):
        for key in ("metadata_sha256", "target_sha256", "QC_sha256"):
            if old["frozen"]["baseline_population"][task][key] != source["frozen"]["baseline_population"][task][key]:
                raise ValueError("New base changed target population/QC: " + task + "/" + key)
    provenance = dict(protocol=PROTOCOL, source_checkpoint_sha256=sha(args.source_checkpoint),
        source_cache_sha256=sha(old_root/"cache.json"), source_base_sha256=old["frozen"]["base_checkpoint_sha256"])
    expected = dict(protocol=BODY_PROTOCOL, feature_encoding=BEAT16_ENCODING, history_preprocessing=B_POLICY,
        body_adaptation=provenance, prediction_source_cache=str(source_root),
        prediction_source_cache_sha256=sha(source_root/"cache.json"), frozen=source["frozen"])
    ledger = output/"cache.json"
    if args.resume and ledger.exists():
        manifest = read(ledger)
        if any(manifest.get(k) != v for k,v in expected.items()):
            raise ValueError("Adaptation resume changed old body features or new P predictions")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use new adaptation cache or --resume")
        manifest = {**expected, "completed": False, "records": []}
        write_ledger(ledger,manifest)
    key = lambda r: (r["split"],r["pair_id"])
    lookup = {key(r):r for r in old["records"]}
    new_lookup = {key(r):r for r in source["records"]}
    if set(lookup) != set(new_lookup) or len(lookup)!=len(old["records"]) or len(new_lookup)!=len(source["records"]):
        raise ValueError("Old B cache/new FM cache populations differ or have duplicate pairs")
    existing = {key(r):r for r in manifest["records"]}
    if not set(existing).issubset(new_lookup):
        raise ValueError("Partial adaptation cache contains unexpected records")
    for record in source["records"]:
        identity = key(record)
        original = lookup[identity]
        for field in ("split","pair_id","subject_id","target_record_id","history_record_id","history_visible_sha256","I_II_eligible"):
            if original[field]!=record[field]:
                raise ValueError("New base changed visible history/identity/QC: "+field)
        with np.load(old_root/original["file"],allow_pickle=False) as left, np.load(source_root/record["file"],allow_pickle=False) as right:
            for field in ("anchor","target_ii","quality"):
                if left[field].dtype!=right[field].dtype or not np.array_equal(left[field],right[field]):
                    raise ValueError("Adaptation changed "+field)
            arrays = {k:right[k].copy() for k in right.files}
            # Only the base-II input changes; reuse precisely the same O16/B features.
            arrays.update({k:left[k].copy() for k in ("beat","stats","available")})
        if identity in existing:
            if sha(output/existing[identity]["file"])!=existing[identity]["sha256"]:
                raise ValueError("Partial adaptation cache changed")
            continue
        dest=output/record["file"]; dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(".pending")
        with temp.open("wb") as h: np.savez_compressed(h,**arrays)
        temp.replace(dest)
        manifest["records"].append({**record,"sha256":sha(dest),"history_available":bool(arrays["available"]),
            "processed_history_sha256":original["processed_history_sha256"]})
        write_ledger(ledger,manifest)
    if len(manifest["records"])!=len(source["records"]): raise ValueError("Missing adaptation records")
    manifest["completed"]=True; write_ledger(ledger,manifest)
    load_cache(output)
    report=dict(ready=True,training_started=False,records=len(manifest["records"]),
        source_body_sha256=SELECTED_B43_SHA256,new_base_sha256=source["frozen"]["base_checkpoint_sha256"],
        same_I_targets_QC_history_B_features=True,changed_input="new base predicted II only")
    write(output/"preflight.json",report); print(report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True)
    p.add_argument("--old-body-cache",required=True)
    p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--resume",action="store_true")
    prepare(p.parse_args())


if __name__=="__main__": main()
