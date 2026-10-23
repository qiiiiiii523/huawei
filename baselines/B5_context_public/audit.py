"""Read-only PTB header/source audit. Does NOT certify waveform QC or start training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from . import PROTOCOL

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def partition(subjects):
    # Match the existing PTB public-slow patient's deterministic split.
    order = sorted(set(subjects), key=lambda s: hashlib.sha256(f"42:{s}".encode()).hexdigest())
    val = set(order[:max(1, round(.2 * len(order)))])
    return {s: "validation" if s in val else "train" for s in order}


def inspect_header(path, root, minimum):
    lines = [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    first = lines[0].split()
    channels, fs, samples = int(first[1]), float(first[2]), int(first[3])
    signals = [s.split() for s in lines[1:1 + channels]]
    if (first[0] != path.stem or channels < 12 or len(signals) != channels or fs != 1000
            or samples < minimum * fs or any(len(s) < 9 for s in signals)):
        raise ValueError("Incomplete header, non-1000Hz source, or shorter than minimum")
    names = [s[-1].lower() for s in signals]
    if any(names.count(lead) != 1 for lead in LEADS):
        raise ValueError("Missing/ambiguous canonical lead")
    sources = [path]
    for filename in sorted({s[0] for s in signals}):
        file_signals = [s for s in signals if s[0] == filename]
        if any(s[1] != "16" for s in file_signals):
            raise ValueError("Only the PTB documented plain format16 layout is audited")
        p = (path.parent / filename).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size == 0:
            raise ValueError("Missing/empty/outside-root waveform")
        if p.stat().st_size != samples * len(file_signals) * 2:
            raise ValueError("Waveform byte length disagrees with format16 header")
        sources.append(p)
    canonical_files = sorted({s[0] for s in signals if s[-1].lower() in LEADS})
    return dict(record_id=path.relative_to(root).with_suffix("").as_posix(), subject_id=path.parent.name,
                seconds=samples / fs, header_sha256=sha(path),
                sources=[str(p.relative_to(root).as_posix()) for p in sources],
                canonical_sources=[(path.parent / name).relative_to(root).as_posix() for name in canonical_files])


def make_pairs(records, split):
    result = []
    for subject in sorted({r["subject_id"] for r in records}):
        own = sorted((r for r in records if r["subject_id"] == subject), key=lambda r: r["record_id"])
        if len(own) < 2:
            continue
        for n, target in enumerate(own):
            history = own[(n + 1) % len(own)]
            result.append(dict(subject_id=subject, split=split[subject], target_record_id=target["record_id"],
                history_record_id=history["record_id"], target_seconds=target["seconds"], history_seconds=history["seconds"],
                target_weight=1, history_policy="different_record_same_subject_not_claimed_chronologically_prior"))
    return result


def audit(root, output, minimum=10):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not isinstance(minimum, (float, int)) or not 0 < minimum < float("inf"):
        raise ValueError("minimum must be finite and positive")
    if output.is_relative_to(root):
        raise ValueError("Audit output must be outside the read-only source root")
    if output.exists():
        raise FileExistsError("Choose a new audit output; source data are read-only")
    headers = sorted(root.glob("patient*/*.hea"))
    if not headers:
        raise FileNotFoundError("Expected PTB Diagnostic patient*/*.hea")
    split = partition([p.parent.name for p in headers])
    records, rejected, fingerprints, seen = [], [], {}, {}
    for path in headers:
        try:
            row = inspect_header(path, root, minimum)
            digest = hashlib.sha256()
            for name in row["sources"]:
                if name not in fingerprints:
                    fingerprints[name] = sha(root / name)
                fingerprint = fingerprints[name]
                if name in row["canonical_sources"]:
                    digest.update(fingerprint.encode())
            key = digest.hexdigest()
            if key in seen:
                raise ValueError("Duplicate waveform source bytes: " + seen[key])
            seen[key] = row["record_id"]
            records.append(row)
        except (ValueError, OSError, IndexError) as e:
            rejected.append(dict(header=path.relative_to(root).as_posix(), reason=str(e)))
    pairs = make_pairs(records, split)
    counts = {s: dict(patients=len({p["subject_id"] for p in pairs if p["split"] == s}),
                       targets=sum(p["split"] == s for p in pairs)) for s in ("train", "validation")}
    output.mkdir(parents=True)
    with (output / "candidate_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["subject_id", "split", "target_record_id", "history_record_id", "target_seconds",
                  "history_seconds", "target_weight", "history_policy"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(pairs)
    report = dict(protocol=PROTOCOL, completed=True, source_root=str(root), headers=len(headers),
        eligible_headers=len(records), counts=counts, rejected=rejected, records=records,
        patient_split=split, sources_sha256=fingerprints, training_started=False, training_ready=False,
        waveform_QC_completed=False, physical_calibration_checked=False, minimum_seconds=minimum,
        missing_training_prerequisites=["waveform QC and physical calibration", "pin public FM/P/G120/assets and frozen scales",
            "create post-QC matched-target training cache", "current Huawei training and full hybrid evaluator integration"],
        note="Candidate pairs only. Not a training manifest. Same-record segments and target synchronous leads forbidden.")
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("completed", "headers", "eligible_headers", "counts", "training_ready")}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    audit(a.root, a.output)


if __name__ == "__main__":
    main()
