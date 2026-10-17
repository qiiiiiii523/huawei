"""CPU PTB Diagnostic adapter: calibrated uV, anti-alias resampling, patient split, QC."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from baselines.B5.metadata import encode_demographics
from baselines.B5_best import LEADS
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, sha, write, csv_write
from . import PROTOCOL
from .common import empty_output, atomic_json, safe_file
from .core import slow_numpy

QC_PROTOCOL = "PTB-CMP-conservative-technical-QC-v1"


def patient_split(subjects, seed=42):
    subjects = sorted(set(subjects))
    if len(subjects) < 2:
        raise ValueError("Need at least two independent public patients")
    order = sorted(subjects, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    nval = max(1, round(.2 * len(order)))
    validation = set(order[:nval])
    return {s: "validation" if s in validation else "train" for s in subjects}


def longest_run(flags):
    padded = np.r_[False, np.asarray(flags, bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def calibrated_record(path):
    import wfdb
    from scipy.signal import resample_poly
    head = wfdb.rdheader(str(path))
    lookup = {name.lower(): i for i, name in enumerate(head.sig_name)}
    indices = [lookup[name.lower()] for name in LEADS]
    if float(head.fs) != 1000:
        raise ValueError("PTB adapter expects documented 1000Hz; investigate other sources")
    record = wfdb.rdrecord(str(path), channels=indices, physical=False)
    digital = np.asarray(record.d_signal, np.float64).T
    gains = np.asarray(record.adc_gain, float)
    baselines = np.asarray(record.baseline, float)
    factors = {"mv": 1000., "uv": 1., "µv": 1., "μv": 1., "v": 1e6}
    units = [str(u or "mV").lower() for u in record.units]
    if any(u not in factors for u in units) or np.any(gains <= 0):
        raise ValueError("Unknown voltage calibration")
    wave = (digital - baselines[:, None]) / gains[:, None]
    wave *= np.asarray([factors[u] for u in units])[:, None]
    mask, reasons = np.ones(12, bool), []
    for i, name in enumerate(LEADS):
        r = []
        if not np.isfinite(wave[i]).all():
            r.append("nonfinite")
        if np.isfinite(wave[i]).all():
            if np.ptp(wave[i]) < 1.:
                r.append("flat_under_1uV")
            if np.max(np.abs(wave[i])) > 50000:
                r.append("absolute_over_50mV")
            if longest_run(np.diff(digital[i]) == 0) >= 5000:
                r.append("constant_5s")
        resolution = int(record.adc_res[i] or 16)
        if not 8 <= resolution <= 32:
            raise ValueError("Unknown ADC resolution")
        center = int(record.adc_zero[i] or 0)
        rails = (digital[i] <= center - 2 ** (resolution - 1)) | (digital[i] >= center + 2 ** (resolution - 1) - 1)
        if longest_run(rails) >= 20:
            r.append("ADC_rail_20ms")
        mask[i] = not r
        reasons.append(";".join(r))
    # Invalid targets become finite placeholders ONLY under a false supervision mask.
    # Invalid I is never used. Healthy target voltages are never centered or calibrated to Y.
    clean = wave.copy()
    clean[~mask] = 0
    converted = resample_poly(clean, up=1, down=2, axis=-1).astype(np.float32)
    metadata = {}
    for comment in head.comments:
        if ":" in comment:
            key, value = comment.split(":", 1)
            if key.strip().lower() in ("age", "sex", "height", "weight"):
                metadata[key.strip().lower()] = value.strip()
    return converted, mask, reasons, encode_demographics(metadata), head


def pack_record(raw, mask, meta, scales):
    length = min(60000, raw.shape[-1] // 5000 * 5000)
    if length < 25000:
        raise ValueError("Need at least 50 real seconds")
    # Full available record is smoothed before taking the selected chronological segment.
    target = slow_numpy(raw[1:])[:, :length]
    n = length // 10
    return {"anchor": (raw[:1, :length].reshape(1, n, 10).mean(-1) / scales[0]).astype(np.float32),
            "slow_target": (target.reshape(11, n, 10).mean(-1) / scales[1:, None]).astype(np.float32),
            "mask": mask, **meta}


def prepare(root, bundle_path, output, min_train_patients=20, min_validation_patients=5):
    root = Path(root).resolve()
    headers = sorted(root.glob("patient*/*.hea"))
    if not headers:
        raise FileNotFoundError("No patient*/record.hea; pass the PTB-Diagnostic root, not its parent")
    bundle = load_bundle(bundle_path)
    scales = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    split = patient_split([p.parent.name for p in headers])
    output = empty_output(output)
    entries = {"train": [], "validation": []}
    audit, seen, source_checks = [], {}, {}
    for number, path in enumerate(headers, 1):
        import wfdb
        head_check = wfdb.rdheader(str(path.with_suffix("")))
        # Download still running/incomplete is not a quality exclusion. Stop before training manifests.
        for filename in {path.name, *head_check.file_name}:
            source = safe_file(root, (path.parent / filename).relative_to(root))
            if not source.is_file():
                raise FileNotFoundError("Finish the complete PTB download first: " + str(source))
        subject = path.parent.name
        row = {"record": path.relative_to(root).as_posix(), "subject_id": "PTB:" + subject,
               "split": split[subject], "status": "excluded", "reason": ""}
        try:
            wave, mask, reasons, meta, head = calibrated_record(path.with_suffix(""))
            row.update(original_seconds=head.sig_len / head.fs,
                       selected_seconds=min(120, (wave.shape[-1] // 5000) * 10),
                       quality_mask="".join(str(int(v)) for v in mask))
            if wave.shape[-1] < 25000:
                row["reason"] = "shorter_than_50s"
            elif not mask[0] or not mask[1:].any():
                row["reason"] = "unusable_I_or_all_targets"
            else:
                digest = hashlib.sha256(wave.tobytes() + mask.tobytes()).hexdigest()
                if digest in seen:
                    row["reason"] = "duplicate_waveform_of:" + seen[digest]
                else:
                    seen[digest] = row["record"]
                    checks = {str(safe_file(root, (path.parent / filename).relative_to(root))):
                              sha(safe_file(root, (path.parent / filename).relative_to(root)))
                              for filename in {path.name, *head.file_name}}
                    packed = pack_record(wave, mask, meta, scales)
                    relative = split[subject] + "/" + subject + "_" + path.stem + ".npz"
                    destination = safe_file(output, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(destination, **packed)
                    entries[split[subject]].append({"file": relative, "sha256": sha(destination),
                        "subject_id": "PTB:" + subject, "target_record_id": "PTB:" + subject + ":" + path.stem,
                        "native_samples": packed["anchor"].shape[-1], "raw_wave_sha256": digest})
                    row["status"] = "included"
                    source_checks.update(checks)
            for name, reason in zip(LEADS, reasons):
                row["QC_" + name] = reason
        except (OSError, ValueError, KeyError, IndexError) as exc:
            row["reason"] = type(exc).__name__ + ": " + str(exc)
        audit.append(row)
        print(f"PTB {number}/{len(headers)}: {row['record']} {row['status']} {row['reason']}", flush=True)
    columns = sorted(set().union(*(row.keys() for row in audit)))
    csv_write(output / "audit.csv", [{k: row.get(k, "") for k in columns} for row in audit])
    patients = {part: len({e["subject_id"] for e in values}) for part, values in entries.items()}
    report = {"completed": False, "patients": patients, "included_records": {k: len(v) for k, v in entries.items()},
              "headers": len(headers), "excluded_records": sum(r["status"] != "included" for r in audit),
              "duration_policy": "real 50..120s, 10s multiple; no synthetic 120s padding",
              "QC_protocol": QC_PROTOCOL, "QC_thresholds": {"flat_uV": 1, "max_abs_uV": 50000,
                  "constant_seconds": 5, "ADC_rail_seconds": .02},
              "target_means_removed": False}
    write(output / "audit.json", report)
    if patients["train"] < min_train_patients or patients["validation"] < min_validation_patients:
        raise ValueError("Too few usable patients; inspect audit.csv. No completed manifest was written")
    import scipy
    import wfdb
    manifest = {"protocol": PROTOCOL, "completed": True, "bundle_scales_sha256": bundle["fm"]["scales_sha256"],
                "source_root": str(root), "source_checks": source_checks, "patient_split": split,
                "split_seed": 42, "train": entries["train"], "validation": entries["validation"],
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "wfdb": wfdb.__version__},
                "processing": report}
    atomic_json(output / "public.json", manifest)
    write(output / "audit.json", report | {"completed": True})
    print({"completed": True, "training_started": False, "patients": patients}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/home/qht/huawei/data/PTB-Diagnostic")
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.root, a.bundle, a.output)


if __name__ == "__main__":
    main()
