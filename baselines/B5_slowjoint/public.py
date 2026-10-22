"""PTB Diagnostic: original physical calibration/QC, real >=120s, patient split."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from baselines.B5.metadata import encode_demographics
from baselines.B5_best import LEADS
from baselines.B5_best.common import read, write, sha, csv_write
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.combine import slow_curve
from baselines.B5_slow120.model import visible_inputs
from .common import safe_file
from . import PROTOCOL

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


def pack_record(wave, mask, meta, scale):
    if wave.shape[0] != 12 or wave.shape[-1] < 60000:
        raise ValueError("Public joint training needs >=120 real seconds; no padding")
    # Match the Huawei 120s label boundary definition exactly.
    raw = np.asarray(wave[:, :60000], np.float32)
    anchor = raw[:1].copy()
    return {"anchor": (anchor.reshape(1, 6000, 10).mean(-1) / scale[0]).astype(np.float32),
            "slow_target": (slow_curve(raw[1:]).reshape(11, 6000, 10).mean(-1) / scale[1:, None]).astype(np.float32),
            "mask": np.asarray(mask, bool), **visible_inputs(anchor, scale[0], meta)}


def prepare(root, bundle_path, output, min_train=20, min_validation=5):
    import wfdb
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new public cache directory")
    heads = sorted(root.glob("patient*/*.hea"))
    if not heads:
        raise FileNotFoundError("Expected PTB Diagnostic patient*/*.hea, not PTB-XL")
    # Same deterministic patient partition as the existing public-slow route,
    # BEFORE duration/QC selection; one patient never straddles splits.
    split = patient_split([p.parent.name for p in heads])
    bundle = load_bundle(bundle_path)
    scale = np.asarray(bundle["fm"]["scales"]["d12"], np.float32)
    output.mkdir(parents=True)
    entries, audit, sources, seen = {"train": [], "validation": []}, [], {}, set()
    for number, path in enumerate(heads, 1):
        head = wfdb.rdheader(str(path.with_suffix("")))
        files = [safe_file(root, (path.parent / name).relative_to(root)) for name in {path.name, *head.file_name}]
        if any(not f.is_file() for f in files):
            raise FileNotFoundError("Complete the PTB download before preparing: " + str(path))
        subject = path.parent.name
        row = dict(record=path.relative_to(root).as_posix(), subject_id="PTB:" + subject,
                   split=split[subject], seconds=head.sig_len / head.fs, status="excluded", reason="")
        if row["seconds"] < 120:
            row["reason"] = "shorter_than_120_real_seconds"
        else:
            wave, mask, reasons, meta, _ = calibrated_record(path.with_suffix(""))
            row["quality_mask"] = "".join(str(int(v)) for v in mask)
            row["QC"] = "|".join(f"{lead}:{reason}" for lead, reason in zip(LEADS, reasons) if reason)
            if not mask[0] or not mask[1:].any():
                row["reason"] = "unusable_I_or_all_targets"
            else:
                digest = hashlib.sha256(np.ascontiguousarray(wave[:, :60000]).tobytes()).hexdigest()
                if digest in seen:
                    row["reason"] = "duplicate_waveform"
                else:
                    seen.add(digest)
                    relative = f"{split[subject]}/{subject}_{path.stem}.npz"
                    dest = safe_file(output, relative)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(dest, **pack_record(wave, mask, meta, scale))
                    entries[split[subject]].append(dict(file=relative, sha256=sha(dest), subject_id="PTB:" + subject,
                        target_record_id=f"PTB:{subject}:{path.stem}"))
                    sources.update({str(f): sha(f) for f in files})
                    row["status"] = "included"
        audit.append(row)
        print(f"PTB120 {number}/{len(heads)} {row['record']}: {row['status']} {row['reason']}", flush=True)
    columns = sorted(set().union(*(r.keys() for r in audit)))
    csv_write(output / "audit.csv", [{k: r.get(k, "") for k in columns} for r in audit])
    patients = {k: len({e["subject_id"] for e in v}) for k, v in entries.items()}
    report = dict(completed=False, headers=len(heads), patients=patients,
                  records={k: len(v) for k, v in entries.items()}, duration_policy="first real120s; shorter excluded",
                  target_means_removed=False, split_seed=42, sampling_input_hz=5, local_input_hz=50,
                  slow_label="full120s 501-point centered moving average then 10-point block mean")
    write(output / "audit.json", report)
    if patients["train"] < min_train or patients["validation"] < min_validation:
        raise ValueError("Too few eligible PTB120 patients; inspect audit.json/csv, do not train P")
    manifest = dict(protocol=PROTOCOL, completed=True, scales_uV=scale.tolist(), source_checks=sources,
                    patient_split=split, train=entries["train"], validation=entries["validation"])
    write(output / "public.json", manifest)
    write(output / "audit.json", report | {"completed": True})
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/home/qht/huawei/data/PTB_db")
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.root, a.bundle, a.output)


if __name__ == "__main__":
    main()


