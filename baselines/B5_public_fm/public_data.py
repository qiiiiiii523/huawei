"""WFDB physical-voltage audit, frozen patient splits and read-only public caches."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from baselines.B5.metadata import read_csv, encode_demographics
from ecg12gen.contracts import D12_LEADS
from .common import read, write, sha, csv_write

QC_POLICY = {"minimum_ptp_uV": 1.0, "maximum_constant_run_samples": 2500,
             "digital_rail_fraction": 0.001, "minimum_good_leads": 6}


def lead_order(names):
    names = [str(v).strip().upper() for v in names]
    expected = [v.upper() for v in D12_LEADS]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
        raise ValueError("Expected all 12 distinct standard leads")
    return [names.index(v) for v in expected]


def physical_uV(values, units, names, fs):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (5000, 12) or float(fs) != 500:
        raise ValueError("Requires a complete native 500Hz/10s/12-lead record; no truncation or resampling")
    factors = {"v": 1e6, "mv": 1e3, "uv": 1., "μv": 1., "µv": 1.}
    if len(units) != 12:
        raise ValueError("Missing WFDB voltage units")
    try:
        scale = np.asarray([factors[str(u).strip().lower()] for u in units])
    except KeyError as exc:
        raise ValueError("Unknown voltage unit") from exc
    return ((values * scale)[..., lead_order(names)].T).astype(np.float32)


def longest_constant(x):
    starts = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1, len(x)]
    return int(np.diff(starts).max())


def quality_mask(raw, digital=None, adc_res=None):
    mask, reasons = [], []
    for i, x in enumerate(raw):
        why = []
        if not np.isfinite(x).all():
            why.append("nonfinite")
        else:
            if np.ptp(x) < QC_POLICY["minimum_ptp_uV"]:
                why.append("flat")
            if longest_constant(x) >= QC_POLICY["maximum_constant_run_samples"]:
                why.append("constant_run")
        if digital is not None and adc_res is not None:
            bits = int(adc_res[i] or 0)
            if 1 < bits <= 32:
                rail = 2 ** (bits - 1)
                d = digital[i]
                if np.mean((d <= -rail) | (d >= rail - 1)) >= QC_POLICY["digital_rail_fraction"]:
                    why.append("ADC_rail")
        mask.append(not why)
        reasons.append("|".join(why))
    return np.asarray(mask, bool), reasons


def waveform_digest(raw):
    # Exact physical float32 waveform identity, independent of file name/container.
    return hashlib.sha256(np.asarray(raw, dtype="<f4").tobytes()).hexdigest()


def chapman_split(patient, seed=42):
    bucket = int(hashlib.sha256(f"ChapmanSplit-v1|{seed}|{patient}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def contained(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Record path escapes its dataset root")
    return path


def descriptors(ptb_root, chapman_root, patient_map=None):
    ptb_root, chapman_root = Path(ptb_root).resolve(), Path(chapman_root).resolve()
    metadata = read_csv(ptb_root / "ptbxl_database.csv")
    seen, patients, result = set(), {}, []
    for row in metadata:
        patient, ecg = row["patient_id"].strip(), row["ecg_id"].strip()
        fold = int(row["strat_fold"])
        if not patient or ecg in seen or fold not in range(1, 11):
            raise ValueError("Invalid PTB-XL patient/fold/duplicate ID")
        if patients.setdefault(patient, fold) != fold:
            raise ValueError("PTB-XL patient crosses folds")
        seen.add(ecg)
        rel = Path(row["filename_hr"])
        if "records500" not in rel.parts:
            raise ValueError("PTB-XL must use records500, never records100")
        path = contained(ptb_root, rel)
        result.append(dict(source="ptbxl", key="ptbxl:" + ecg, patient="ptbxl:" + patient,
                           split="train" if fold <= 8 else "validation" if fold == 9 else "test",
                           path=str(path), demographics={k: row.get(k, "") for k in ("age", "sex", "height", "weight")}))
    if not result:
        raise ValueError("Empty PTB-XL metadata")
    mapping = {}
    if patient_map:
        for row in read_csv(patient_map):
            record, patient = row["record_id"].strip(), row["patient_id"].strip()
            if not patient or record in mapping:
                raise ValueError("Invalid/duplicate Chapman patient-map row")
            mapping[record] = patient
    heads = sorted(chapman_root.rglob("*.hea"))
    if not heads:
        raise FileNotFoundError("No Chapman .hea files; point to the extracted root containing WFDBRecords")
    if len({p.stem for p in heads}) != len(heads):
        raise ValueError("Duplicate Chapman record IDs across folders")
    # The official release describes patient ECGs but does not expose a cross-record
    # patient-ID table. Declare the assumption; never infer patient identity from age/sex.
    for head in heads:
        patient = mapping[head.stem] if patient_map else head.stem
        result.append(dict(source="chapman", key="chapman:" + head.stem, patient="chapman:" + patient,
                           split=chapman_split(patient), path=str(head.with_suffix("")), demographics=None))
    # Prefer held-out PTB identities when exact duplicates appear; never retain a train copy of held-out ECG.
    rank = {"test": 0, "validation": 1, "train": 2}
    return sorted(result, key=lambda r: (r["source"] != "ptbxl", rank[r["split"]], r["key"]))


def header_demographics(comments):
    values = {}
    for line in comments:
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip().lower() in {"age", "sex", "height", "weight"}:
                values[key.strip().lower()] = value.strip()
    return values


def audit_public(ptb_root, chapman_root, output, patient_map=None, expected_chapman_records=45152):
    import wfdb
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, audit, seen, errors = [], [], {}, []
    rows = descriptors(ptb_root, chapman_root, patient_map)
    count = sum(r["source"] == "chapman" for r in rows)
    if count != expected_chapman_records:
        raise ValueError(f"Expected {expected_chapman_records} Chapman headers, found {count}; finish/extract the full download first")
    for index, row in enumerate(rows):
        try:
            path = Path(row["path"])
            header = wfdb.rdheader(str(path))
            # Missing/incomplete files are fatal, not silently converted into QC exclusions.
            if header.n_sig != 12 or header.sig_len != 5000 or header.fs != 500:
                raise ValueError("Unexpected source shape/rate")
            order = lead_order(header.sig_name)
            source_files = [path.with_suffix(".hea")]
            for filename in set(header.file_name):
                file = contained(path.parent, filename)
                if not file.is_file() or file.stat().st_size == 0:
                    raise FileNotFoundError(file)
                source_files.append(file)
            physical = wfdb.rdrecord(str(path), physical=True)
            raw = physical_uV(physical.p_signal, physical.units, physical.sig_name, physical.fs)
            digital = wfdb.rdrecord(str(path), physical=False).d_signal[:, order].T
            mask, reasons = quality_mask(raw, digital, [header.adc_res[i] for i in order])
            identity = waveform_digest(raw)
            previous = seen.get(identity)
            status = "usable"
            if previous:
                status = "exact_duplicate"
            elif not mask[0] or mask.sum() < QC_POLICY["minimum_good_leads"]:
                status = "insufficient_good_leads"
            seen.setdefault(identity, row["key"])
            entry = {k: row[k] for k in ("source", "key", "patient", "split")}
            entry.update(status=status, duplicate_of=previous or "", waveform_sha256=identity,
                         good_leads=int(mask.sum()), lead_reasons=";".join(reasons))
            audit.append(entry)
            if status != "usable":
                continue
            cache = "waveforms/" + hashlib.sha256(row["key"].encode()).hexdigest() + ".npy"
            (output / "waveforms").mkdir(exist_ok=True)
            # Invalid leads have no supervision; zeroing here only makes preprocessing finite.
            clean = np.where(mask[:, None], raw, 0).astype(np.float32)
            np.save(output / cache, clean, allow_pickle=False)
            demo = row["demographics"] if row["source"] == "ptbxl" else header_demographics(header.comments)
            manifest.append({**row, "demographics": demo, "quality_mask": mask.tolist(),
                             "waveform_sha256": identity, "cache": cache, "cache_sha256": sha(output / cache),
                             "source_files": [{"path": str(p), "sha256": sha(p)} for p in source_files]})
        except (OSError, ValueError, IndexError, TypeError) as exc:
            errors.append({"key": row["key"], "error": str(exc)})
        if (index + 1) % 500 == 0:
            print(f"Public audit: {index + 1}/{len(rows)}", flush=True)
    if audit:
        csv_write(output / "audit.csv", audit)
    write(output / "errors.json", errors)
    if errors:
        raise ValueError(f"Public audit found {len(errors)} file/format errors; see {output / 'errors.json'}. Fix download, then use a new output.")
    counts = Counter((r["source"], r["split"]) for r in manifest)
    if any(counts[(s, split)] == 0 for s in ("ptbxl", "chapman") for split in ("train", "validation", "test")):
        raise ValueError("Both sources need nonempty patient-disjoint train/validation/test splits")
    write(output / "manifest.json", manifest)
    summary = {"completed": True, "QC_policy": QC_POLICY,
               "counts": {s: {split: {"records": counts[(s, split)],
                   "patients": len({r['patient'] for r in manifest if r['source'] == s and r['split'] == split})}
                   for split in ("train", "validation", "test")} for s in ("ptbxl", "chapman")},
               "excluded": dict(Counter(r["status"] for r in audit if r["status"] != "usable")),
               "chapman_patient_identity": "explicit_map" if patient_map else "official_patient_ECG_record_ID_assumption",
               "patient_map_sha256": sha(patient_map) if patient_map else None,
               "warning": "Basic signal QC only, not clinical adjudication; near-duplicates and hidden cross-dataset patient identity are not guaranteed detected"}
    write(output / "audit_summary.json", summary)
    return summary


class PublicDataset:
    def __init__(self, root, source, split, preprocessor):
        self.root, self.preprocessor = Path(root), preprocessor
        self.rows = [r for r in read(self.root / "manifest.json") if r["source"] == source and r["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty {source}/{split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = np.load(self.root / row["cache"], allow_pickle=False)
        if raw.shape != (12, 5000) or not np.isfinite(raw).all():
            raise ValueError("Invalid frozen public waveform")
        return {"anchor": self.preprocessor.transform_window(raw[:1], "ecg_machine_i").model_signal,
                "target": self.preprocessor.transform_d12_target(raw).model_signal,
                "quality_mask": np.asarray(row["quality_mask"], bool),
                **encode_demographics(row["demographics"], "ptbxl" if row["source"] == "ptbxl" else "huawei"),
                "key": row["key"] + ":0", "target_uV": raw,
                "evaluation_metadata": {"target_record_id": row["key"], "subject_id": row["patient"]}}
