"""Local-only PTB-XL records500 adapter; never downloads or reads diagnostic conditions."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ecg12gen.contracts import D12_LEADS
from .metadata import encode_demographics, read_csv


def locate_ptbxl(data_root: str | Path, explicit_root: str | Path | None = None) -> Path:
    root = Path(explicit_root or data_root).resolve()
    if (root / "ptbxl_database.csv").is_file():
        return root
    candidates = []
    for pattern in ("*/ptbxl_database.csv", "*/*/ptbxl_database.csv", "*/*/*/ptbxl_database.csv"):
        candidates.extend(root.glob(pattern))
    candidates = sorted({candidate.parent.resolve() for candidate in candidates})
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("Several PTB-XL versions found; set paths.ptbxl_root explicitly")
    raise FileNotFoundError(f"PTB-XL not ready in {root}: need ptbxl_database.csv and records500 .hea/.dat files. "
                            "records100 alone is insufficient; no automatic download is performed.")


def lead_order(names: list[str]) -> list[int]:
    normalized = [str(name).strip().upper() for name in names]
    expected = [name.upper() for name in D12_LEADS]
    if len(normalized) != 12 or len(set(normalized)) != 12 or set(normalized) != set(expected):
        raise ValueError(f"Expected all 12 distinct standard leads, got {names}")
    return [normalized.index(name) for name in expected]


def physical_to_uV(values: np.ndarray, units: list[str], names: list[str], fs: float) -> np.ndarray:
    if float(fs) != 500.0 or values.shape != (5000, 12):
        raise ValueError("B5 public adapter requires native 500 Hz, complete 10-second 12-lead records")
    factors = {"v": 1e6, "mv": 1e3, "uv": 1., "μv": 1., "µv": 1.}
    try:
        scaling = np.asarray([factors[str(unit).strip().lower()] for unit in units], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Unknown WFDB voltage unit: {exc}") from exc
    if scaling.shape != (12,):
        raise ValueError("WFDB units must cover every channel")
    raw = (np.asarray(values, dtype=np.float64) * scaling[None, :])[:, lead_order(names)].T
    if not np.isfinite(raw).all():
        raise ValueError("Public waveform contains NaN/Inf; fix or explicitly audit before use")
    return raw.astype(np.float32)


class PTBXLDataset:
    FOLDS = {"train": set(range(1, 9)), "validation": {9}, "test": {10}}

    def __init__(self, root: str | Path, split: str, preprocessor: Any,
                 check_files: bool = True) -> None:
        self.root = Path(root).resolve()
        if split not in self.FOLDS:
            raise ValueError("Unknown PTB-XL split")
        self.split, self.preprocessor = split, preprocessor
        all_rows = read_csv(self.root / "ptbxl_database.csv")
        if not all_rows:
            raise ValueError("Empty PTB-XL metadata")
        required = {"ecg_id", "patient_id", "strat_fold", "filename_hr"}
        if not required.issubset(all_rows[0]):
            raise ValueError("PTB-XL CSV must contain ecg_id, patient_id, strat_fold, filename_hr")
        patients: dict[str, int] = {}
        ecg_ids = set()
        for row in all_rows:
            fold = int(row["strat_fold"])
            if fold not in range(1, 11) or not row["patient_id"].strip():
                raise ValueError("Invalid fold/patient identity")
            patient = row["patient_id"]
            if patient in patients and patients[patient] != fold:
                raise ValueError("PTB-XL patient crosses fold boundaries")
            patients[patient] = fold
            if row["ecg_id"] in ecg_ids:
                raise ValueError("Duplicate PTB-XL ecg_id")
            ecg_ids.add(row["ecg_id"])
        self.rows = [row for row in all_rows if int(row["strat_fold"]) in self.FOLDS[split]]
        if not self.rows:
            raise ValueError(f"No records in PTB-XL {split} folds")
        if check_files:
            report = self.file_audit()
            if report["missing_count"]:
                raise FileNotFoundError(f"PTB-XL download incomplete: {report['missing_count']} records missing files "
                                        f"in {split}; examples={report['missing_examples']}. No partial training is allowed.")

    def record_path(self, row: dict[str, str]) -> Path:
        relative = Path(row["filename_hr"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.root) or "records500" not in relative.parts:
            raise ValueError("filename_hr must be a relative records500 path within PTB-XL root")
        return path

    def file_audit(self) -> dict[str, Any]:
        missing = []
        truncated = 0
        for row in self.rows:
            path = self.record_path(row)
            if not path.with_suffix(".hea").is_file() or not path.with_suffix(".dat").is_file():
                missing.append(row["filename_hr"])
            elif path.with_suffix(".dat").stat().st_size < 5000 * 12 * 2:
                # Official records500 uses 16-bit storage; a still-growing .dat is not ready.
                missing.append(row["filename_hr"])
                truncated += 1
        return {"split": self.split, "records": len(self.rows), "patients": len({r['patient_id'] for r in self.rows}),
                "missing_count": len(missing), "truncated_count": truncated, "missing_examples": missing[:5]}

    @property
    def manifest_digest(self) -> str:
        payload = "\n".join("|".join(str(r.get(key, "")) for key in
                            ("ecg_id", "patient_id", "strat_fold", "filename_hr", "age", "sex", "height", "weight"))
                            for r in self.rows)
        return hashlib.sha256(payload.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import wfdb
        row = self.rows[index]
        record = wfdb.rdrecord(str(self.record_path(row)), physical=True)
        raw = physical_to_uV(record.p_signal, record.units, record.sig_name, record.fs)
        quality = np.ptp(raw, axis=1) > 0
        if not quality[0] or not quality[1:].any():
            raise ValueError(f"Public record {row['ecg_id']} lacks usable I/missing leads")
        record_id = f"ptbxl:{row['ecg_id']}"
        target = self.preprocessor.transform_d12_target(raw).model_signal
        anchor = self.preprocessor.transform_window(raw[:1], "ecg_machine_i").model_signal
        metadata = {"target_record_id": record_id, "pair_id": record_id,
                    "subject_id": f"ptbxl:{row['patient_id']}", "start_sample_500hz": "0", "expected_window_count": "1"}
        return {"anchor": anchor, "target": target, "target_uV": raw.copy(), "quality_mask": quality,
                **encode_demographics(row, "ptbxl"), "key": record_id + ":0", "evaluation_metadata": metadata}
