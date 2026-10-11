"""Strict UTF-8/GB18030 CSV decoding without rewriting organizer files."""
from __future__ import annotations
import csv
import io
from pathlib import Path


def read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], str]:
    blob = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = blob.decode(encoding)
        except UnicodeDecodeError:
            continue
        return list(csv.DictReader(io.StringIO(text))), encoding
    raise UnicodeError(f"CSV is neither UTF-8 nor GB18030: {path}")
