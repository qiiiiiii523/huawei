from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baselines.B5.objective import file_sha256 as sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def code_hashes():
    repo = Path(__file__).resolve().parents[2]
    paths = list((repo/'baselines/B5_scratch').glob('*.py'))
    paths += [repo/'baselines/B5'/name for name in ('model.py','flow.py','losses.py','conditions.py','metadata.py','data.py','config.py','runtime.py','objective.py','checkpoint.py','selection.py')]
    paths += [repo/'ecg12gen'/name for name in ('evaluate.py','preprocessing.py','losses.py')]
    paths += [repo/'baselines/B5_best'/name for name in ('__init__.py','scoring.py','combine.py')]
    return {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(paths)}


def finite_json(value):
    if isinstance(value, dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [finite_json(v) for v in value]
    if isinstance(value, (np.floating,float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value
