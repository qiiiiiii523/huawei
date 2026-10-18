from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from baselines.B5_best.common import read, sha, state_sha, csv_write
from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_hash():
    digest = hashlib.sha256()
    for folder in ("baselines/B5_public_fm", "baselines/B5", "baselines/B5_best", "ecg12gen"):
        for path in sorted((ROOT / folder).glob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def fresh_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {path}; put tee logs beside it")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def training_lock(output):
    """OS-held lock is automatically released on process death; no stale PID lock cleanup."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".training.lock").open("a+b")
    handle.seek(0)
    try:
        import fcntl
    except ImportError:
        import msvcrt
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        acquire = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        release = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
    try:
        acquire()
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another training process is using this group output: " + str(output)) from exc
    try:
        yield
    finally:
        handle.seek(0)
        release()
        handle.close()


def load_study(path, verify_data=True):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("No completed study.json: prepare must finish successfully before check/train. "
                                "Inspect " + str(path.parent / "public/errors.json"))
    study = read(path)
    if study.get("protocol") != PROTOCOL or study.get("completed") is not True:
        raise ValueError("A completed PublicFM study.json is required")
    if study["implementation_sha256"] != code_hash():
        raise ValueError("Experiment implementation changed; restore the study's code or prepare a new study")
    for name, expected in study["files"].items():
        if sha(path.parent / name) != expected:
            raise ValueError("Frozen study artifact changed: " + name)
    if verify_data:
        for name, check in study["huawei_sources"].items():
            print("Checking Huawei data: " + name, flush=True)
            if sha(check["path"]) != check["sha256"]:
                raise ValueError("Huawei data/QC/demographics changed: " + name)
        for row in read(path.parent / "public/manifest.json"):
            if sha(path.parent / "public" / row["cache"]) != row["cache_sha256"]:
                raise ValueError("Public waveform cache changed: " + row["key"])
    return path, study
