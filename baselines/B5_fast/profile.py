"""The selected FM bytes and frozen fast extraction/sampling contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "configs" / "b5_fast.json"
SOURCE_BUNDLE_SHA = "990f5d95312dba0ab3e4bde416ee15d3b464b902fc63601aab1f852d9881f6f8"
SOURCE_FM_SHA = "74a5e401233c1593b5d7b22da48510a5539e0c65befec5e3daa270809d7e2603"
FM_STATE_SHA = "990e92e30f5255046c6aa73ab55f1ef4e22fc814a05f4fc92c14457bd6d35651"
CONTRACT_SHA = "bae0a40eda82bea1ee295552b0f956cc190f438df971074aa2992a1e50485432"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_profile(path=DEFAULT_PROFILE):
    value = read(path)
    # A moved checkpoint is permitted; model/sampling/decomposition cannot silently change.
    frozen = {k: v for k, v in value.items() if k != "checkpoint"}
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != CONTRACT_SHA or not isinstance(value.get("checkpoint"), str) or not value["checkpoint"]:
        raise ValueError("Use the frozen B5-fast model/sampling/decomposition profile")
    if (value["protocol"] != PROTOCOL or value["source_bundle_sha256"] != SOURCE_BUNDLE_SHA
            or value["source_fm_checkpoint_sha256"] != SOURCE_FM_SHA or value["fm_state_sha256"] != FM_STATE_SHA):
        raise ValueError("B5-fast source identity mismatch")
    return value


def contract(profile):
    return {k: v for k, v in profile.items() if k != "checkpoint"}


def resolve_checkpoint(profile, override=None):
    path = Path(override or profile["checkpoint"])
    return path if path.is_absolute() else ROOT / path
