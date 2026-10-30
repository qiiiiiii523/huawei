"""Deployment-only G120 + C3 inference for the official submission entry."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from baselines.B5.metadata import encode_demographics
from baselines.B5.runtime import seed_all
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, sha
from baselines.B5_best.predict import Predictor as BasePredictor
from baselines.B5_best_context.adapters import ContextRouter
from baselines.B5_best_context.body_features import feature_encoding, relation_features
from baselines.B5_best_context.body_history import history_policy, process_history
from baselines.B5_best_context.body_model import BodyAdapter
from baselines.B5_slow120 import PROTOCOL as SLOW_PROTOCOL
from baselines.B5_slow120.common import code_sha
from baselines.B5_slow120.model import SlowCorrection, correct_base, inputs, visible_inputs

from .body_scope import transform
from .profile import BASE_SHA, BODY_SHA, SAMPLING, check_payloads


PROTOCOL = "B5-final-g120-c3-submission-v1"
C3_PROTOCOL = "B5-C3-history-multitemplate-v1"


def _slow_source_sha(folder: Path) -> str:
    """Use the original Slow120 source fingerprint on staged source files."""
    digest = hashlib.sha256()
    for path in sorted(folder.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _hex(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"Invalid {name} SHA256 in submission.json")
    return value.lower()


@dataclass(frozen=True)
class Assets:
    base: Path
    original_body: Path
    g120: Path
    c3: Path
    study: Path
    study_payload: dict
    c3_payload: dict

    def report(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "base_checkpoint": str(self.base), "base_sha256": sha(self.base),
            "original_body_checkpoint": str(self.original_body), "original_body_sha256": sha(self.original_body),
            "g120_checkpoint": str(self.g120), "g120_sha256": sha(self.g120),
            "c3_checkpoint": str(self.c3), "c3_sha256": sha(self.c3),
            "slow120_study": str(self.study), "slow120_study_sha256": sha(self.study),
            "sampling": SAMPLING, "slow_beta": .75,
            "composition": "FM+P -> G120 slow residual -> task-specific machine FIR25 or body C3 five-limb projection",
            "training_started": False, "target_used_at_inference": False,
        }


def resolve_assets(root: str | Path, settings: dict) -> Assets:
    root = Path(root).resolve()
    paths = {name: _resolve(root, settings[name]) for name in
             ("base_checkpoint", "original_body_checkpoint", "g120_checkpoint", "c3_checkpoint", "slow120_study")}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing G120+C3 deployment asset {name}: {path}")
    expected = {
        "base_checkpoint": BASE_SHA,
        "original_body_checkpoint": BODY_SHA,
        "g120_checkpoint": _hex(settings.get("g120_sha256"), "g120"),
        "c3_checkpoint": _hex(settings.get("c3_sha256"), "C3"),
        "slow120_study": _hex(settings.get("slow120_study_sha256"), "slow120 study"),
    }
    for name, digest in expected.items():
        if sha(paths[name]) != digest:
            raise ValueError(f"Wrong {name} bytes; deployment assets must match the selected SHA256")

    study = read(paths["slow120_study"])
    deployed_source = root / "runtime_source" / "B5_slow120"
    deployed_code_sha = _slow_source_sha(deployed_source) if deployed_source.is_dir() else code_sha()
    if (study.get("protocol") != SLOW_PROTOCOL or study.get("completed") is not True
            or study.get("base_sha256") != BASE_SHA or study.get("body_sha256") != BODY_SHA
            or study.get("sampling") != SAMPLING or study.get("slow_beta") != .75
            or study.get("code_sha256") != deployed_code_sha):
        raise ValueError("Slow120 study does not match the selected G120 code/base/settings")
    model = study.get("model")
    if not isinstance(model, dict) or set(model) != {"channels", "layers", "heads"}:
        raise ValueError("Slow120 study has an unsupported model specification")

    c3 = torch.load(paths["c3_checkpoint"], map_location="cpu", weights_only=False)
    config = c3.get("config", {})
    if c3.get("protocol") != C3_PROTOCOL or config.get("representation") != "single" or not isinstance(c3.get("model"), dict):
        raise ValueError("Selected C3 checkpoint is not the pinned single-template C3 deployment model")
    return Assets(paths["base_checkpoint"], paths["original_body_checkpoint"], paths["g120_checkpoint"],
                  paths["c3_checkpoint"], paths["slow120_study"], study, c3)


def _batch(fields: dict, device: torch.device) -> dict:
    return {name: torch.from_numpy(np.asarray(fields[name])[None].copy()).to(device)
            for name in ("global_i", "local_i", "numeric", "sex", "field_mask", "age_topcoded")}


class Predictor:
    """One base FM draw, followed by G120 and the selected route-specific postprocess."""

    def __init__(self, device: torch.device, *, assets: Assets):
        self.device, self.assets = device, assets
        bundle = load_bundle(assets.base)
        check_payloads(bundle, torch.load(assets.original_body, map_location="cpu", weights_only=False))
        seed_all(42, True)
        self.base = BasePredictor(bundle, device)
        self.g120 = SlowCorrection("G120", **assets.study_payload["model"]).to(device)
        payload = torch.load(assets.g120, map_location="cpu", weights_only=True)
        if (payload.get("protocol") != SLOW_PROTOCOL or payload.get("group") != "G120"
                or payload.get("study_sha256") != sha(assets.study)
                or payload.get("base_sha256") != BASE_SHA or payload.get("body_sha256") != BODY_SHA):
            raise ValueError("G120 checkpoint does not belong to the packaged study/base")
        self.g120.load_state_dict(payload["model"], strict=True)
        self.g120.eval()

        # C3 preserves the original BodyIIAdapter architecture and B history
        # preprocessing, then replaces only its learned state with C3 weights.
        self.c3 = BodyAdapter(assets.original_body, BASE_SHA, device, expected_group="context")
        if assets.c3_payload["config"].get("history_preprocessing") != history_policy(self.c3.payload):
            raise ValueError("C3 and original body checkpoint use different visible-history preprocessing")
        self.c3.model.load_state_dict(assets.c3_payload["model"], strict=True)
        self.c3.model.eval()
        self.machine = ContextRouter(machine_mode="fir25")

    @torch.no_grad()
    def _g120_base(self, anchor: np.ndarray, metadata: dict, record_key: str, batch_size: int) -> np.ndarray:
        encoded = encode_demographics(metadata)
        base = self.base.predict_encoded(anchor, encoded, record_key, batch_size)
        fields = visible_inputs(anchor, self.base.scale[0], encoded)
        delta = self.g120(**inputs(_batch(fields, self.device)))[0].cpu().numpy()
        if delta.shape != (11, 6000) or not np.isfinite(delta).all():
            raise ValueError("G120 produced an invalid slow residual")
        return correct_base(base, delta, self.base.scale)

    def _body_c3(self, base: np.ndarray, anchor: np.ndarray, history):
        processed = process_history(history, history_policy(self.c3.payload))
        beat, stats, available = relation_features(processed, encoding=feature_encoding(self.c3.payload))
        corrected_ii, scales, enabled = self.c3.predict_ii_features(anchor, base[1], beat, stats, available)
        output = transform(base, anchor, corrected_ii, projection=True, scope="five")
        if not np.array_equal(output[[0, *range(6, 12)]], base[[0, *range(6, 12)]]):
            raise AssertionError("C3 changed I or chest leads")
        delta = corrected_ii.astype(np.float64) - base[1].astype(np.float64)
        return output, {
            "adapter": "body_C3_single_five_limb", "body_protocol": C3_PROTOCOL,
            "projection_applied": True, "body_scope": "five", "body_context_used": bool(enabled),
            "history_available": bool(available), "fallback": "none" if enabled else "learned_generic_C3",
            "max_II_correction_uV": float(np.max(np.abs(delta))),
            "rms_II_correction_uV": float(np.sqrt(np.mean(delta ** 2))),
            "zero_current_I_windows": int((scales == 0).sum()),
        }

    def predict_tasks(self, anchor_uV, metadata, record_key, *, machine_history=None, body_history=None, batch_size=8):
        base = self._g120_base(anchor_uV, metadata, record_key, batch_size)
        task1, task1_status = self.machine.apply(base, anchor_uV, task_id="task1", input_type="watch_ecg")
        machine, machine_status = self.machine.apply(base, anchor_uV, task_id="task2",
                                                      input_type="ecg_machine_d6", history=machine_history)
        body, body_status = self._body_c3(base, anchor_uV, body_history)
        return {"gen1": task1, "gen6_scale": body, "gen6_d6": machine}, {
            "gen1": {**task1_status, "g120_applied": True},
            "gen6_scale": {**body_status, "g120_applied": True},
            "gen6_d6": {**machine_status, "g120_applied": True},
        }
