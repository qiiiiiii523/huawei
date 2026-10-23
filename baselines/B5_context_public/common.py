import hashlib
from pathlib import Path
import torch
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.context_d6_v6 import ContextD6V6UNet, load_b5_weights
from baselines.B5_best.common import sha
from . import PROTOCOL
from .sampling import SAMPLING


def code_sha():
    root = Path(__file__).resolve().parents[1]
    files = list(Path(__file__).parent.glob("*.py"))
    for p in ("B5", "B5_best", "B5_best_context", "B5_slow120", "context_d6_v2"):
        files.extend((root / p).glob("*.py"))
    files.extend(root.glob("context_d6_v*.py"))
    d = hashlib.sha256()
    for f in sorted(set(files)):
        d.update(str(f.relative_to(root)).encode()); d.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return d.hexdigest()


def create_model(study, arm, device, state=None):
    cfg = ModelConfig.from_dict(study["model_config"])
    model = (B5UNet(cfg) if arm == "control" else ContextD6V6UNet(cfg))
    if state is None:
        cp = torch.load(study["public_checkpoint"], map_location="cpu", weights_only=True)
        if arm == "control": model.load_state_dict(cp["ema"], strict=True)
        else: load_b5_weights(model, cp["ema"])
    else: model.load_state_dict(state, strict=True)
    return model.to(device)


def load_candidate(path, study, root, device):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (cp.get("protocol") != PROTOCOL or cp.get("study_sha256") != sha(Path(root) / "study.json")
            or cp.get("sampling") != SAMPLING):
        raise ValueError("Wrong study or sampling in candidate")
    return create_model(study, cp["arm"], device, cp["ema"]).eval(), cp
