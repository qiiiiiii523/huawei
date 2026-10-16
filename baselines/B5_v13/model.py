"""Full-resolution local residual, restricted to V1/V3 velocity channels."""
import torch
from torch import nn
from torch.nn import functional as F

from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet

KERNELS = (5, 15, 31)
MISSING_INDICES = (5, 7)  # Missing order: II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6.


class LocalV13(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv1d(channels, 16, 1)
        self.paths = nn.ModuleList(nn.Conv1d(16, 16, k, padding=k // 2) for k in KERNELS)
        self.project = nn.Conv1d(48, 2, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.register_buffer("selector", F.one_hot(torch.tensor(MISSING_INDICES), 11).float(),
                             persistent=False)

    def forward(self, features):
        reduced = F.silu(self.reduce(features))
        local = F.silu(torch.cat([path(reduced) for path in self.paths], dim=1))
        return torch.einsum("bkt,kc->bct", self.project(local), self.selector.to(local.dtype))


class B5V13(B5UNet):
    def __init__(self, config=ModelConfig()):
        super().__init__(config)
        self.local_v13 = LocalV13(config.channels[0])

    def output_velocity(self, features):
        return self.output(features) + self.local_v13(features)


def initialized_model(bundle, group, device):
    if group not in ("control", "v13"):
        raise ValueError("group must be control or v13")
    config = ModelConfig.from_dict(bundle["fm"]["model_config"])
    model = (B5V13(config) if group == "v13" else B5UNet(config))
    state = bundle["fm"]["ema"]
    if group == "control":
        model.load_state_dict(state, strict=True)
    else:
        incompatible = model.load_state_dict(state, strict=False)
        expected = {k for k in model.state_dict() if k.startswith("local_v13.")}
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise ValueError("Only the new local_v13 branch may be missing")
    return model.to(device)
