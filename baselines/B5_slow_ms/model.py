"""All inputs are visible I/demographics or the frozen predictor's own output."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from baselines.B5_best.slow import SlowCNN, SameConv, Block
from . import CORRECTION, SLOW_PROTOCOL


class Correction(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = copy.deepcopy(CORRECTION if config is None else config)
        if config != CORRECTION:
            raise ValueError('Unknown correction architecture')
        c = config['channels']
        self.sex = nn.Embedding(3, 4)
        self.metadata = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 32))
        # No true slow target, FM cache or target mean enters this branch.
        self.input = SameConv(12, c)
        self.paths = nn.ModuleList(nn.ModuleList(Block(c, d) for d in ds) for ds in config['dilations'])
        self.output = nn.Conv1d(3 * c, 11, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, anchor, base_prediction, numeric, sex, field_mask, age_topcoded):
        v = torch.cat((numeric, self.sex(sex), field_mask.float(), age_topcoded), dim=-1)
        meta = self.metadata(v)
        x = self.input(torch.cat((anchor, base_prediction), dim=1))
        features = []
        for path in self.paths:
            h = x
            for block in path:
                h = block(h, meta)
            features.append(F.silu(h))
        return self.output(torch.cat(features, dim=1))


class SlowMSResidual(nn.Module):
    def __init__(self, base=None, config=None):
        super().__init__()
        self.base = SlowCNN() if base is None else base
        self.base.requires_grad_(False).eval()
        self.correction = Correction(config)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, anchor, numeric, sex, field_mask, age_topcoded):
        if anchor.ndim != 3 or anchor.shape[1:] != (1, 2500):
            raise ValueError('Correction expects the same visible S50 window [B,1,2500]')
        metadata = dict(numeric=numeric, sex=sex, field_mask=field_mask, age_topcoded=age_topcoded)
        with torch.no_grad():
            base_prediction = self.base(anchor, **metadata)
        return base_prediction + self.correction(anchor, base_prediction, **metadata)


def load_residual(checkpoint, device):
    cfg = checkpoint.get('config', {})
    if (checkpoint.get('protocol') != SLOW_PROTOCOL or cfg.get('context_seconds') != 50 or
            cfg.get('group') != 'F' or cfg.get('model_channels') != 48 or cfg.get('correction') != CORRECTION):
        raise ValueError('Invalid portable multiscale residual slow model')
    model = SlowMSResidual(config=cfg['correction'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).requires_grad_(False).eval()
