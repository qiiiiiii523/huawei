"""Initial lightweight U-Net; no device adapter or baseline head."""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv1d(cout, cout, 3, padding=1), nn.ReLU())


class LightweightUNet(nn.Module):
    def __init__(self, input_channels, widths=(16, 32, 64, 128)):
        super().__init__()
        if input_channels not in (1, 6) or len(widths) != 4:
            raise ValueError('Expected 1/6 inputs and four encoder widths')
        self.encoder = nn.ModuleList([block(a, b) for a, b in
            zip((input_channels,) + tuple(widths[:-1]), widths)])
        self.decoder = nn.ModuleList([block(widths[i + 1] + widths[i], widths[i])
                                      for i in (2, 1, 0)])
        self.output = nn.Conv1d(widths[0], 12, 1)

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.encoder):
            if i:
                x = F.max_pool1d(x, 2)
            x = layer(x)
            skips.append(x)
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = layer(torch.cat((x, skip), dim=1))
        return self.output(x)
