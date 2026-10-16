"""Target-free differentiable S50 inference and exact block-center interpolation."""
import torch

FIELDS = ("anchor", "numeric", "sex", "field_mask", "age_topcoded")


def generate(model, inputs):
    anchor = inputs["anchor"]
    if anchor.ndim != 3 or anchor.shape[1:] != (1, 6000) or not torch.isfinite(anchor).all():
        raise ValueError("Expected visible 120s I at 50Hz [B,1,6000]")
    padded = torch.cat((anchor[..., :1].expand(-1, -1, 1000), anchor,
                        anchor[..., -1:].expand(-1, -1, 1000)), dim=-1)
    windows = padded.unfold(-1, 2500, 500).permute(0, 2, 1, 3).reshape(-1, 1, 2500)
    metadata = {k: inputs[k].repeat_interleave(12, dim=0) for k in FIELDS[1:]}
    center = model(windows, **metadata)[..., 1000:1500]
    return center.reshape(len(anchor), 12, 11, 500).permute(0, 2, 1, 3).reshape(-1, 11, 6000)


def upsample(native):
    """np.interp at centers=10*i+4.5, with deterministic reduction-only backward."""
    if native.shape[-1] != 6000:
        raise ValueError("Expected whole 120s slow prediction at 50Hz")
    weights = (torch.arange(10, device=native.device, dtype=native.dtype) + .5) / 10
    middle = (native[..., :-1, None] * (1 - weights) + native[..., 1:, None] * weights)
    return torch.cat((native[..., :1].expand(*native.shape[:-1], 5),
                      middle.flatten(-2), native[..., -1:].expand(*native.shape[:-1], 5)), dim=-1)


def device_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def model_inputs(batch):
    # Explicit allow-list: neither raw targets nor fixed FM predictions enter the CNN.
    return {key: batch[key] for key in FIELDS}
