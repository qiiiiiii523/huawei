"""Keep the original objective; constrain changes to the pretrained P tail."""
import torch
from torch.nn import functional as F
from baselines.B5_slow120.losses import objective as original_objective


def objective(delta, batch, scales, p_change=None, preservation_weight=0.):
    loss, parts = original_objective(delta, batch, scales)
    if p_change is not None:
        valid = batch["mask"][:, 1:, None].bool()
        change = torch.where(valid, p_change, 0.)
        penalty = F.huber_loss(change, torch.zeros_like(change), reduction="sum")
        penalty = penalty / (valid.sum().clamp_min(1) * change.shape[-1])
        loss = loss + preservation_weight * penalty
        parts["P_preservation"] = penalty
    return loss, parts
