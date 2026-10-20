"""Fast Huber first; optional fixed-P combination PCC is a separate experiment."""
import torch
from ecg12gen.losses import masked_huber_loss, masked_pcc_loss
from .model import highpass


def direct_objective(model, batch, weights):
    q = model(**batch["inputs"])
    full = q.transpose(0, 1).reshape(1, 11, -1)
    start = batch["centre"] * 5000
    predicted = highpass(full)[..., start:start + 5000]
    mask = batch["mask"][:, 1:].bool()
    if not mask.any():
        raise ValueError("No reliable target leads")
    safe_pred = torch.where(mask[..., None], predicted, 0.)
    target = torch.where(mask[..., None], batch["fast_target"], 0.)
    huber = masked_huber_loss(safe_pred, target, mask, weights["huber_delta"])
    pcc = predicted.new_zeros(())
    if weights["combo_pcc"]:
        combined = torch.where(mask[..., None], predicted + .75 * batch["slow"], 0.)
        truth = torch.where(mask[..., None], batch["target"][:, 1:], 0.)
        pcc = masked_pcc_loss(combined, truth, mask)
    total = huber + weights["combo_pcc"] * pcc
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite directfast loss")
    return total, {"fast_huber": huber, "combo_pcc": pcc}
