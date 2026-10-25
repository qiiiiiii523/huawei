"""Mask-safe seven-channel slow flow matching and endpoint MAE/PCC."""
import torch


def flow_loss(model, batch, config, noise_std, *, noise=None, time=None):
    target = batch['target']
    valid = batch['lead_mask'].bool()[:, :, None].expand_as(target)
    if not valid.flatten(1).any(1).all():
        raise ValueError('Every item must have supervised leads')
    target = torch.where(valid, target, 0.)
    if not torch.isfinite(target).all():
        raise ValueError('Nonfinite supervised target')
    noise = torch.randn_like(target) * target.new_tensor(noise_std)[None, :, None] if noise is None else noise
    time = torch.rand(len(target), device=target.device) if time is None else time
    t = time[:, None, None]
    state = torch.where(valid, (1-t)*noise + t*target, noise)
    velocity = torch.where(valid, target-noise, 0.)
    prediction = model(state, time, batch['anchor'], batch['metadata'])
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite model output')
    weights = target.new_tensor([config.ii_weight, 1, 1, 1, 1, 1, 1])[None, :, None]
    denominator = (valid * weights).sum()
    fm = (torch.where(valid, (prediction-velocity).square(), 0.)*weights).sum()/denominator
    endpoint = state + (1-t)*prediction
    safe = torch.where(valid, endpoint, 0.)
    mae = ((safe-target).abs()*weights).sum()/denominator
    pc, yc = safe-safe.mean(-1, keepdim=True), target-target.mean(-1, keepdim=True)
    pp, yy = pc.square().sum(-1), yc.square().sum(-1)
    eligible = batch['lead_mask'].bool() & (yy/target.shape[-1] > config.target_variance_floor)
    corr = ((pc*yc).sum(-1) / (pp*yy).clamp_min(1e-12).sqrt()).clamp(-1, 1)
    eligible_weights = eligible * weights[0, :, 0]
    pcc = ((1-corr)*eligible_weights).sum()/eligible_weights.sum().clamp_min(1)
    total = fm + config.mae_weight*mae + config.pcc_weight*pcc
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite loss')
    return total, dict(fm=fm, mae=mae, pcc=pcc)
