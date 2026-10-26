"""Optional training-only endpoint-gradient audit; no parameter updates or tuning."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import default_collate
from .checkpoint import load_checkpoint
from .config import read_json
from .data import WindowDataset, read_manifest
from .losses import flow_loss
from .model import B7UNet
from .train import balanced_indices, move, validate_settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--init-checkpoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batches', type=int, default=2)
    a = p.parse_args()
    if Path(a.output).exists() or a.batches < 1:
        raise ValueError('Use a new audit output and positive batches')
    cfg = read_json(a.config); mc, lc, sc = validate_settings(cfg)
    manifest = read_manifest(a.manifest)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    model = B7UNet(mc).to(a.device)
    if a.init_checkpoint:
        ck = load_checkpoint(a.init_checkpoint)
        if ck['model_config'] != mc.to_dict() or list(ck['scales_uV']) != list(manifest['scales_uV']):
            # Checkpoints serialize tuple widths as tuples; config uses ModelConfig too.
            from .config import ModelConfig
            if ModelConfig.from_dict(ck['model_config']) != mc or list(ck['scales_uV']) != list(manifest['scales_uV']):
                raise ValueError('Audit checkpoint architecture/scales mismatch')
        model.load_state_dict(ck['ema'])
    dataset = WindowDataset(manifest, 10, 0., include_d12=True)
    captured = []
    hook = model.register_forward_hook(lambda m, args, output: captured.append(output))
    results = []
    for _ in range(a.batches):
        captured.clear()
        batch = move(default_collate([dataset[i] for i in balanced_indices(dataset, 2)]), a.device)
        total, parts = flow_loss(model, batch, lc)
        velocity = captured[-1]
        reference = parts['fm'] + lc.mae_weight * parts['mae'] + lc.pcc_weight * parts['pcc']
        ref_grad = torch.autograd.grad(reference, velocity, retain_graph=True)[0].norm()
        row = {'losses': {k: float(v.detach()) for k, v in parts.items()}, 'reference_velocity_grad_norm': float(ref_grad)}
        for name, weight in (('limb_wave', lc.limb_wave_weight), ('projection', lc.projection_weight)):
            if weight:
                grad = torch.autograd.grad(weight * parts[name], velocity, retain_graph=True)[0].norm()
                row[name + '_weighted_velocity_grad_ratio'] = float(grad / ref_grad.clamp_min(1e-12))
        results.append(row)
    hook.remove()
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'optimizer_updates': 0, 'loss_config': asdict(lc), 'batches': results,
        'scope': 'train-only, batch2, velocity-output gradients; not whole-parameter gradients; weights unchanged'}, indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
