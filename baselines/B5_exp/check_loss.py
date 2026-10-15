"""C/M/F shared initialization and TRAIN-only loss calibration; no updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baselines.B5.runtime import seed_all, device_from_name
from . import PROTOCOL
from .core import SlowCNN, Records, load_prepared, generate, device_batch, sha, json_write, objective
from .train import state_hash
from .losses import LOSS_PROTOCOL, auxiliary_terms


def implementation_hashes():
    package = Path(__file__).resolve().parent
    paths = [package / name for name in ('core.py', 'losses.py', 'train_loss.py', 'check_loss.py',
                                         'loss_metrics.py', 'evaluate.py', 'compare_loss.py')]
    paths += [package.parent / 'B5/runtime.py', package.parents[1] / 'ecg12gen/evaluate.py']
    paths += [package / name for name in ('context_audit.py', 'legacy_context.py', 'check_s50.py', 'compare_context.py')]
    paths += [package.parents[1] / 'configs/experiments/b5_exp_s50_reference.json']
    # Git may check out CRLF on Windows and LF on Linux; these are the same code.
    return {str(path.relative_to(package.parents[1])).replace('\\', '/'):
            hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in paths}


def initializer(path, prepared):
    source = torch.load(path, map_location='cpu', weights_only=True)
    config = source['config']
    if source.get('protocol') != PROTOCOL or config.get('context_seconds') != 30:
        raise ValueError('Initialization must be an S30 slow predictor, not an FM checkpoint')
    if config.get('prepared_sha256') != sha(Path(prepared) / 'prepared.json'):
        raise ValueError('S30 initializer and prepared data disagree')
    if config.get('model_channels', 48) != 48:
        raise ValueError('Expected original 48-channel SlowCNN')
    model = SlowCNN()
    model.load_state_dict(source['model'], strict=True)
    return model, source


def check(args):
    output = Path(args.report).resolve()
    if output.exists():
        raise FileExistsError('Shared calibration already exists; reuse it for all three groups')
    prepared, info = load_prepared(args.prepared)
    if not 0 < args.gradient_fraction <= 1:
        raise ValueError('Gradient fraction must be in (0,1]')
    device = device_from_name(args.device)
    seed_all(42, True)
    model, source = initializer(args.init_checkpoint, prepared)
    model.to(device).eval()
    initial_hash = state_hash(model.state_dict())
    dataset = Records(prepared / 'train.npz')
    if len(dataset) < 2:
        raise ValueError('Calibration requires two TRAIN records')
    batch = {key: torch.stack([dataset[i][key] for i in (0, 1)]) for key in dataset[0]}
    batch = device_batch(batch, device)
    inputs = {k: batch[k] for k in ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')}
    with torch.no_grad():
        prediction = generate(model, inputs, 30)
    # Probe gradients with respect to generated curves, NOT optimizer updates.
    prediction = prediction.detach().requires_grad_(True)
    base, parts = objective(prediction, batch['target'], batch['mask'])
    mean, residual = auxiliary_terms(prediction, batch['target'], batch['mask'], args.residual_width)
    terms = {'base': base, 'mean': mean, 'residual': residual}
    norms = {name: float(torch.linalg.vector_norm(torch.autograd.grad(value, prediction, retain_graph=True)[0]))
             for name, value in terms.items()}
    if not all(np.isfinite(v) and v > 1e-12 for v in norms.values()):
        raise ValueError('Degenerate calibration gradients; inspect training data/initializer before training')
    raw_weights = {key: args.gradient_fraction * norms['base'] / norms[key] for key in ('mean', 'residual')}
    weights = {key: float(np.clip(value, .01, 10.)) for key, value in raw_weights.items()}
    report = {'protocol': LOSS_PROTOCOL, 'ready': True, 'training_started': False, 'optimizer_updates': 0,
              'prepared': str(prepared), 'prepared_sha256': sha(prepared / 'prepared.json'),
              'init_checkpoint': str(Path(args.init_checkpoint).resolve()), 'init_checkpoint_sha256': sha(args.init_checkpoint),
              'initial_weights_sha256': initial_hash, 'source_epoch': int(source['epoch']), 'context_seconds': 30,
              'implementation_sha256': implementation_hashes(),
              'parameters': sum(p.numel() for p in model.parameters()), 'prediction_shape': list(prediction.shape),
              'record_indices': [0, 1], 'training_target_ids': [r['target_record_id'] for r in
                  json.loads((prepared / 'records.json').read_text(encoding='utf-8'))['train'][:2]],
              'calibration_split': 'train_only_first_two_records', 'gradient_reference': 'prediction_not_parameter_gradient',
              'gradient_fraction': args.gradient_fraction, 'raw_weights': raw_weights, 'weights': weights,
              'weight_clipped': {k: weights[k] != raw_weights[k] for k in weights},
              'gradient_norms': norms, 'loss_values': {k: float(v.detach()) for k, v in terms.items()},
              'weighted_gradient_ratios': {k: weights[k] * norms[k] / norms['base'] for k in weights},
              'residual_width_native': args.residual_width, 'native_rate_hz': 50,
              'residual_operator': 'full_record_centered_edge_repeat_average', 'target_means_removed': False,
              'score_protocol': info['applied_score_protocol'], 'note': 'Exploratory fixed weights, not validated optimal weights. No validation-driven tuning.'}
    if state_hash(model.state_dict()) != initial_hash:
        raise AssertionError('Preflight changed model weights')
    output.parent.mkdir(parents=True, exist_ok=True)
    json_write(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gradient-fraction', type=float, default=.1)
    parser.add_argument('--residual-width', type=int, default=51)
    check(parser.parse_args())


if __name__ == '__main__':
    main()
