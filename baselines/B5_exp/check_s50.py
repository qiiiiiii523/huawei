"""No-training S50 preflight: strict S30 equivalence plus reuse of matching C run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from baselines.B5.runtime import seed_all, device_from_name
from .core import Records, load_prepared, sha, json_write, generate, contexts, device_batch
from .check_loss import initializer, implementation_hashes, check as check_calibration
from .train import state_hash
from .legacy_context import contexts_s30_reference, generate_s30_reference
from .context_audit import CONTEXT_PROTOCOL, audit_control, verify_model_and_fixed_code, canonical


def check(args):
    report = Path(args.report).resolve()
    if report.exists():
        raise FileExistsError('Preflight already exists; reuse it instead of overwriting')
    prepared, info = load_prepared(args.prepared)
    seed_all(42, True)
    current = implementation_hashes()
    verify_model_and_fixed_code(current)
    model, source = initializer(args.init_checkpoint, prepared)
    initial_state = state_hash(model.state_dict())
    control_reference, control_config = audit_control(args.control_run, sha(prepared / 'prepared.json'),
                                                      sha(args.init_checkpoint), initial_state)
    if control_reference['B5_checkpoint_sha256'] != info['checkpoint_sha256']:
        raise ValueError('Control used a different fixed B5 model')
    device = device_from_name(args.device)
    model.to(device).eval()
    sample = Records(prepared / 'train.npz')[0]
    inputs = {k: sample[k][None] for k in ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')}
    inputs = device_batch(inputs, device)
    shapes = {}
    with torch.no_grad():
        legacy = generate_s30_reference(model, inputs)
        s30 = generate(model, inputs, 30)
        for seconds in (10, 30):
            if not torch.equal(contexts(inputs['anchor'], seconds), contexts_s30_reference(inputs['anchor'], seconds)):
                raise AssertionError('Original context slicing changed')
        if not torch.equal(s30, legacy):
            raise AssertionError('Original S30 prediction changed')
        s50 = generate(model, inputs, 50)
        if s50.shape != (1, 11, 6000) or not torch.isfinite(s50).all():
            raise ValueError('Invalid S50 stitched prediction')
        for seconds in (10, 30, 50):
            shapes[str(seconds)] = list(contexts(inputs['anchor'], seconds).shape)
    if state_hash(model.state_dict()) != initial_state:
        raise AssertionError('Preflight changed model weights')
    report.parent.mkdir(parents=True, exist_ok=True)
    # Regenerate the ordinary calibration under the new implementation. It remains
    # S30 TRAIN-only calibration; both coefficients are inactive in this ablation.
    with tempfile.TemporaryDirectory(dir=report.parent) as temporary:
        calibrated = Path(temporary) / 'calibration.json'
        check_calibration(SimpleNamespace(prepared=str(prepared), init_checkpoint=args.init_checkpoint,
                          report=str(calibrated), device=args.device, gradient_fraction=.1, residual_width=51))
        value = json.loads(calibrated.read_text(encoding='utf-8'))
    value['s50_compatibility'] = {'protocol': CONTEXT_PROTOCOL, 'ready': True,
        'source_context_seconds': 30, 'compared_context_seconds': [30, 50],
        's30_bitwise_equal': True, 's30_max_abs_difference': float((s30 - legacy).abs().max()),
        'input_shapes': shapes, 'stitched_output_shape': list(s50.shape),
        'center_crop_s50_native': [1000, 1500], 'center_crop_s30_native': [500, 1000],
        'edge_padding': 'repeat endpoints of same 120s I record only',
        'control_reference': control_reference, 'control_canonical_config': canonical(control_config),
        'source_epoch': int(source['epoch']), 'parameters': sum(p.numel() for p in model.parameters()),
        'optimizer_updates': 0, 'training_started': False, 'target_used_at_inference': False,
        'note': 'Same weights need not yield same S30/S50 predictions: the visible I context differs. S30 equivalence checked on one training record; no full GPU training equivalence claim.'}
    json_write(report, value)
    print(json.dumps({'ready': True, 'training_started': False, 'optimizer_updates': 0,
                      's30_bitwise_equal': True, 's30_max_abs_difference': 0.,
                      'parameters': value['s50_compatibility']['parameters'], 'input_shapes': shapes,
                      'stitched_output_shape': list(s50.shape), 'control_reused': str(Path(args.control_run).resolve()),
                      'report': str(report)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--control-run', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', default='cpu')
    check(parser.parse_args())


if __name__ == '__main__':
    main()
