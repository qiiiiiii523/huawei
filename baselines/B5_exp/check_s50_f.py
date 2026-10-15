"""Train-only loss-scale check and read-only reuse audit for S50-F."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5.runtime import seed_all, device_from_name
from .core import Records, load_prepared, sha, json_write, generate, device_batch, objective
from .check_loss import initializer, implementation_hashes
from .train import state_hash
from .losses import LOSS_PROTOCOL, auxiliary_terms
from .s50_f_audit import (PAIR_PROTOCOL, FIXED_F_WEIGHT, audit_control, canonical, verify_unchanged_code)


def check(args):
    output = Path(args.report).resolve()
    if output.exists():
        raise FileExistsError('S50-F preflight already exists; reuse it')
    prepared, info = load_prepared(args.prepared)
    current = implementation_hashes()
    verify_unchanged_code(current)
    seed_all(42, True)
    model, source = initializer(args.init_checkpoint, prepared)
    initial = state_hash(model.state_dict())
    proof, control = audit_control(args.control_run,sha(prepared/'prepared.json'),sha(args.init_checkpoint),initial,info['checkpoint_sha256'])
    data = Records(prepared/'train.npz')
    if len(data) < 2:
        raise ValueError('Two training records are required for loss-scale check')
    batch = {key:torch.stack([data[i][key] for i in (0,1)]) for key in data[0]}
    device = device_from_name(args.device)
    model.to(device).eval()
    batch = device_batch(batch,device)
    with torch.no_grad():
        prediction = generate(model,{k:batch[k] for k in ('anchor','numeric','sex','field_mask','age_topcoded')},50)
    prediction = prediction.detach().requires_grad_(True)
    base,_ = objective(prediction,batch['target'],batch['mask'])
    mean,residual = auxiliary_terms(prediction,batch['target'],batch['mask'],51)
    terms = {'base':base,'mean':mean,'residual':residual}
    norms = {k:float(torch.linalg.vector_norm(torch.autograd.grad(v,prediction,retain_graph=True)[0])) for k,v in terms.items()}
    if not all(np.isfinite(v) and v > 1e-12 for v in norms.values()):
        raise ValueError('Degenerate loss gradient; inspect training-only preflight')
    if state_hash(model.state_dict()) != initial:
        raise AssertionError('Preflight changed weights')
    value = {'protocol':LOSS_PROTOCOL,'ready':True,'training_started':False,'optimizer_updates':0,
        'prepared':str(prepared),'prepared_sha256':sha(prepared/'prepared.json'),
        'init_checkpoint':str(Path(args.init_checkpoint).resolve()),'init_checkpoint_sha256':sha(args.init_checkpoint),
        'initial_weights_sha256':initial,'source_epoch':int(source['epoch']),'context_seconds':50,
        'implementation_sha256':current,'calibration_split':'train_only_first_two_records',
        'weights':control['calibrated_weights'],'residual_width_native':51,'native_rate_hz':50,
        'parameters':sum(p.numel() for p in model.parameters()),'prediction_shape':list(prediction.shape),
        'loss_values':{k:float(v.detach()) for k,v in terms.items()},'gradient_norms':norms,
        'fixed_F_weight':FIXED_F_WEIGHT,'weighted_residual_gradient_ratio':FIXED_F_WEIGHT*norms['residual']/norms['base'],
        'weight_policy':'reuse_existing_F_weight_fixed_before_validation_no_retuning',
        'score_protocol':info['applied_score_protocol'],'target_means_removed':False,
        's50_f_pair':{'protocol':PAIR_PROTOCOL,'ready':True,'control_reference':proof,
            'canonical_control':canonical(control),'initial_evaluation':str(Path(args.control_run).resolve()/'initial_evaluation.json')},
        'note':'Only fixed-F activation changes; initialize original S30 best50, not S50 best20. Loss-scale check is not a performance result.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    json_write(output,value)
    print({k:value[k] for k in ('ready','training_started','optimizer_updates','parameters','prediction_shape','fixed_F_weight','weighted_residual_gradient_ratio')},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--init-checkpoint',required=True)
    parser.add_argument('--control-run',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--device',default='cpu')
    check(parser.parse_args())


if __name__=='__main__':
    main()
