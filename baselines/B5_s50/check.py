"""Synthetic zero-init ODE and adapter-gradient checks, never optimizer updates."""
import argparse
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import load_checkpoint
from baselines.B5.config import ModelConfig
from baselines.B5.flow import sample
from baselines.B5.losses import flow_loss
from baselines.B5.runtime import seed_all
from . import INPUTS,SLOW_GROUPS,SCORE_PROTOCOL
from .common import read,write,state_hash,sha
from .model import make_model,BoundModel
from .prepare import load_shared
from .slow import slow_window


def check(args):
    report=Path(args.report)
    if report.exists(): raise FileExistsError('Preflight report already exists; reuse it')
    shared,info=load_shared(args.shared)
    checkpoint=load_checkpoint(info['fm_checkpoint'])
    seed_all(42,True)
    cfg=ModelConfig.from_dict(info['model_config'])
    a=make_model(cfg,'control',checkpoint['ema']).eval()
    with np.load(Path(info['prepared'])/'train.npz',allow_pickle=False) as pack:
        inputs={key:torch.from_numpy(pack[key][:1].copy()) for key in INPUTS}
    # Use a short synthetic waveform: no labels or original target curves enter.
    inputs['anchor']=torch.linspace(-1,1,256)[None,None]
    state=torch.randn(1,11,256);time=torch.tensor([.4])
    with torch.no_grad():
        va,_=BoundModel(a)(state,time,*[inputs[key] for key in INPUTS])
        pa=sample(BoundModel(a),inputs,['preflight'],seed=42,steps=2,solver='heun',samples=1)
    groups={};adapter_hashes=[]
    for group in SLOW_GROUPS:
        seed_all(int(info['training']['seed']),True)
        b=make_model(cfg,group,checkpoint['ema']).eval()
        adapter_hashes.append(state_hash(b.slow_adapter.state_dict()))
        native=np.load(shared/('train_'+group+'.npy'),mmap_mode='r')
        slow=torch.from_numpy(slow_window(native[0],0,256)[None]);native._mmap.close()
        before=state_hash(b.state_dict())
        with torch.no_grad():
            vb,_=BoundModel(b,slow)(state,time,*[inputs[key] for key in INPUTS])
            pb=sample(BoundModel(b,slow),inputs,['preflight'],seed=42,steps=2,solver='heun',samples=1)
        if not torch.equal(va,vb) or not torch.equal(pa,pb): raise AssertionError('Zero residual changed original B5: '+group)
        batch={**inputs,'target':torch.randn(1,12,256),'quality_mask':torch.ones(1,12,dtype=torch.bool)}
        b.train()
        loss,_=flow_loss(BoundModel(b,slow),batch,info['loss'],torch.tensor(info['scales']['d12']))
        loss.backward()  # Gradient check only, no optimizer is constructed.
        grad=float(b.slow_adapter[-1].weight.grad.norm())
        if not np.isfinite(grad) or grad<=0: raise AssertionError('New projection receives no gradient: '+group)
        if state_hash(b.state_dict())!=before: raise AssertionError('Preflight changed weights')
        groups[group]={'velocity_max_abs_difference':float((va-vb).abs().max()),'ode_max_abs_difference':float((pa-pb).abs().max()),
            'adapter_projection_gradient_norm':grad,'parameters':sum(p.numel() for p in b.parameters()),
            'slow_source_state_sha256':info['slow_sources'][group]['state_sha256']}
    if len(set(adapter_hashes))!=1: raise AssertionError('S50/S50-F adapters do not share initialization')
    write(report,{'ready':True,'training_started':False,'optimizer_updates':0,
        'shared_sha256':sha(shared/'shared.json'),'score_protocol':SCORE_PROTOCOL,'groups':groups,
        'source_ema_same':True,'zero_projection':True,'experimental_adapter_same_initialization':True,
        'control_parameters':sum(p.numel() for p in a.parameters()),
        'frozen_slow_condition':True,'slow_target_used_as_condition':False,'test_length':256,'ode_test_steps':2,'ode_test_samples':1,
        'note':'Synthetic CPU function check, not an ECG score. Full Heun16/K16 epoch-0 validation still runs in each group.'})
    print(read(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared',required=True)
    parser.add_argument('--report',required=True)
    check(parser.parse_args())


if __name__=='__main__': main()
