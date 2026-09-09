"""M1-only contract and synthetic gradient checks; no competition arrays required."""
from __future__ import annotations
import inspect,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ecg12gen.contracts import ContractError,D12_LEADS
from ecg12gen.evaluate import evaluate_joint_anchor_predictions
from ecg12gen.losses import joint_anchor_sync_loss
from ecg12gen.m1_axial import M1AxialLeadTimeModel,AxialLeadTimeBlock,ARCHITECTURE_VERSION
from ecg12gen.m1_axial_train import validate_p0_checkpoint

def main():
    torch.set_num_threads(2); torch.manual_seed(42); anchor=torch.randn(1,1,5000); watch=torch.randn(1,1,5000); d6=torch.randn(1,6,5000); d6mask=torch.ones(1,6,dtype=torch.bool)
    p0=M1AxialLeadTimeModel(fusion_mode='none',task_id='task1').eval(); out,trace=p0(anchor,return_trace=True); assert out.shape==(1,12,5000); assert trace.z_shape==(1,12,250,128); assert trace.time_attention_calls==4 and trace.lead_attention_calls==4 and not trace.time_attention_is_causal; assert p0.watch_context_encoder is None; assert not torch.equal(out[:,:1],anchor)
    # Every fusion path must execute both axial attentions and backpropagate.
    for mode in ('film','gated_residual','film_gated_residual'):
        model=M1AxialLeadTimeModel(fusion_mode=mode,task_id='task1'); y=model(anchor,context=watch,context_source_type='watch_ecg'); assert y.shape==(1,12,5000) and model.last_trace.time_attention_calls==4 and model.last_trace.lead_attention_calls==4; y.square().mean().backward()
    model=M1AxialLeadTimeModel(fusion_mode='film',task_id='task2'); y=model(anchor,context=d6,context_source_type='machine/holter',context_lead_mask=d6mask) if False else model(anchor,context=d6,context_source_type='ecg_machine_d6',context_lead_mask=d6mask); y.square().mean().backward()
    for mode in ('gated_residual','film_gated_residual'):
        model=M1AxialLeadTimeModel(fusion_mode=mode,task_id='task2'); assert abs(float(torch.sigmoid(model.gate.bias).detach())-0.05)<1e-3; assert torch.count_nonzero(model.residual[-1].weight)==0 and torch.count_nonzero(model.residual[-1].bias)==0
    # none cannot receive context, and task2 cannot route an invented combined source.
    try: p0(anchor,context=watch,context_source_type='watch_ecg'); raise AssertionError('none read context')
    except ContractError: pass
    try: M1AxialLeadTimeModel(fusion_mode='film',task_id='task2')(anchor,context=d6,context_source_type='body_scale_d6+ecg_machine_d6',context_lead_mask=d6mask); raise AssertionError('combined source accepted')
    except ContractError: pass
    assert list(D12_LEADS)==['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'] and ARCHITECTURE_VERSION.startswith('M1-axial')
    pred=torch.zeros(2,12,5000).numpy(); target=torch.randn(2,12,5000).numpy(); raw_anchor=torch.randn(2,1,5000).numpy(); _,_,_,submit=evaluate_joint_anchor_predictions(pred,target,raw_anchor,'task1'); assert torch.equal(torch.from_numpy(submit[:,:1]),torch.from_numpy(raw_anchor))
    assert '--target' not in (ROOT/'scripts/predict_m1_axial.py').read_text(encoding='utf-8'); assert 'context_model' not in inspect.getsource(joint_anchor_sync_loss)
    try: validate_p0_checkpoint({'model':{}},p0,torch.ones(12)); raise AssertionError('legacy checkpoint accepted')
    except ValueError as error: assert 'metadata' in str(error)
    print('PASS: M1 Axial shape/trace; 4 fusion routes forward-backward; no causal time mask; canonical leads; none isolation; d6 exclusivity; gate/residual init; submit I exact; no target CLI; P0 metadata guard')
if __name__=='__main__':main()
