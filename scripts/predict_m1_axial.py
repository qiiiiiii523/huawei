"""Formal M1 inference: explicit anchor/context inputs only; no hidden target argument."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ecg12gen.m1_axial import M1AxialLeadTimeModel,ARCHITECTURE_ID,ARCHITECTURE_VERSION,architecture_config_hash
from ecg12gen.preprocessing import ECGPreprocessor,PreprocessingConfig
from ecg12gen.m1_protocol import check_checkpoint_protocol

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',required=True); p.add_argument('--anchor-npy',required=True); p.add_argument('--task-id',choices=('task1','task2'),required=True); p.add_argument('--context-source-type',choices=('watch_ecg','body_scale_d6','ecg_machine_d6')); p.add_argument('--watch-context-npy'); p.add_argument('--d6-context-npy'); p.add_argument('--run-dir'); p.add_argument('--shuffled-context',action='store_true'); p.add_argument('--output-dir',required=True); p.add_argument('--context-baseline-npy',help='Visible complete-record context medians [C] or [N,C]'); p.add_argument('--context-valid-mask-npy'); p.add_argument('--copy-observed-i',action='store_true'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--device',default='cpu'); a=p.parse_args()
    if a.batch_size<1: p.error('batch-size must be positive')
    ck=torch.load(Path(a.checkpoint),map_location='cpu',weights_only=True)
    check_checkpoint_protocol(ck)
    if ck.get('architecture_version')!=ARCHITECTURE_VERSION: raise SystemExit('checkpoint architecture mismatch: old B3-like M1 checkpoints are not accepted')
    if ck.get('lead_order')!=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']: raise SystemExit('checkpoint lead order is incompatible')
    if ck.get('architecture_id')!=ARCHITECTURE_ID or ck.get('architecture_config_hash')!=architecture_config_hash(ck.get('architecture_config')): raise SystemExit('checkpoint architecture fingerprint is incompatible')
    mode=ck.get('fusion_mode','none'); p1=mode!='none'
    if not p1 and any((a.context_source_type,a.watch_context_npy,a.d6_context_npy,a.context_baseline_npy,a.context_valid_mask_npy)): raise SystemExit('P0 inference cannot receive context')
    if p1 and not a.context_baseline_npy: raise SystemExit('P1 requires visible complete-record context baselines; do not center per window')
    if p1 and ck.get('task_id')!=a.task_id: raise SystemExit('P1 checkpoint task mismatch')
    if p1 and ck.get('context_source_type')!=a.context_source_type: raise SystemExit('P1 checkpoint context-source mismatch')
    if p1 and ck.get('body_scale_variant')!='A_raw_window': raise SystemExit('Use the matching detrended-input adapter for B-variant checkpoints')
    output=Path(a.output_dir)
    if output.exists() and any(output.iterdir()): raise SystemExit('Use an empty inference output directory')
    if p1 and not a.context_source_type: raise SystemExit('P1 inference requires --context-source-type')
    if a.task_id=='task1':
        if p1 and (a.context_source_type!='watch_ecg' or not a.watch_context_npy): raise SystemExit('task1 P1 inference requires watch context')
        if a.d6_context_npy: raise SystemExit('task1 cannot receive d6 context')
    else:
        if p1 and a.context_source_type not in {'body_scale_d6','ecg_machine_d6'}: raise SystemExit('task2 requires exactly one d6 source type')
        if p1 and not a.d6_context_npy: raise SystemExit('task2 P1 inference requires one d6 context')
        if a.watch_context_npy: raise SystemExit('task2 cannot receive watch context')
    run=Path(a.run_dir).resolve() if a.run_dir else Path(a.checkpoint).resolve().parent; pc=PreprocessingConfig.from_yaml(ROOT/'configs'/'preprocessing.yaml'); # frozen NPZ is loaded and checked below
    frozen=ECGPreprocessor.load(pc,run/'preprocessing_scales.npz')
    expected_scale=np.asarray(ck.get('target_d12_scale_uV',[]),dtype=np.float32)
    if expected_scale.shape!=(12,) or not np.array_equal(expected_scale,frozen.scale_uV_by_source['d12']): raise SystemExit('Checkpoint and frozen target scales differ')
    if p1 and (a.context_source_type not in frozen.scale_uV_by_source or not np.array_equal(np.asarray(ck.get('context_scale_uV',[]),dtype=np.float32),frozen.scale_uV_by_source[a.context_source_type])): raise SystemExit('Checkpoint and frozen context scales differ')
    pre=frozen; scales={k:v.tolist() for k,v in frozen.scale_uV_by_source.items()}
    anchor=np.asarray(np.load(a.anchor_npy),dtype=np.float32)
    if anchor.ndim!=3 or not len(anchor) or anchor.shape[1:]!=(1,5000): raise SystemExit('anchor-npy must be [N,1,5000]')
    anchor_model,_,_=pre.transform_batch(anchor,'ecg_machine_i'); context_model=None; context_mask=None
    if p1:
        cpath=a.watch_context_npy if a.task_id=='task1' else a.d6_context_npy; context=np.asarray(np.load(cpath),dtype=np.float32); expected=1 if a.task_id=='task1' else 6
        if context.ndim!=3 or context.shape[1:]!=(expected,5000): raise SystemExit('context npy has invalid shape')
        baseline=np.asarray(np.load(a.context_baseline_npy),dtype=np.float32)
        if baseline.shape==(expected,): baseline=np.broadcast_to(baseline,(len(context),expected)).copy()
        context_model,_,_=pre.transform_batch(context,a.context_source_type,baseline); context_mask=torch.ones((context.shape[0],expected),dtype=torch.bool)
        valid=np.ones_like(context_model,dtype=bool) if not a.context_valid_mask_npy else np.asarray(np.load(a.context_valid_mask_npy),dtype=bool)
        if valid.shape==(len(context),5000): valid=np.broadcast_to(valid[:,None,:],context_model.shape)
        if valid.shape!=context_model.shape: raise SystemExit('context validity mask shape mismatch')
        available=valid.all(axis=(1,2)); context_model=np.where(valid,context_model,0.).astype(np.float32)
        if len(context)!=len(anchor): raise SystemExit('context and anchor batch lengths differ')
        if a.shuffled_context:
            if context_model.shape[0] < 2: raise SystemExit('shuffled-context requires at least two samples')
            order=np.roll(np.arange(context_model.shape[0]),1); context_model=context_model[order]; available=available[order]
    model=M1AxialLeadTimeModel(fusion_mode=mode,task_id=a.task_id,config=ck['architecture_config']); model.load_state_dict(ck['model'],strict=True); model.to(a.device).eval(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    predictions=[]
    with torch.no_grad():
        for begin in range(0,len(anchor_model),a.batch_size):
            end=begin+a.batch_size; kwargs={}
            if p1: kwargs.update(context=torch.from_numpy(context_model[begin:end]).to(a.device),context_source_type=a.context_source_type,context_lead_mask=context_mask[begin:end].to(a.device),context_available=torch.from_numpy(available[begin:end]).to(a.device))
            predictions.append(model(torch.from_numpy(anchor_model[begin:end]).to(a.device),**kwargs).cpu().numpy())
    pred=np.concatenate(predictions)*np.asarray(scales['d12'],dtype=np.float32)[None,:,None]
    submit=pred.copy()
    if a.copy_observed_i: submit[:,:1]=anchor
    np.save(out/'prediction_raw.npy',pred.astype(np.float32)); np.save(out/'prediction_submit.npy',submit.astype(np.float32)); (out/'inference_contract.json').write_text(json.dumps({'architecture_version':ARCHITECTURE_VERSION,'fusion_mode':mode,'task_id':a.task_id,'context_source_type':a.context_source_type,'anchor_npy':str(Path(a.anchor_npy).resolve()),'hidden_target_argument_used':False,'copy_observed_i':a.copy_observed_i,'preprocessing_protocol':'v3-raw-target-record-context','shuffled_context':bool(a.shuffled_context)},indent=2),encoding='utf-8'); print(f'Wrote prediction_raw.npy and prediction_submit.npy to {out}')
if __name__=='__main__':main()
