"""Formal M1 inference: explicit anchor/context inputs only; no hidden target argument."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ecg12gen.m1_axial import M1AxialLeadTimeModel,ARCHITECTURE_ID,ARCHITECTURE_VERSION,architecture_config_hash
from ecg12gen.preprocessing import ECGPreprocessor,PreprocessingConfig

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',required=True); p.add_argument('--anchor-npy',required=True); p.add_argument('--task-id',choices=('task1','task2'),required=True); p.add_argument('--context-source-type',choices=('watch_ecg','body_scale_d6','ecg_machine_d6')); p.add_argument('--watch-context-npy'); p.add_argument('--d6-context-npy'); p.add_argument('--run-dir'); p.add_argument('--shuffled-context',action='store_true'); p.add_argument('--output-dir',required=True); p.add_argument('--device',default='cpu'); a=p.parse_args()
    ck=torch.load(Path(a.checkpoint),map_location='cpu',weights_only=False)
    if ck.get('architecture_version')!=ARCHITECTURE_VERSION: raise SystemExit('checkpoint architecture mismatch: old B3-like M1 checkpoints are not accepted')
    if ck.get('lead_order')!=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']: raise SystemExit('checkpoint lead order is incompatible')
    if ck.get('architecture_id')!=ARCHITECTURE_ID or ck.get('architecture_config_hash')!=architecture_config_hash(ck.get('architecture_config')): raise SystemExit('checkpoint architecture fingerprint is incompatible')
    mode=ck.get('fusion_mode','none'); p1=mode!='none'
    if p1 and not a.context_source_type: raise SystemExit('P1 inference requires --context-source-type')
    if a.task_id=='task1':
        if p1 and (a.context_source_type!='watch_ecg' or not a.watch_context_npy): raise SystemExit('task1 P1 inference requires watch context')
        if a.d6_context_npy: raise SystemExit('task1 cannot receive d6 context')
    else:
        if p1 and a.context_source_type not in {'body_scale_d6','ecg_machine_d6'}: raise SystemExit('task2 requires exactly one d6 source type')
        if p1 and not a.d6_context_npy: raise SystemExit('task2 P1 inference requires one d6 context')
        if a.watch_context_npy: raise SystemExit('task2 cannot receive watch context')
    run=Path(a.run_dir).resolve() if a.run_dir else Path(a.checkpoint).resolve().parent; pc=PreprocessingConfig.from_yaml(ROOT/'configs'/'preprocessing.yaml'); scales=json.loads((run/'preprocessing_scales.json').read_text(encoding='utf-8')); pre=ECGPreprocessor(pc,{k:np.asarray(v,dtype=np.float32) for k,v in scales.items()})
    anchor=np.asarray(np.load(a.anchor_npy),dtype=np.float32)
    if anchor.ndim!=3 or anchor.shape[1:]!=(1,5000): raise SystemExit('anchor-npy must be [N,1,5000]')
    anchor_model,_,_=pre.transform_batch(anchor,'ecg_machine_i'); context_model=None; context_mask=None
    if p1:
        cpath=a.watch_context_npy if a.task_id=='task1' else a.d6_context_npy; context=np.asarray(np.load(cpath),dtype=np.float32); expected=1 if a.task_id=='task1' else 6
        if context.ndim!=3 or context.shape[1:]!=(expected,5000): raise SystemExit('context npy has invalid shape')
        context_model,_,_=pre.transform_batch(context,a.context_source_type); context_mask=torch.ones((context.shape[0],expected),dtype=torch.bool)
        if a.shuffled_context:
            if context_model.shape[0] < 2: raise SystemExit('shuffled-context requires at least two samples')
            context_model=context_model[np.roll(np.arange(context_model.shape[0]),1)]
    model=M1AxialLeadTimeModel(fusion_mode=mode,task_id=a.task_id,config=ck['architecture_config']); model.load_state_dict(ck['model'],strict=True); model.to(a.device).eval(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        kwargs={}
        if p1: kwargs.update(context=torch.from_numpy(context_model).to(a.device),context_source_type=a.context_source_type,context_lead_mask=context_mask.to(a.device))
        pred=model(torch.from_numpy(anchor_model).to(a.device),**kwargs).cpu().numpy()*np.asarray(scales['d12'],dtype=np.float32)[None,:,None]
    submit=pred.copy(); submit[:,:1]=anchor
    np.save(out/'prediction_raw.npy',pred.astype(np.float32)); np.save(out/'prediction_submit.npy',submit.astype(np.float32)); (out/'inference_contract.json').write_text(json.dumps({'architecture_version':ARCHITECTURE_VERSION,'fusion_mode':mode,'task_id':a.task_id,'context_source_type':a.context_source_type,'anchor_npy':str(Path(a.anchor_npy).resolve()),'hidden_target_argument_used':False,'shuffled_context':bool(a.shuffled_context)},indent=2),encoding='utf-8'); print(f'Wrote prediction_raw.npy and prediction_submit.npy to {out}')
if __name__=='__main__':main()
