"""Full-record main-v3 evaluation of an existing M1 checkpoint; no training."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/common.yaml'))
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--task-id',required=True,choices=('task1','task2'))
    parser.add_argument('--scales',help='Defaults to preprocessing_scales.npz beside checkpoint')
    parser.add_argument('--body-scale-variant',default='A_raw_window',choices=('A_raw_window','B_detrend_0p2Hz_then_window'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): parser.error('Use an empty validation output directory')
    import torch
    from ecg12gen.dataset import ECGDataConfig
    from ecg12gen.m1_protocol import checkpoint_preprocessor
    from ecg12gen.m1_axial import M1AxialLeadTimeModel
    from ecg12gen.m1_data import M1PreparedDataset, _m1_joint_anchor_samples
    from ecg12gen.m1_axial_train import validate
    from ecg12gen.training import seed_everything
    seed_everything(42,deterministic=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=ECGDataConfig.from_yaml(args.config)
    scales=Path(args.scales) if args.scales else Path(args.checkpoint).parent/'preprocessing_scales.npz'
    pre=checkpoint_preprocessor(checkpoint,config,scales)
    stage=checkpoint['stage']; mode=checkpoint['fusion_mode']; source=checkpoint.get('context_source_type')
    if stage=='P0_anchor_only' and (mode!='none' or source is not None): raise ValueError('Invalid P0 checkpoint conditions')
    if stage=='P1_joint_anchor':
        if checkpoint['task_id']!=args.task_id: raise ValueError('P1 checkpoint task mismatch')
        if mode=='none' or not source: raise ValueError('P1 checkpoint lacks context routing')
        if checkpoint.get('body_scale_variant')!=args.body_scale_variant: raise ValueError('P1 context preprocessing variant mismatch')
    elif stage!='P0_anchor_only': raise ValueError('Unknown M1 checkpoint stage')
    samples=_m1_joint_anchor_samples(config,args.task_id,'validation',args.body_scale_variant)
    if stage=='P1_joint_anchor': samples=[s for s in samples if s.context_source_type==source]
    if not samples: raise ValueError('Empty validation population')
    dataset=M1PreparedDataset(samples,pre,stage,source)
    model=M1AxialLeadTimeModel(fusion_mode=mode,task_id=args.task_id,config=checkpoint['architecture_config'])
    for key in ('architecture_version','architecture_id','architecture_config_hash','lead_order','d_model'):
        if checkpoint.get(key)!=model.architecture_metadata[key]: raise ValueError(f'M1 checkpoint {key} mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(args.device)
    summary=validate(model,dataset,pre.scale_uV_by_source['d12'],torch.device(args.device),output)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
