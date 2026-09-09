"""M1 P0/P1 training: only shared strict/joint losses and raw-uV V0 selection."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from .m1_axial import M1AxialLeadTimeModel
from .m1_data import M1PreparedDataset, m1_collate
from .losses import strict_anchor_pretrain_loss, joint_anchor_sync_loss
from .evaluate import evaluate_joint_anchor_predictions
from .training import seed_everything

def loader(ds,batch_size,shuffle,seed):
    g=torch.Generator(); g.manual_seed(seed); return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=0,collate_fn=m1_collate,generator=g)
def move(batch,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
def forward(model,batch):
    kwargs={'lead_mask':torch.cat((torch.ones((batch['anchor_model'].shape[0],1),dtype=torch.bool,device=batch['anchor_model'].device),torch.zeros((batch['anchor_model'].shape[0],11),dtype=torch.bool,device=batch['anchor_model'].device)),1)}
    if model.fusion_mode!='none': kwargs.update(context=batch['context_model'],context_source_type=batch['context_source_type'],context_lead_mask=batch['context_lead_mask'])
    return model(batch['anchor_model'],**kwargs)
def _loss(model,pred,batch,d12_scale):
    if model.fusion_mode=='none': return strict_anchor_pretrain_loss(pred,batch['target_model'],batch['anchor_model'],d12_scale_uV=d12_scale)
    return joint_anchor_sync_loss(pred,batch['target_model'],batch['anchor_model'],d12_scale_uV=d12_scale)
def _base_name(name):
    return name.startswith(('cnn_encoder.','anchor_projection.','time_position','lead_embedding','lead_state_embedding','axial_blocks.','final_norm.','decoder.'))
def _set_anchor_requires_grad(model,enabled):
    for name,p in model.named_parameters():
        if _base_name(name): p.requires_grad=enabled
def validate(model,ds,d12_scale,device,out,max_batches=None):
    model.eval(); preds=[]; targets=[]; anchors=[]
    with torch.no_grad():
        for batch_index, raw in enumerate(loader(ds,4,False,42)):
            if max_batches is not None and batch_index >= max_batches: break
            batch=move(raw,device); preds.append(forward(model,batch).cpu().numpy()); targets.append(raw['raw_target_uV'].numpy()); anchors.append(raw['raw_anchor_uV'].numpy())
    prediction=np.concatenate(preds).astype(np.float32)*np.asarray(d12_scale,dtype=np.float32)[None,:,None]; target=np.concatenate(targets).astype(np.float32); anchor=np.concatenate(anchors).astype(np.float32)
    summary,raw_details,submit_details,submit=evaluate_joint_anchor_predictions(prediction,target,anchor,model.task_id)
    out.mkdir(parents=True,exist_ok=True); np.save(out/'prediction_raw.npy',prediction); np.save(out/'prediction_submit.npy',submit)
    (out/'validation_metrics.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    return summary

def required_checkpoint_metadata(model,d12_scale):
    return {**model.architecture_metadata,'stage':'P0_anchor_only','target_d12_scale_uV':np.asarray(d12_scale,dtype=np.float32).tolist()}
def validate_p0_checkpoint(checkpoint,model,d12_scale):
    expected=model.architecture_metadata; required=('architecture_version','architecture_id','architecture_config_hash','d_model','lead_order','architecture_config')
    missing=[x for x in required if x not in checkpoint]
    if missing: raise ValueError('P0 checkpoint missing architecture metadata: '+','.join(missing))
    for key in ('architecture_version','architecture_id','architecture_config_hash','d_model','lead_order','architecture_config'):
        if checkpoint[key]!=expected[key]: raise ValueError(f'incompatible P0 checkpoint {key}: architecture mismatch')
    if checkpoint.get('stage')!='P0_anchor_only' or checkpoint.get('fusion_mode','none')!='none': raise ValueError('P1 requires a P0_anchor_only checkpoint')
    scale=np.asarray(checkpoint.get('target_d12_scale_uV',[]),dtype=np.float32)
    if scale.shape!=(12,) or not np.allclose(scale,d12_scale,rtol=0,atol=1e-5): raise ValueError('incompatible P0 checkpoint target d12 scale')
def load_p0_into_p1(model,path,d12_scale):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=False); validate_p0_checkpoint(checkpoint,model,d12_scale)
    incompatible=model.load_state_dict(checkpoint['model'],strict=False)
    unexpected=[x for x in incompatible.unexpected_keys]
    if unexpected: raise ValueError('P0 checkpoint has unexpected parameters: '+','.join(unexpected))
    return checkpoint

def fit_m1(model,train_ds,val_ds,d12_scale,output_dir,*,stage,epochs=1,device='cpu',p0_checkpoint=None,backbone_lr=1e-3,fusion_lr=2e-3,freeze_anchor_epochs=0,max_train_batches=None,max_validation_batches=None):
    if stage not in {'P0_anchor_only','P1_joint_anchor'}: raise ValueError('invalid M1 stage')
    if stage=='P1_joint_anchor' and not p0_checkpoint: raise ValueError('P1 requires --p0-checkpoint')
    seed_everything(42,deterministic=True); device=torch.device(device); model.to(device); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if stage=='P1_joint_anchor': load_p0_into_p1(model,p0_checkpoint,d12_scale)
    base=[]; fusion=[]
    for name,p in model.named_parameters(): (base if _base_name(name) else fusion).append(p)
    groups=[{'params':base,'lr':backbone_lr}]
    if fusion: groups.append({'params':fusion,'lr':fusion_lr})
    opt=torch.optim.AdamW(groups,weight_decay=1e-4); train_loader=loader(train_ds,4,True,42); history=[]; best=-float('inf')
    for epoch in range(1,epochs+1):
        if stage=='P1_joint_anchor': _set_anchor_requires_grad(model,epoch>freeze_anchor_epochs)
        model.train(); total=0.; steps=0
        for batch_index, raw in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches: break
            batch=move(raw,device); opt.zero_grad(set_to_none=True); loss=_loss(model,forward(model,batch),batch,torch.as_tensor(d12_scale,device=device)); loss.backward(); opt.step(); total+=float(loss.detach()); steps+=1
        metrics=validate(model,val_ds,d12_scale,device,out,max_validation_batches); row={'epoch':epoch,'train_loss':total/max(1,steps),'validation':metrics}; history.append(row); metric=float(metrics['r_submit_12'])
        print(f'epoch={epoch}; train_loss={row["train_loss"]:.6f}; r_submit_12={metric:.6f}; r_missing11={float(metrics["r_missing11"]):.6f}')
        if metric>best:
            best=metric; ckpt={**model.architecture_metadata,**required_checkpoint_metadata(model,d12_scale),'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'stage':stage,'target_d12_scale_uV':np.asarray(d12_scale,dtype=np.float32).tolist()}; torch.save(ckpt,out/'m1_best.pt')
    (out/'history.json').write_text(json.dumps({'best_metric':best,'history':history},indent=2,default=str),encoding='utf-8'); return out/'m1_best.pt'
