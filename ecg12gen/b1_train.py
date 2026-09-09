"""Protocol-native B1 training loop; data/loss/evaluation are main APIs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from .b1_model import B1Model
from .contracts import ContractError
from .evaluate import evaluate_joint_anchor_predictions, evaluate_centered_diagnostic
from .losses import strict_anchor_pretrain_loss, joint_anchor_sync_loss
from .protocol_data import StrictDataset, JointDataset, collate, fit_preprocessor
from .training import seed_everything

def _move(b, dev): return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
def _forward(m,b):
    return m(b["anchor_i"]) if m.fusion_mode=="none" else m(b["anchor_i"],context_ecg=b["context"],context_source_type=b["context_source_type"],context_lead_mask=b["context_lead_mask"])
def _loader(ds,bs,shuffle,seed): return DataLoader(ds,batch_size=bs,shuffle=shuffle,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(seed))
def _raw(out, base, scale): return out.astype(np.float32)*scale[None,:,None]+base.astype(np.float32)[:,:,None]

@torch.no_grad()
def validate(model, ds, scale, task, dev, bs=4, shuffle_context=False):
    model.eval(); ps=[]; ys=[]; aa=[]
    for batch in _loader(ds,bs,False,42):
        moved=_move(batch,dev)
        if shuffle_context and model.fusion_mode!="none":
            p=torch.roll(torch.arange(moved["context"].shape[0],device=dev),1); moved["context"]=moved["context"][p]; moved["context_lead_mask"]=moved["context_lead_mask"][p]; moved["context_source_type"]=[moved["context_source_type"][int(i)] for i in p.cpu()]
        o=_forward(model,moved); base=model.predict_baseline(moved["anchor_i"]).cpu().numpy(); ps.append(_raw(o.cpu().numpy(),base,scale)); ys.append(batch["target_raw"].numpy()); aa.append(batch["anchor_raw"].numpy())
    p,t,a=np.concatenate(ps),np.concatenate(ys),np.concatenate(aa); summary,raw,submit,psub=evaluate_joint_anchor_predictions(p,t,a,task)
    return {"summary":summary,"raw_details":raw,"submit_details":submit,"prediction_raw":p,"prediction_submit":psub,"target_raw":t,"anchor_raw":a}

def train_b1(args: Any) -> Path:
    if args.stage=="P0_anchor_only" and args.fusion_mode!="none": raise ContractError("P0 requires fusion_mode=none")
    if args.stage=="P1-C3" and (args.fusion_mode!="film_gated_residual" or not args.p0_checkpoint): raise ContractError("P1-C3 requires compatible P0 checkpoint")
    seed_everything(args.seed,deterministic=True); dev=torch.device(args.device); pre=fit_preprocessor(args.config,args.task_id,args.body_scale_variant,args.context_source_type if args.stage=="P1-C3" else None)
    model=B1Model(fusion_mode=args.fusion_mode,dropout=args.dropout).to(dev)
    if args.stage=="P1-C3":
        ck=torch.load(args.p0_checkpoint,map_location="cpu",weights_only=False)
        if ck.get("architecture_id")!=model.architecture_id or ck.get("architecture_config_hash")!=model.architecture_config_hash: raise ContractError("P0/P1 architecture metadata mismatch")
        model.load_state_dict(ck["model"],strict=True)
    train_ds=StrictDataset(args.config,pre) if args.stage=="P0_anchor_only" else JointDataset(args.config,args.task_id,"train",pre,args.body_scale_variant,args.context_source_type)
    val_ds=JointDataset(args.config,args.task_id,"validation",pre,args.body_scale_variant,args.context_source_type if args.stage=="P1-C3" else None)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); scale=pre.scale_uV_by_source["d12"]; opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best=-float("inf"); hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in _loader(train_ds,args.batch_size,True,args.seed):
            b=_move(batch,dev); opt.zero_grad(set_to_none=True); pred=_forward(model,b); loss=(strict_anchor_pretrain_loss(pred,b["target"],b["anchor_i"],d12_scale_uV=torch.as_tensor(scale,device=dev)) if args.stage=="P0_anchor_only" else joint_anchor_sync_loss(pred,b["target"],b["anchor_i"],d12_scale_uV=torch.as_tensor(scale,device=dev))); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu()))
        val=validate(model,val_ds,scale,args.task_id,dev,args.batch_size); metric=float(val["summary"]["r_submit_12"]); row={"epoch":ep,"train_loss":float(np.mean(losses)),"validation":val["summary"]}
        if args.stage=="P1-C3":
            sh=validate(model,val_ds,scale,args.task_id,dev,args.batch_size,True); row["shuffled_context"]={"r_submit_12":float(sh["summary"]["r_submit_12"]),"r_missing11":float(sh["summary"]["r_missing11"])}
        hist.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        if metric>best:
            best=metric; ck={"model":model.state_dict(),"task_id":args.task_id,"stage":args.stage,"fusion_mode":args.fusion_mode,"architecture_id":model.architecture_id,"architecture_config_hash":model.architecture_config_hash,"parameter_count":model.parameter_count,"epoch":ep,"best_metric":best,"scale_uV":scale.tolist()}; torch.save(ck,out/"b1_best.pt"); np.save(out/"prediction_E1_raw_uV.npy",val["prediction_raw"]); np.save(out/"prediction_E2_submit_uV.npy",val["prediction_submit"]); np.save(out/"validation_target_raw_uV.npy",val["target_raw"]); centered,centered_details=evaluate_centered_diagnostic(val["prediction_submit"],val["target_raw"],args.task_id); (out/"validation_summary.json").write_text(json.dumps(val["summary"],indent=2),encoding="utf-8"); (out/"validation_metrics.json").write_text(json.dumps({"E1_raw_details":val["raw_details"],"E2_submit_details":val["submit_details"],"E2_submit_centered":centered,"E2_submit_centered_details":centered_details},indent=2),encoding="utf-8")
    (out/"history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8"); return out/"b1_best.pt"
