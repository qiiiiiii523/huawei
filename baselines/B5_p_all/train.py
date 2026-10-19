"""New paired continuation from P best; original vs additional ALL supervision only."""
import argparse
import csv
import json
from pathlib import Path
import time
import torch
from baselines.B5.checkpoint import atomic_save,pack_rng,restore_rng
from baselines.B5.runtime import device_from_name,seed_all
from baselines.B5_best.common import write,sha,state_sha
from baselines.B5_public_slow.common import model_from_bundle,SourceSampler,read_record
from baselines.B5_public_slow.core import batch_record,generate,FIELDS
from . import PROTOCOL,GROUPS
from .common import load_study,load_checkpoint,objective,group_settings
from .evaluate import evaluate


def run(study_path,group,output,device_name,resume=False):
    study_path,output=Path(study_path).resolve(),Path(output).resolve()
    if group not in GROUPS:raise ValueError('Unknown group')
    study,bundle,manifest=load_study(study_path,deep=True)
    cfg=study['training'];device=device_from_name(device_name);study_sha=sha(study_path)
    seed_all(cfg['seed']+2,True)
    model=model_from_bundle(bundle,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sampler=SourceSampler(manifest['train'],study['sampler_seed'])
    scales=torch.tensor(bundle['fm']['scales']['d12'],device=device)
    settings=group_settings(study,group)
    step,best_r,best_step=0,None,None
    if resume:
        cp,_,_,_=load_checkpoint(output/'last.pt',study_path,device)
        if cp['group']!=group or cp['output']!=str(output):raise ValueError('Resume group/output differs')
        previous,_,_,_=load_checkpoint(output/'best.pt',study_path,torch.device('cpu'))
        if previous['group']!=group or previous['output']!=str(output):raise ValueError('Existing best belongs to another group')
        model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer'])
        sampler.restore(cp['sampler']);restore_rng(cp['rng'])
        step=cp['step'];best_r=previous['best_r'];best_step=previous['best_step']
    else:
        if output.exists() and any(output.iterdir()):raise FileExistsError('Use empty group output; put tee log beside it')
        output.mkdir(parents=True,exist_ok=True)
        write(output/'initialization.json',{'group':group,'study_sha256':study_sha,'training':cfg,
              'source_checkpoint':study['source_checkpoint'],'source_step':study['source_step'],
              'initial_slow_state_sha256':state_sha(model.state_dict()),'FM_state_sha256':bundle['fm_state_sha256'],
              'sampling':study['sampling'],'beta':study['beta'],'multiscale':settings,
              'optimizer_initialization':'fresh_AdamW_same_all_arms','epoch_zero_is_candidate':True,
              'sampler_seed':study['sampler_seed'],'FM_optimizer_updates':0,'public_updates':0})

    def payload(score):
        return {'protocol':PROTOCOL,'group':group,'study_sha256':study_sha,'output':str(output),
                'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler':sampler.state(),
                'rng':pack_rng(),'best_r':best_r,'best_step':best_step,'validation_score':score,
                'multiscale':settings,'FM_state_sha256':bundle['fm_state_sha256'],
                'sampling_config':study['sampling'],'score_protocol':study['score_protocol'],
                'torch_version':str(torch.__version__)}

    def validate():
        rng=pack_rng()
        try:
            folder=output/'validation'/f'step{step:06d}'
            if folder.exists():folder=folder.with_name(folder.name+'_attempt_'+str(time.time_ns()))
            score=evaluate(model,study,bundle,device,folder,group,step,initial=step==0)
            file=output/'validation_metrics.csv';existed=file.exists()
            with file.open('a',encoding='utf-8',newline='') as f:
                row={'step':step,**score};writer=csv.DictWriter(f,fieldnames=list(row))
                if not existed:writer.writeheader()
                writer.writerow(row)
            return score
        finally:restore_rng(rng)

    if not resume:
        score=validate();best_r=score['main_correlation'];best_step=0
        atomic_save(output/'best.pt',payload(score));atomic_save(output/'last.pt',payload(score))
    write(output/'completed.json',{'completed':False,'group':group,'resuming':resume})
    for step in range(step+1,cfg['updates']+1):
        model.train();optimizer.zero_grad(set_to_none=True);totals={}
        for _ in range(2):
            record=read_record(study['huawei']['root'],sampler.draw())
            batch=batch_record(record,device)
            prediction=generate(model,{key:batch[key] for key in FIELDS})
            loss,parts=objective(prediction,batch,scales,study,group)
            (loss/2).backward()
            for k,v in {'loss':loss,**parts}.items():totals[k]=totals.get(k,0.)+float(v.detach())/2
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        optimizer.step()
        score=None;improved=False
        if step%cfg['validate_every']==0 or step==cfg['updates']:
            score=validate();improved=score['main_correlation']>best_r
            if improved:best_r=score['main_correlation'];best_step=step
            if state_sha(bundle['fm']['ema'])!=bundle['fm_state_sha256']:raise ValueError('Frozen FM changed')
            write(output/'coverage.json',sampler.coverage())
        if improved:atomic_save(output/'best.pt',payload(score))
        atomic_save(output/'last.pt',payload(score))
        row={'step':step,'train':totals,'best_r':best_r,'best_step':best_step,'selected_best':improved,
             'FM_optimizer_updates':0}
        if score is not None:row['validation']=score
        with (output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step==1 or step%10==0 or score is not None:print(json.dumps(row,allow_nan=False),flush=True)
    write(output/'completed.json',{'completed':True,'group':group,'updates':cfg['updates'],'best_r':best_r,
          'best_step':best_step,'FM_optimizer_updates':0,'public_updates':0,'multiscale_weight':settings['weight'],
          'score_protocol':study['score_protocol']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('study','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--execute-training',action='store_true')
    a=p.parse_args()
    if not a.execute_training:p.error('No training started; use --execute-training explicitly on the server')
    run(a.study,a.group,a.output,a.device,a.resume)


if __name__=='__main__':main()
