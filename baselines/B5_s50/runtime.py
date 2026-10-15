"""Three matched arms, original FM objective, QC selection and full-score reference."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg12gen.evaluate import evaluate_record_predictions,write_report,write_competition_score
from baselines.B5.checkpoint import load_checkpoint,atomic_save,pack_rng,restore_rng
from baselines.B5.config import ModelConfig
from baselines.B5.flow import sample
from baselines.B5.losses import flow_loss
from baselines.B5.runtime import seed_all,seed_worker,device_from_name
from . import PROTOCOL,SCORE_PROTOCOL,INPUTS,GROUPS
from .common import read,write,sha,state_hash,csv_read,csv_write,finite_json
from .data import ConditionedDataset,collate,to_device
from .prepare import source_datasets,load_shared
from .model import make_model,BoundModel
from .scoring import score_tasks,diagnostics
from .audit import check_initial_score

CANDIDATES={'score':'best.pt','r':'best_r.pt','bonus':'best_bonus.pt'}


def slow_source(info,group):
    if group not in GROUPS: raise ValueError('Expected control, slow or slow_f')
    # Control never injects this cache; using the S50 lookup keeps loading identical.
    return info['slow_sources']['slow_f' if group=='slow_f' else 'slow']


def datasets(shared,info,group):
    config=read(shared/'source_config.json')
    checkpoint=load_checkpoint(info['fm_checkpoint'])
    _,train,validation=source_datasets(config,checkpoint)
    metadata=read(shared/'records.json')
    variant='slow_f' if group=='slow_f' else 'slow'
    train=ConditionedDataset(train,np.load(shared/('train_'+variant+'.npy'),mmap_mode='r'),metadata['train'])
    val_native=np.load(shared/('validation_'+variant+'.npy'),mmap_mode='r')
    validation={task:ConditionedDataset(ds,val_native,metadata['validation']) for task,ds in validation.items()}
    return train,validation,metadata


@contextmanager
def ema_evaluation(model,ema):
    rng=pack_rng()
    training=model.training
    current={k:v.detach().clone() for k,v in model.state_dict().items()}
    try:
        model.load_state_dict(ema,strict=True)
        model.eval()
        yield
    finally:
        model.load_state_dict(current,strict=True)
        model.train(training)
        restore_rng(rng)  # Validation must not perturb subsequent training noise/dropout.


@torch.no_grad()
def validate(model,validation,shared,info,device,output,plots=False):
    model.eval()
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use empty evaluation output')
    output.mkdir(parents=True,exist_ok=True)
    summaries={};full_summaries={};qualities={}
    scale=np.asarray(info['scales']['d12'],np.float32)[None,:,None]
    settings=info['sampling']
    arrays={}
    for task,dataset in validation.items():
        folder=output/task;folder.mkdir()
        prediction=np.lib.format.open_memmap(folder/'prediction_uV.npy',mode='w+',dtype=np.float32,shape=(len(dataset),12,5000))
        source=Path(info['evaluation_dir'])/task
        target=np.load(source/'target_uV.npy',mmap_mode='r',allow_pickle=False)
        rows=csv_read(source/'window_metadata.csv')
        quality=np.zeros((len(dataset),12),bool)
        loader=DataLoader(dataset,batch_size=int(info['validation']['batch_size']),shuffle=False,num_workers=0,collate_fn=collate)
        offset=0
        for batch in loader:
            tensors=to_device({key:batch[key] for key in (*INPUTS,'slow')},device)
            bound=BoundModel(model,tensors['slow'])
            generated=sample(bound,{key:tensors[key] for key in INPUTS},batch['keys'],**settings)
            size=len(generated)
            expected=[str(r['target_record_id'])+':'+str(r['start_sample_500hz']) for r in rows[offset:offset+size]]
            if batch['keys']!=expected: raise ValueError('Evaluation rows/targets were reordered')
            prediction[offset:offset+size]=generated.float().cpu().numpy()*scale
            quality[offset:offset+size]=batch['quality_mask'].cpu().numpy()
            offset+=size
        expected_quality=np.load(shared/(task+'_quality.npy'),allow_pickle=False)
        if offset!=len(dataset) or not np.array_equal(quality,expected_quality): raise ValueError('Current QC masks/population differ from prepared')
        prediction.flush()
        summary,details=evaluate_record_predictions(prediction,target,task,rows,target_quality_mask=quality)
        summary.update(competition_score_protocol=SCORE_PROTOCOL,checkpoint_selection_metric='competition_total_score',slow_condition_used=hasattr(model,'slow_adapter'),
            sampling=settings,score_population='target_lead_QC',target_used_at_inference=False)
        write_report(folder,summary,details,title='B5-S50 QC selection; same raw-uV predictions')
        np.save(folder/'target_quality_mask.npy',quality)
        full,full_details=evaluate_record_predictions(prediction,target,task,rows)
        full.update(sampling=settings,target_used_at_inference=False,score_population='full_validation_reference_only',
                    checkpoint_selection_metric='not_selected_reference_only')
        write_report(output/'full_population'/task,full,full_details,title='Full-population reference; not checkpoint selector')
        csv_write(folder/'window_metadata.csv',rows)
        write(folder/'target_reference.json',{'path':str(source/'target_uV.npy'),
              'sha256':info['evaluation_files'][task+'/target_uV.npy']})
        summaries[task]=summary
        full_summaries[task]=full;qualities[task]=quality
        arrays[task]=(prediction,target)
    scores=score_tasks(summaries)
    write_competition_score(output,scores)
    full_scores=score_tasks(full_summaries,qc=False)
    write_competition_score(output/'full_population',full_scores)
    metadata=read(shared/'records.json')
    pairs={p['target_record_id']:p for p in metadata['pairs']['task2']}
    masks=np.stack([qualities['task2'][pairs[r['target_record_id']]['indices'][0],1:] for r in metadata['validation']])
    wave=diagnostics(*arrays['task2'],metadata,masks,output,plots)
    result={'summaries':summaries,'score':scores,'full_summaries':full_summaries,'full_score':full_scores,'diagnostics':wave}
    write(output/'metrics.json',finite_json(result))
    del arrays
    return result


def run_config(group,shared,info):
    source=slow_source(info,group)
    return {'protocol':PROTOCOL,'group':group,'shared_sha256':sha(shared/'shared.json'),
        'fm_sha256':info['fm_sha256'],'slow_sha256':source['sha256'],
        'slow_variant':'none' if group=='control' else group,'slow_state_sha256':source['state_sha256'],
        'implementation_sha256':info['implementation_sha256'],'model_config':info['model_config'],
        'loss':info['loss'],'training':info['training'],'sampling':info['sampling'],
        'score_protocol':SCORE_PROTOCOL,'slow_frozen':True,'adapter_hidden':32,
        'zero_init_candidate_included':True,'ema_initialized_from_same_source':True}


def update_best(best,scores,epoch):
    if scores.get('competition_score_protocol')!=SCORE_PROTOCOL: raise ValueError('Select checkpoints only with QC scores')
    values={'score':scores['competition_total_score'],'r':scores['main_correlation'],'bonus':scores['task2_rmse_bonus_score']}
    changed=[]
    for key,value in values.items():
        if not np.isfinite(value): raise ValueError('Undefined checkpoint selector')
        if value>best[key]['value']:
            best[key]={'value':float(value),'epoch':epoch};changed.append(key)
    return changed


def run(args):
    print('Checking shared cache, initializers and original data (read-only)...',flush=True)
    shared,info=load_shared(args.shared)
    config=run_config(args.group,shared,info)
    output=Path(args.output).resolve()
    if not args.resume and output.exists() and any(output.iterdir()): raise FileExistsError('Use empty run output or exact --resume')
    device=device_from_name(args.device)
    training=info['training']
    seed_all(int(training['seed']),bool(training['deterministic']))
    source=load_checkpoint(info['fm_checkpoint'])
    model=make_model(ModelConfig.from_dict(info['model_config']),args.group,source['ema']).to(device)
    initial={k:v for k,v in model.state_dict().items() if not k.startswith('slow_adapter.')}
    if state_hash(initial)!=info['fm_initial_ema_sha256']: raise ValueError('FM initializer changed')
    selected_source=slow_source(info,args.group)
    slow_cp=torch.load(selected_source['checkpoint'],map_location='cpu',weights_only=True)
    if state_hash(slow_cp['model'])!=selected_source['state_sha256']: raise ValueError('Selected slow state changed')
    # Reset after architecture construction: the adapter must not shift the
    # control/experimental training noise, metadata dropout or shuffle stream.
    seed_all(int(training['seed']),bool(training['deterministic']))
    generator=torch.Generator().manual_seed(int(training['seed']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(training['learning_rate']),weight_decay=float(training['weight_decay']))
    amp=bool(training['amp']) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    train,val,_=datasets(shared,info,args.group)
    loader=DataLoader(train,batch_size=int(training['batch_size']),shuffle=True,generator=generator,
        num_workers=int(training['workers']),collate_fn=collate,worker_init_fn=seed_worker,persistent_workers=False)
    best={k:{'value':-float('inf'),'epoch':None} for k in CANDIDATES}
    snapshots={};start=0;updates=0
    if args.resume:
        from .checkpoint import load
        checkpoint=load(args.resume)
        if checkpoint['config']!=config or checkpoint.get('run_directory')!=str(output):
            raise ValueError('Exact resume changed group, cache, settings or output; never resume the original B5 run')
        if not 0<=checkpoint['epoch']<=int(training['epochs']): raise ValueError('Resume epoch outside budget')
        for name in ('history.jsonl','initial_evaluation.json','resolved_config.json'):
            if not (output/name).is_file(): raise FileNotFoundError(name)
        model.load_state_dict(checkpoint['raw_model'],strict=True)
        ema={k:v.to(device).clone() for k,v in checkpoint['ema'].items()}
        optimizer.load_state_dict(checkpoint['optimizer']);scaler.load_state_dict(checkpoint['grad_scaler'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value): values[key]=value.to(device)
        generator.set_state(checkpoint['loader_rng'].cpu());restore_rng(checkpoint['rng'])
        best,snapshots,start,updates=checkpoint['best'],checkpoint['best_snapshots'],checkpoint['epoch'],checkpoint['updates']
        history=[json.loads(line) for line in (output/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not any(row['epoch']==start for row in history): raise ValueError('Resume history is missing saved epoch')
        (output/'history.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in history if row['epoch']<=start),encoding='utf-8')
        # Recover candidate aliases if an interrupted save left a newer alias.
        for key,name in CANDIDATES.items():
            candidate=snapshots[key]
            if candidate['epoch']!=best[key]['epoch']: raise ValueError('Best checkpoint ledger is inconsistent')
            repair=True
            try:
                existing=load(output/name)
                repair=existing['epoch']!=candidate['epoch'] or existing['model_ema_sha256']!=candidate['model_ema_sha256']
            except Exception: pass
            if repair: atomic_save(output/name,candidate)
    output.mkdir(parents=True,exist_ok=True)
    write(output/'resolved_config.json',config)
    scale_tensor=torch.as_tensor(info['scales']['d12'],device=device)

    def save(epoch,record,changed):
        weights={k:v.detach().cpu().clone() for k,v in ema.items()}
        candidate={'protocol':PROTOCOL,'score_protocol':SCORE_PROTOCOL,'group':args.group,'config':config,
            'epoch':epoch,'source_checkpoint_epoch':source['epoch'],'model_config':info['model_config'],
            'base_architecture_hash':source['architecture_hash'],'model':weights,'model_ema_sha256':state_hash(weights),
            'scales':info['scales'],'scales_sha256':info['scales_sha256'],
            'slow_state':slow_cp['model'],'slow_state_sha256':selected_source['state_sha256'],
            'slow_variant':config['slow_variant'],
            'sampling':info['sampling'],'metrics':record.get('validation')}
        for key in changed: snapshots[key]=candidate
        payload={**candidate,'run_directory':str(output),'updates':updates,'raw_model':model.state_dict(),
            'ema':ema,'optimizer':optimizer.state_dict(),'grad_scaler':scaler.state_dict(),
            'rng':pack_rng(),'loader_rng':generator.get_state(),'best':best,'best_snapshots':snapshots}
        with (output/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(finite_json(record),ensure_ascii=False,allow_nan=False)+'\n')
        for key in changed: atomic_save(output/CANDIDATES[key],snapshots[key])
        atomic_save(output/'last.pt',payload)
        print(json.dumps(finite_json(record),ensure_ascii=False,allow_nan=False),flush=True)

    def evaluate_epoch(epoch):
        folder=output/'validation'/f'epoch_{epoch:04d}'
        if folder.exists():
            # Last.pt defines the completed epoch boundary; preserve interrupted outputs.
            backup=folder.with_name(folder.name+'_interrupted_'+str(__import__('time').time_ns()))
            folder.rename(backup)
        with ema_evaluation(model,ema):
            return validate(model,val,shared,info,device,folder)

    print({'phase':'start','group':args.group,'start_epoch':start,'additional_epochs':training['epochs'],
        'learning_rate':training['learning_rate'],'parameters':sum(p.numel() for p in model.parameters()),
        'new_parameters':sum(p.numel() for k,p in model.named_parameters() if k.startswith('slow_adapter.')),
        'sampling':info['sampling'],'frozen_slow':True},flush=True)
    if not args.resume:
        result=evaluate_epoch(0)
        check_initial_score(result['score'],info)
        check_initial_score(result['full_score'],info,full=True)
        write(output/'initial_evaluation.json',finite_json(result))
        changed=update_best(best,result['score'],0)
        save(0,{'epoch':0,'phase':'common_initializer','validation':result,'best':best},changed)
    for epoch in range(start+1,int(training['epochs'])+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        running={};count=0
        for index,batch in enumerate(loader):
            b=to_device(batch,device)
            accumulation=int(training['gradient_accumulation'])
            divisor=min(accumulation,len(loader)-(index//accumulation)*accumulation)
            with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):
                loss,parts=flow_loss(BoundModel(model,b['slow']),b,info['loss'],scale_tensor)
            scaler.scale(loss/divisor).backward()
            for key,value in {'loss':loss,**parts}.items(): running[key]=running.get(key,0.)+float(value.detach())*len(b['anchor'])
            count+=len(b['anchor'])
            if (index+1)%accumulation==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(training['gradient_clip']),error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);updates+=1
                with torch.no_grad():
                    for key,value in model.state_dict().items():
                        if torch.is_floating_point(ema[key]): ema[key].mul_(float(training['ema_decay'])).add_(value,alpha=1-float(training['ema_decay']))
                        else: ema[key].copy_(value)
        if count==0: raise ValueError('Empty training set')
        record={'epoch':epoch,'updates':updates,'train':{k:v/count for k,v in running.items()}}
        changed=[]
        if epoch%int(training['validate_every'])==0 or epoch==int(training['epochs']):
            result=evaluate_epoch(epoch);record['validation']=result;changed=update_best(best,result['score'],epoch)
        record['best']=copy.deepcopy(best)
        save(epoch,record,changed)
    write(output/'completed.json',{'completed':True,'group':args.group,'additional_epochs':int(training['epochs']),
                                 'best':best,'source_fm_sha256':info['fm_sha256'],'slow_frozen':True})
