"""Audit source artifacts and generate S50 conditions once. Inference only, no optimizer."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from baselines.B5.checkpoint import load_checkpoint,checkpoint_preprocessor,scales_digest
from baselines.B5.config import ModelConfig
from baselines.B5.data import HuaweiTrainDataset,HuaweiValidationDataset,demographics_table,preprocessing_config
from baselines.B5.objective import loss_signature
from baselines.B5.runtime import device_from_name,seed_all
from . import PROTOCOL,SCORE_PROTOCOL,INPUTS,SLOW_GROUPS
from .common import read,write,sha,state_hash,code_hashes,csv_read
from .data import verify_inputs
from .slow import load_slow,predict_native,validate_slow_pair
from .scoring import score_tasks
from ecg12gen.evaluate import evaluate_record_predictions


def source_datasets(config,checkpoint):
    preprocessor=checkpoint_preprocessor(checkpoint,preprocessing_config(config))
    demo=demographics_table(config)
    train=HuaweiTrainDataset(config,preprocessor,demo)
    val={task:HuaweiValidationDataset(config,task,preprocessor,demo) for task in ('task1','task2')}
    for key,dataset in [('train',train),*val.items()]:
        if dataset.manifest_digest!=checkpoint['manifests'][key]:
            raise ValueError('Source FM population changed: '+key)
    return preprocessor,train,val


def prepare(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use a fresh shared cache; never overwrite an existing experiment')
    prepared=Path(args.prepared).resolve()
    info=read(prepared/'prepared.json')
    if info.get('protocol')!='B5-slow-probe-v1' or not info.get('completed'):
        raise ValueError('Requires completed S30/S50 prepared inputs')
    for name,expected in info['files'].items():
        if sha(prepared/name)!=expected: raise ValueError('Prepared file changed: '+name)
    fm=load_checkpoint(args.fm_checkpoint)
    if sha(args.fm_checkpoint)!=info['checkpoint_sha256']:
        raise ValueError('Use the original best.pt bound to prepared; a different FM initializer needs a new audited reference')
    if fm.get('cache_protocol')!='independent-record-context-v1' or fm.get('stage')!='finetune':
        raise ValueError('Requires the specified main-v2 finetuned B5 source')
    for key,check in info['source_checks'].items():
        print('Checking source data: '+key,flush=True)
        if sha(check['path'])!=check['sha256'] or fm['manifests'][key]!=check['sha256']:
            raise ValueError('Original data changed: '+key)
    for name,expected in info['evaluation_files'].items():
        if sha(Path(info['evaluation_dir'])/name)!=expected: raise ValueError('Original evaluation changed: '+name)
    paths={'slow':Path(args.slow_checkpoint).resolve(),'slow_f':Path(args.slow_f_checkpoint).resolve()}
    slow_cps={key:torch.load(path,map_location='cpu',weights_only=True) for key,path in paths.items()}
    validate_slow_pair(slow_cps)
    for cp in slow_cps.values():
        if cp['config']['prepared_sha256']!=sha(prepared/'prepared.json'):
            raise ValueError('Slow checkpoint belongs to different prepared data')
    cfg=copy.deepcopy(fm.get('config') or read(Path(args.fm_checkpoint).parent/'resolved_config.json'))
    if ModelConfig.from_dict(cfg['model']).fingerprint!=fm['architecture_hash']:
        raise ValueError('Source configuration and FM checkpoint architecture differ')
    if loss_signature(cfg['loss'])!=loss_signature(fm['loss_config']): raise ValueError('Source loss configuration differs')
    cfg['training']=copy.deepcopy(fm['training_config'])
    for key in ('seed','batch_size','gradient_accumulation','learning_rate','weight_decay','ema_decay','amp','gradient_clip','workers','deterministic'):
        if key not in cfg['training']: raise ValueError('Missing source training setting: '+key)
    cfg['training'].update(epochs=20,validate_every=5)
    cfg['sampling']={'seed':42,'solver':'heun','steps':16,'samples':16}
    preprocessor,train,val=source_datasets(cfg,fm)
    if not np.array_equal(np.asarray(fm['scales']['d12'],np.float32),np.asarray(info['scale_uV'],np.float32)):
        raise ValueError('Slow and FM target scales differ; do not multiply by a second scale')
    metadata=read(prepared/'records.json')
    packs={}
    for split in ('train','validation'):
        with np.load(prepared/(split+'.npz'),allow_pickle=False) as archive:
            # Target/label arrays are deliberately not loaded for condition generation.
            packs[split]={k:archive[k].copy() for k in INPUTS}
    verify_inputs(train,metadata['train'],packs['train'])
    # Validation pairs may repeat a target; use exactly one complete machine pair per record for the audit.
    from .audit import verify_validation_inputs
    verify_validation_inputs(val,metadata,packs['validation'])
    output.mkdir(parents=True,exist_ok=True)
    source_summaries={};source_full_summaries={};qualities={}
    for task,ds in val.items():
        rows=csv_read(Path(info['evaluation_dir'])/task/'window_metadata.csv')
        identity=('pair_id','target_record_id','subject_id','start_sample_500hz','input_type')
        if len(rows)!=len(ds) or any(any(str(a[k])!=str(b[k]) for k in identity) for a,b in zip(rows,ds.rows)):
            raise ValueError('Original evaluation order differs from current dataset')
        # Re-score identical raw predictions under current QC and full rules.
        folder=Path(info['evaluation_dir'])/task
        p=np.load(folder/'prediction_uV.npy',mmap_mode='r',allow_pickle=False)
        y=np.load(folder/'target_uV.npy',mmap_mode='r',allow_pickle=False)
        masks=[]
        for i in range(len(ds)):
            item=ds[i];mask=np.asarray(item['quality_mask'])
            if mask.shape!=(12,) or not np.isin(mask,[0,1]).all(): raise ValueError('Invalid current target QC mask')
            if not np.array_equal(item['target_uV'],y[i]): raise ValueError('Reference raw targets differ from current validation')
            masks.append(mask.astype(bool))
        qualities[task]=np.stack(masks);np.save(output/(task+'_quality.npy'),qualities[task])
        source_summaries[task]=evaluate_record_predictions(p,y,task,rows,target_quality_mask=qualities[task])[0]
        source_full_summaries[task]=evaluate_record_predictions(p,y,task,rows)[0]
        p._mmap.close();y._mmap.close()
    references={}
    for variant,path in paths.items():
        reference_file=path.parent/('evaluation_'+path.stem)/'evaluation.json'
        if not reference_file.is_file(): continue
        report=read(reference_file)
        if not (report.get('completed') and report.get('probe_checkpoint_sha256')==sha(path)
                and report.get('B5_checkpoint_sha256')==sha(args.fm_checkpoint)
                and report.get('prepared_sha256')==sha(prepared/'prepared.json')
                and report.get('context_seconds')==50 and report.get('group')==slow_cps[variant]['config']['group']
                and report.get('target_used_at_inference') is False):
            raise ValueError('Hybrid reference provenance differs: '+variant)
        qc={};full={}
        for task in val:
            source=Path(info['evaluation_dir'])/task
            p=np.load(reference_file.parent/'learned_slow_hybrid'/task/'prediction_uV.npy',mmap_mode='r',allow_pickle=False)
            y=np.load(source/'target_uV.npy',mmap_mode='r',allow_pickle=False)
            rows=csv_read(source/'window_metadata.csv')
            qc[task]=evaluate_record_predictions(p,y,task,rows,target_quality_mask=qualities[task])[0]
            full[task]=evaluate_record_predictions(p,y,task,rows)[0]
            p._mmap.close();y._mmap.close()
        references[variant]={'path':str(reference_file.resolve()),'sha256':sha(reference_file),'epoch':report['epoch'],
            'source_checkpoint_sha256':sha(path),'score':score_tasks(qc),'full_score':score_tasks(full,qc=False),
            'result_kind':'fixed_output_hybrid_not_FM_integration'}
    seed_all(42,True)
    device=device_from_name(args.device)
    slow_sources={}
    for variant in SLOW_GROUPS:
        cp=slow_cps[variant];slow_model=load_slow(cp,device,variant)
        for split,arrays in packs.items():
            cached=np.lib.format.open_memmap(output/(split+'_'+variant+'.npy'),mode='w+',dtype=np.float32,shape=(len(arrays['anchor']),11,6000))
            for start in range(0,len(cached),2):
                inputs={k:torch.from_numpy(v[start:start+2]).to(device) for k,v in arrays.items()}
                cached[start:start+2]=predict_native(slow_model,inputs).cpu().numpy()
                print(f'{variant} cache {split}: {min(start+2,len(cached))}/{len(cached)} records',flush=True)
            cached.flush();cached._mmap.close()
        slow_sources[variant]={'checkpoint':str(paths[variant]),'sha256':sha(paths[variant]),'state_sha256':state_hash(cp['model']),
            'source_group':cp['config']['group'],'source_config':cp['config'],'frozen':True}
        del slow_model
    write(output/'source_config.json',cfg)
    write(output/'records.json',metadata)
    payload={'protocol':PROTOCOL,'completed':True,'training_started':False,'optimizer_updates':0,
        'score_protocol':SCORE_PROTOCOL,'fm_checkpoint':str(Path(args.fm_checkpoint).resolve()),'fm_sha256':sha(args.fm_checkpoint),
        'slow_sources':slow_sources,'fm_initial_ema_sha256':state_hash(fm['ema']),
        'prepared':str(prepared),'prepared_sha256':sha(prepared/'prepared.json'),
        'source_checks':info['source_checks'],'evaluation_dir':info['evaluation_dir'],'evaluation_files':info['evaluation_files'],
        'scales':fm['scales'],'scales_sha256':scales_digest(fm['scales']),'source_summaries':source_summaries,
        'source_full_summaries':source_full_summaries,'fixed_hybrid_references':references,
        'model_config':fm['model_config'],'loss':cfg['loss'],'training':cfg['training'],'sampling':cfg['sampling'],
        'validation':cfg['validation'],'manifests':fm['manifests'],'implementation_sha256':code_hashes(),
        'train_records':len(metadata['train']),'validation_records':len(metadata['validation']),
        'train_windows':len(train),'validation_windows':{k:len(v) for k,v in val.items()},
        'files':{name:sha(output/name) for name in ('train_slow.npy','validation_slow.npy','train_slow_f.npy','validation_slow_f.npy',
            'task1_quality.npy','task2_quality.npy','source_config.json','records.json')}}
    write(output/'shared.json',payload)
    print({'completed':True,'optimizer_updates':0,'shared':str(output),'score_protocol':SCORE_PROTOCOL},flush=True)


def load_shared(path,verify_sources=True):
    path=Path(path).resolve()
    info=read(path/'shared.json')
    if not info.get('completed') or info.get('protocol')!=PROTOCOL:
        raise ValueError('Requires new QC three-arm cache; the v1 cache cannot be resumed')
    if info.get('score_protocol')!=SCORE_PROTOCOL or set(info['slow_sources'])!=set(SLOW_GROUPS): raise ValueError('Incomplete QC/slow-source protocol')
    if info['implementation_sha256']!=code_hashes(): raise ValueError('Implementation changed after prepare')
    for name,expected in info['files'].items():
        if sha(path/name)!=expected: raise ValueError('Shared cache changed: '+name)
    if verify_sources:
        if sha(info['fm_checkpoint'])!=info['fm_sha256']: raise ValueError('FM initializer changed')
        for source in info['slow_sources'].values():
            if sha(source['checkpoint'])!=source['sha256']: raise ValueError('Slow initializer changed')
        if sha(Path(info['prepared'])/'prepared.json')!=info['prepared_sha256']: raise ValueError('Prepared metadata changed')
        for check in info['source_checks'].values():
            if sha(check['path'])!=check['sha256']: raise ValueError('Source data changed: '+check['path'])
        for name,expected in info['evaluation_files'].items():
            if sha(Path(info['evaluation_dir'])/name)!=expected: raise ValueError('Reference evaluation changed: '+name)
    return path,info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fm-checkpoint',required=True)
    parser.add_argument('--slow-checkpoint',required=True)
    parser.add_argument('--slow-f-checkpoint',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    prepare(parser.parse_args())


if __name__=='__main__': main()
