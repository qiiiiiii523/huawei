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
from . import PROTOCOL,SCORE_PROTOCOL,INPUTS
from .common import read,write,sha,state_hash,code_hashes,csv_read
from .data import verify_inputs
from .slow import load_slow,predict_native
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
        raise ValueError('Use the exact FM best.pt that generated the slow experiment; not best_r.pt or another model')
    if fm.get('cache_protocol')!='independent-record-context-v1' or fm.get('stage')!='finetune':
        raise ValueError('Requires the specified main-v2 finetuned B5 source')
    for key,check in info['source_checks'].items():
        print('Checking source data: '+key,flush=True)
        if sha(check['path'])!=check['sha256'] or fm['manifests'][key]!=check['sha256']:
            raise ValueError('Original data changed: '+key)
    for name,expected in info['evaluation_files'].items():
        if sha(Path(info['evaluation_dir'])/name)!=expected: raise ValueError('Original evaluation changed: '+name)
    slow_cp=torch.load(args.slow_checkpoint,map_location='cpu',weights_only=True)
    if slow_cp['config']['prepared_sha256']!=sha(prepared/'prepared.json'):
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
    source_summaries={}
    for task,ds in val.items():
        rows=csv_read(Path(info['evaluation_dir'])/task/'window_metadata.csv')
        identity=('pair_id','target_record_id','subject_id','start_sample_500hz','input_type')
        if len(rows)!=len(ds) or any(any(str(a[k])!=str(b[k]) for k in identity) for a,b in zip(rows,ds.rows)):
            raise ValueError('Original evaluation order differs from current dataset')
        # Historical CSVs may precede target-equal aggregation. Re-score the
        # original frozen predictions with this experiment's explicit v2 rule.
        folder=Path(info['evaluation_dir'])/task
        source_summaries[task]=evaluate_record_predictions(
            np.load(folder/'prediction_uV.npy',mmap_mode='r',allow_pickle=False),
            np.load(folder/'target_uV.npy',mmap_mode='r',allow_pickle=False),task,rows)[0]
    reference=None
    reference_file=Path(args.slow_checkpoint).parent/('evaluation_'+Path(args.slow_checkpoint).stem)/'evaluation.json'
    if reference_file.is_file():
        report=read(reference_file)
        if (report.get('completed') and report.get('probe_checkpoint_sha256')==sha(args.slow_checkpoint)
                and report.get('B5_checkpoint_sha256')==sha(args.fm_checkpoint)
                and report.get('prepared_sha256')==sha(prepared/'prepared.json')):
            reference={'path':str(reference_file.resolve()),'sha256':sha(reference_file),'report':report}
    seed_all(42,True)
    device=device_from_name(args.device)
    slow_model=load_slow(slow_cp,device)
    output.mkdir(parents=True,exist_ok=True)
    for split,arrays in packs.items():
        cached=np.lib.format.open_memmap(output/(split+'_slow.npy'),mode='w+',dtype=np.float32,shape=(len(arrays['anchor']),11,6000))
        for start in range(0,len(cached),2):
            inputs={k:torch.from_numpy(v[start:start+2]).to(device) for k,v in arrays.items()}
            cached[start:start+2]=predict_native(slow_model,inputs).cpu().numpy()
            print(f'S50 cache {split}: {min(start+2,len(cached))}/{len(cached)} records',flush=True)
        cached.flush()
        del cached
    write(output/'source_config.json',cfg)
    write(output/'records.json',metadata)
    payload={'protocol':PROTOCOL,'completed':True,'training_started':False,'optimizer_updates':0,
        'score_protocol':SCORE_PROTOCOL,'fm_checkpoint':str(Path(args.fm_checkpoint).resolve()),'fm_sha256':sha(args.fm_checkpoint),
        'slow_checkpoint':str(Path(args.slow_checkpoint).resolve()),'slow_sha256':sha(args.slow_checkpoint),
        'slow_state_sha256':state_hash(slow_cp['model']),'fm_initial_ema_sha256':state_hash(fm['ema']),
        'prepared':str(prepared),'prepared_sha256':sha(prepared/'prepared.json'),
        'source_checks':info['source_checks'],'evaluation_dir':info['evaluation_dir'],'evaluation_files':info['evaluation_files'],
        'scales':fm['scales'],'scales_sha256':scales_digest(fm['scales']),'source_summaries':source_summaries,
        'fixed_hybrid_reference':reference,
        'model_config':fm['model_config'],'loss':cfg['loss'],'training':cfg['training'],'sampling':cfg['sampling'],
        'validation':cfg['validation'],'manifests':fm['manifests'],'implementation_sha256':code_hashes(),
        'train_records':len(metadata['train']),'validation_records':len(metadata['validation']),
        'train_windows':len(train),'validation_windows':{k:len(v) for k,v in val.items()},
        'files':{name:sha(output/name) for name in ('train_slow.npy','validation_slow.npy','source_config.json','records.json')}}
    write(output/'shared.json',payload)
    print({'completed':True,'optimizer_updates':0,'shared':str(output),'score_protocol':SCORE_PROTOCOL},flush=True)


def load_shared(path,verify_sources=True):
    path=Path(path).resolve()
    info=read(path/'shared.json')
    if not info.get('completed') or info.get('protocol')!=PROTOCOL: raise ValueError('Incomplete shared cache')
    if info['implementation_sha256']!=code_hashes(): raise ValueError('Implementation changed after prepare')
    for name,expected in info['files'].items():
        if sha(path/name)!=expected: raise ValueError('Shared cache changed: '+name)
    if verify_sources:
        for source,key in [('fm_checkpoint','fm_sha256'),('slow_checkpoint','slow_sha256')]:
            if sha(info[source])!=info[key]: raise ValueError('Initializer changed: '+source)
        for check in info['source_checks'].values():
            if sha(check['path'])!=check['sha256']: raise ValueError('Source data changed: '+check['path'])
        for name,expected in info['evaluation_files'].items():
            if sha(Path(info['evaluation_dir'])/name)!=expected: raise ValueError('Reference evaluation changed: '+name)
    return path,info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fm-checkpoint',required=True)
    parser.add_argument('--slow-checkpoint',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    prepare(parser.parse_args())


if __name__=='__main__': main()
