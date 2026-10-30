"""Official input -> pinned B5-final -> official 39-column CSV; inference only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from competition_interface import output as official_output
from baselines.B5_best.common import read,write,sha
from .profile import resolve_assets,DEFAULT_PROFILE
from .official_data import OfficialDataset


STANDARD_PROTOCOL = 'B5-final-official-submission-v1'
G120_C3_PROTOCOL = 'B5-final-g120-c3-submission-v1'
TCN_PROTOCOL = 'B5-final-tcn120-official-submission-v1'


def executable_root():
    return Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[2]


def resource_root():
    return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[2]))


def load_settings(path):
    value=read(path)
    if value.get('protocol') not in (STANDARD_PROTOCOL,G120_C3_PROTOCOL,TCN_PROTOCOL): raise ValueError('Wrong submission config protocol')
    if value.get('device') not in ('auto','cpu','cuda') or not isinstance(value.get('batch_size'),int) or value['batch_size']<1:
        raise ValueError('Invalid device/batch size')
    for name in ('d12_fs_fallback','d6_fs_fallback','scale_fs_fallback'):
        if value.get(name) is not None and (isinstance(value[name],bool) or not 1<=float(value[name])<=10000): raise ValueError('Invalid explicit rate fallback')
    if not 120<=float(value.get('max_history_seconds',0))<=86400: raise ValueError('Invalid history memory limit')
    if value['protocol']==G120_C3_PROTOCOL:
        required=('base_checkpoint','original_body_checkpoint','g120_checkpoint','c3_checkpoint','slow120_study',
                  'g120_sha256','c3_sha256','slow120_study_sha256')
        if any(not isinstance(value.get(name),str) or not value[name] for name in required):
            raise ValueError('Incomplete G120+C3 submission configuration')
    if value['protocol']==TCN_PROTOCOL:
        required=('base_checkpoint','body_checkpoint','tcn_checkpoint','tcn_sha256')
        if any(not isinstance(value.get(name),str) or not value[name] for name in required):
            raise ValueError('Incomplete TCN submission configuration')
    return value


def run(data_root,output,settings,*,assets=None,device=None,engine=None):
    data_root=Path(data_root).resolve();output=Path(output).resolve()
    if output==data_root or data_root in output.parents: raise ValueError('Results must not be written into test_data')
    if output.exists() and any(output.iterdir()): raise FileExistsError('Results directory must be empty; archive previous results before a new run')
    dataset=OfficialDataset(data_root,settings)
    if not dataset.reader.list_subjects(): raise ValueError('No test subjects')
    # Validate all required current-I inputs before spending GPU time or writing
    # any result. Per-person preflight keeps only I length/identity, not labels.
    for person in dataset:
        dataset.anchor(person)
    if engine is None:
        from .predict import Predictor
        from baselines.B5.runtime import device_from_name
        engine=Predictor(device_from_name(device or settings['device']),assets=assets)
    output.mkdir(parents=True,exist_ok=True)
    audit=output/'_audit';audit.mkdir()
    subjects=[]
    write(audit/'run.json',{'completed':False,'training_started':False,'target_used_at_inference':False,
        'data_root':str(data_root),'settings':settings,'assets':assets.report() if assets else {},
        'score_contract':'organizer reference r-only; RMSE diagnostic','subject_count':len(dataset.reader.list_subjects())})
    try:
        for person in dataset:
            userid=person['subject_id'];anchor,anchor_info=dataset.anchor(person)
            machine,machine_info=dataset.history(person,'ecg_machine_d6')
            body,body_info=dataset.history(person,'body_scale_d6')
            record_key=userid+':'+anchor_info['record_id']
            print('Predicting '+userid+' (Heun16/K16; three routes share the base forward)',flush=True)
            predictions,status=engine.predict_tasks(anchor,person['userinfo'] or {},record_key,
                machine_history=machine,body_history=body,batch_size=settings['batch_size'])
            payload={}
            for task in official_output.TASKS:
                raw=np.asarray(predictions[task])
                if raw.shape!=(12,60000) or not np.isfinite(raw).all(): raise ValueError('Invalid model prediction: '+task)
                payload[task]={'fs':500,'unit':'uV','leads':dict(zip(official_output.LEADS,raw))}
            path=official_output.save_result(userid,**payload,output_dir=output)
            # Official reader validates the actual disk file, not just our arrays.
            checked=official_output.read_result(path,expected_userid=userid)
            del checked
            report={'subject_id':userid,'record_key':record_key,'anchor':anchor_info,
                'history_selection':{'machine':machine_info,'body':body_info},'context':status,
                'watch_records':len(person['watch']),'watch_used':False,'prediction_sha256':sha(path),
                'target_used_at_inference':False,'shape_per_task':[12,60000],'unit':'uV','fs':500}
            write(audit/(userid+'.json'),report);subjects.append({'subject_id':userid,'csv':path.name,'sha256':sha(path)})
            # Caches are useful for repeated predictions of one subject; never retain
            # every person's history/template for the complete test population.
            if hasattr(engine,'engine'):
                engine.engine.router._fir_cache.clear()
                body_adapter=engine.engine.router.body_adapter
                if body_adapter is not None: body_adapter.cache.clear()
            print('Saved '+str(path),flush=True)
        expected={p['subject_id'] for p in subjects}
        if {p.stem for p in output.glob('*.csv')}!=expected: raise ValueError('Missing or extra output subjects')
        write(audit/'completed.json',{'completed':True,'subjects':subjects,'training_started':False,
              'target_used_at_inference':False,'official_output_readback_checked':True})
    except Exception as exc:
        write(audit/'failed.json',{'completed':False,'completed_subjects':subjects,'error':str(exc)})
        raise
    return subjects


def main(argv=None):
    root=executable_root();resources=resource_root()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default=str(root/'test_data'))
    p.add_argument('--output-dir',default=str(root/'results'))
    p.add_argument('--submission-config',default=str(root/'submission.json' if (root/'submission.json').is_file() else resources/'configs/b5_submission.json'))
    p.add_argument('--profile',default=str(resources/'configs/b5_final.json'))
    p.add_argument('--repository-root',default=str(root))
    p.add_argument('--checkpoint');p.add_argument('--body-checkpoint')
    p.add_argument('--device',choices=('auto','cpu','cuda'))
    p.add_argument('--batch-size',type=int)
    p.add_argument('--inspect-input',action='store_true',help='Read organizer format only; works on the 4-point example; no model required')
    p.add_argument('--check-model',action='store_true',help='Verify and load both selected models only; no ECG forward or training')
    args=p.parse_args(argv);settings=load_settings(args.submission_config)
    if args.batch_size is not None:
        if args.batch_size<1: p.error('batch-size must be positive')
        settings['batch_size']=args.batch_size
    if args.device: settings['device']=args.device
    if args.inspect_input:
        print(json.dumps(OfficialDataset(args.data_root,settings).inspect(),ensure_ascii=False,indent=2));return
    if settings['protocol'] in (G120_C3_PROTOCOL,TCN_PROTOCOL) and (args.checkpoint or args.body_checkpoint):
        p.error('Selected deployment protocol pins model files in submission.json; do not override old base/body arguments')
    if settings['protocol']==G120_C3_PROTOCOL:
        from .g120_submission import Predictor,resolve_assets
        assets=resolve_assets(args.repository_root,settings)
    elif settings['protocol']==TCN_PROTOCOL:
        from .tcn_submission import Predictor,resolve_assets
        assets=resolve_assets(args.repository_root,args.profile,settings)
    else:
        checkpoint=args.checkpoint or settings['base_checkpoint']
        body=args.body_checkpoint or settings['body_checkpoint']
        assets=resolve_assets(args.repository_root,args.profile,checkpoint,body)
        from .predict import Predictor
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from baselines.B5.runtime import device_from_name
    engine=Predictor(device_from_name(settings['device']),assets=assets)
    if args.check_model:
        print(json.dumps({'model_loaded':True,'model_forward_checked':False,'training_started':False,
                          **assets.report()},ensure_ascii=False,indent=2));return
    run(args.data_root,args.output_dir,settings,assets=assets,device=settings['device'],engine=engine)


if __name__=='__main__': main()
