"""Reassess saved B5 candidates and slow hybrids under identical full/QC scoring; no training."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from ecg12gen.evaluate import (evaluate_record_predictions,write_report,write_competition_score,
    competition_score,COMPETITION_SCORE_PROTOCOL,QC_COMPETITION_SCORE_PROTOCOL)
from .reassessment_cache import (SAMPLING,read,write,csv_read,csv_write,state_digest,payload_digest,
    file_sha256,checked_arrays,verified_prepared,trusted_cached_evaluation,evaluation_provenance)

CANDIDATES=('best','best_r','best_bonus','last')


def scoring(summaries,qc):
    a,b=summaries['task1'],summaries['task2']
    return competition_score(a['r_missing11'],b['r_missing11'],b['ecg_machine_d6_task2_missing_lead_mean_rmse_uV'],
        b['body_scale_d6_task2_missing_lead_mean_rmse_uV'],score_protocol=QC_COMPETITION_SCORE_PROTOCOL if qc else COMPETITION_SCORE_PROTOCOL)


def references(config,checkpoint,output):
    """Read current validation data once; no model forward or target transformation."""
    from .data import common_config
    from ecg12gen.dataset import JointAnchorDataset
    common=common_config(config)
    result={}
    for task in ('task1','task2'):
        dataset=JointAnchorDataset(common,task,'validation')
        raw_rows=[dataset._rows[i] for i in dataset._indices]
        import hashlib
        digest=hashlib.sha256(json.dumps(raw_rows,sort_keys=True).encode()).hexdigest()
        if digest!=checkpoint['manifests'][task]: raise ValueError('Validation population changed: '+task)
        folder=output/task;folder.mkdir(parents=True,exist_ok=True)
        target=np.lib.format.open_memmap(folder/'target_uV.npy',mode='w+',dtype=np.float32,shape=(len(dataset),12,5000))
        quality=np.zeros((len(dataset),12),bool);rows=[]
        for i in range(len(dataset)):
            sample=dataset[i];row=raw_rows[i]
            target[i]=sample.Y_12lead
            mask=np.asarray(sample.target_quality_mask)
            if mask.shape!=(12,) or not np.isin(mask,[0,1]).all(): raise ValueError('Invalid current target quality mask')
            quality[i]=mask
            rows.append({k:row[k] for k in ('pair_id','target_record_id','subject_id','start_sample_500hz','input_type')})
        target.flush();np.save(folder/'target_quality_mask.npy',quality);csv_write(folder/'window_metadata.csv',rows)
        result[task]={'target':target,'quality':quality,'rows':rows,'manifest':digest}
    return result


def score_evaluation(evaluation,reference,output,target_evaluation=None):
    """The raw prediction is identical in both scores; only eligibility masks differ."""
    results={name:{} for name in ('full_population','target_lead_qc')}
    inputs={}
    for task in ('task1','task2'):
        folder=Path(evaluation)/task
        fallback=Path(target_evaluation)/task if target_evaluation else None
        p,y,rows,q=checked_arrays(folder,reference[task],fallback)
        for label,apply_qc in (('full_population',False),('target_lead_qc',True)):
            summary,details=evaluate_record_predictions(p,y,task,rows,target_quality_mask=q if apply_qc else None)
            summary.update(checkpoint_selection_metric='competition_total_score',sampling=SAMPLING,
                           target_used_at_inference=False,posthoc_reassessment=True)
            write_report(output/label/task,summary,details,title='Same raw-uV predictions; '+label)
            results[label][task]=summary
        csv_write(output/'aligned_metadata'/task/'window_metadata.csv',rows)
        (output/'aligned_metadata'/task).mkdir(parents=True,exist_ok=True)
        np.save(output/'aligned_metadata'/task/'target_quality_mask.npy',q)
        source=fallback if fallback else folder
        inputs[task]={name:{'path':str(path.resolve()),'sha256':file_sha256(path)} for name,path in (
            ('prediction',folder/'prediction_uV.npy'),('target',source/'target_uV.npy'),('metadata',source/'window_metadata.csv'))}
        # Windows permits releasing these files before temporary test cleanup.
        p._mmap.close();y._mmap.close()
    scores={}
    for label,apply_qc in (('full_population',False),('target_lead_qc',True)):
        scores[label]=scoring(results[label],apply_qc)
        write_competition_score(output/label,scores[label])
    write(output/'scores.json',scores)
    return scores,inputs


def generate(config,checkpoint_path,checkpoint,device,output):
    """Use the existing B5 validation path; do not construct an optimizer."""
    from .checkpoint import checkpoint_preprocessor,model_from_checkpoint
    from .data import preprocessing_config
    from .runtime import build_validation_datasets,validate,seed_all,device_from_name
    cfg=copy.deepcopy(config);cfg['sampling']=dict(SAMPLING)
    seed_all(42,bool(cfg['training']['deterministic']))
    processor=checkpoint_preprocessor(checkpoint,preprocessing_config(cfg))
    datasets=build_validation_datasets(cfg,processor)
    for task,dataset in datasets.items():
        if dataset.manifest_digest!=checkpoint['manifests'][task]: raise ValueError('Checkpoint validation population changed')
    model=model_from_checkpoint(checkpoint,device_from_name(device))
    validate(model,datasets,cfg,processor,device_from_name(device),output)
    write(output/'inference_sampling.json',evaluation_provenance(output,checkpoint_path,checkpoint))
    del model


def slow_entries(values,prepared,base_sha):
    entries=[]
    for value in values:
        label,separator,raw=value.partition('=')
        if not separator or label not in ('S50','S50-F'): raise ValueError('Use --slow-evaluation S50=/path or S50-F=/path')
        if any(e['label']==label for e in entries): raise ValueError('Duplicate slow evaluation label')
        folder=Path(raw).resolve();marker=read(folder/'evaluation.json')
        if not prepared: raise ValueError('Slow hybrids require --prepared for frozen target/order provenance')
        if (marker.get('protocol')!='B5-slow-probe-v1' or not marker.get('completed') or
                marker.get('context_seconds')!=50 or marker.get('target_used_at_inference') is not False or
                marker.get('B5_checkpoint_sha256')!=base_sha or marker.get('prepared_sha256')!=prepared['_sha256']):
            raise ValueError('Slow hybrid belongs to another source model/data/context')
        checkpoint=Path(marker['probe_checkpoint'])
        if file_sha256(checkpoint)!=marker['probe_checkpoint_sha256']: raise ValueError('Slow checkpoint changed after evaluation')
        if marker.get('group')!=('F' if label=='S50-F' else 'C'): raise ValueError('Slow label/group mismatch')
        for task in ('task1','task2'):
            if not (folder/'learned_slow_hybrid'/task/'prediction_uV.npy').is_file():
                raise FileNotFoundError('Hybrid predictions missing; run its original evaluation first')
        predictions={task:file_sha256(folder/'learned_slow_hybrid'/task/'prediction_uV.npy') for task in ('task1','task2')}
        entries.append({'label':label,'folder':folder,'marker':marker,'marker_sha256':file_sha256(folder/'evaluation.json'),
                        'prediction_sha256':predictions})
    return entries


def run(args):
    from .checkpoint import load_checkpoint
    from .config import load_config,ModelConfig
    from .data import common_config,demographics_path
    run_dir=Path(args.b5_run).resolve();output=Path(args.output_dir).resolve()
    config=load_config(args.config) if args.config else read(run_dir/'resolved_config.json')
    if args.data_repo: config['repository_root']=str(Path(args.data_repo).resolve())
    config['sampling']=dict(SAMPLING)
    checkpoints={name:load_checkpoint(run_dir/(name+'.pt')) for name in CANDIDATES}
    hashes={name:file_sha256(run_dir/(name+'.pt')) for name in CANDIDATES}
    emas={name:state_digest(cp['ema']) for name,cp in checkpoints.items()}
    best=checkpoints['best']
    for name,cp in checkpoints.items():
        if cp.get('cache_protocol')!='independent-record-context-v1' or cp['architecture_hash']!=ModelConfig.from_dict(config['model']).fingerprint:
            raise ValueError('Use compatible main-v2 B5-U checkpoints')
        if any(cp['manifests'][task]!=best['manifests'][task] for task in ('task1','task2')):
            raise ValueError('Four checkpoints must use the same validation population')
        if cp['scales_sha256']!=best['scales_sha256']: raise ValueError('Four checkpoints must use the same signal scales')
        if cp['manifests']['demographics_csv']!=file_sha256(demographics_path(config)):
            raise ValueError('Current demographics differ from training; cannot silently reuse old conditions')
    common=common_config(config)
    qc_path=common.path('device_interpretation_qc_csv')
    # Bind all current validation caches and split/pair metadata, not just QC.
    source_files={'qc':qc_path,'demographics':demographics_path(config),'split':common.path('subject_split_csv')}
    for task in ('task1','task2'):
        source_files[task+'_metadata']=common.path(task+'_output')/(task+'_window_metadata.csv')
        source_files[task+'_pairs']=common.path(task+'_pair_manifest_csv')
        source_files[task+'_target']=common.path(task+'_output')/(task+'_validation_target.npy')
    sources={key:{'path':str(path),'sha256':file_sha256(path)} for key,path in source_files.items()}
    prepared=verified_prepared(args.prepared,run_dir/'best.pt') if args.prepared else None
    if prepared: prepared['_sha256']=file_sha256(Path(args.prepared)/'prepared.json')
    slow=slow_entries(args.slow_evaluation,prepared,hashes['best'])
    request={'checkpoints':hashes,'ema':emas,'sampling':SAMPLING,'config':config,'sources':sources,
        'prepared_sha256':prepared['_sha256'] if prepared else None,
        'slow':{e['label']:{'marker':e['marker_sha256'],'prediction':e['prediction_sha256']} for e in slow},
        'implementation':{name:file_sha256(Path(__file__).parent/name) for name in ('reassess_qc.py','reassessment_cache.py')},
        'evaluator_sha256':file_sha256(Path(__file__).resolve().parents[2]/'ecg12gen/evaluate.py')}
    signature=payload_digest(request)
    if output.exists() and any(output.iterdir()):
        if not args.resume: raise FileExistsError('Use a new output directory, or --resume for this exact reassessment')
        if read(output/'request.json')!=request: raise ValueError('Resume changed candidates/data/QC/implementation/settings')
    else:
        output.mkdir(parents=True,exist_ok=True);write(output/'request.json',request)
    print('Reading current validation caches and target quality masks; no training.',flush=True)
    reference=references(config,best,output/'reference')
    rows=[]
    candidates=[folder for folder in sorted(run_dir.iterdir()) if folder.is_dir() and folder.name.startswith('evaluation')]
    candidates.extend(sorted(run_dir.glob('*/generated/*')))
    if prepared: candidates.append(Path(prepared['evaluation_dir']))
    # Reuse independently verified outputs from an interrupted reassessment too.
    candidates.extend(sorted((output/'generated').glob('*')))
    completed_emas={}
    try:
        for name in CANDIDATES:
            cp=checkpoints[name];checkpoint_path=run_dir/(name+'.pt')
            folder=next((p for p in candidates if trusted_cached_evaluation(p,checkpoint_path,emas[name],prepared,emas['best'])),None)
            if emas[name] in completed_emas: folder=completed_emas[emas[name]]
            if folder is None:
                if not args.generate_missing:
                    raise FileNotFoundError(f'{name}: no checkpoint-hash-verified K16 predictions. Add --generate-missing; this only evaluates, not trains.')
                folder=output/'generated'/name
                if folder.exists(): folder.rename(folder.with_name(name+'_interrupted_'+str(time.time_ns())))
                print('Generating Heun16/K16 from existing '+str(checkpoint_path),flush=True)
                generate(config,checkpoint_path,cp,args.device,folder)
            completed_emas[emas[name]]=folder
            report=output/'B5'/name
            scores,inputs=score_evaluation(folder,reference,report)
            provenance={'completed':True,'request_sha256':signature,'checkpoint':str(checkpoint_path),
                'checkpoint_sha256':hashes[name],'model_ema_sha256':emas[name],'epoch':cp['epoch'],
                'sampling':SAMPLING,'evaluation':str(folder),'inputs':inputs,'scores':scores,'training_started':False}
            write(report/'reassessment.json',provenance)
            rows.extend({'model':'B5','candidate':name,'epoch':int(cp['epoch'])+1,'score_view':view,**score} for view,score in scores.items())
            print(name,scores['target_lead_qc'],flush=True)
        for entry in slow:
            report=output/entry['label'];folder=entry['folder']/'learned_slow_hybrid'
            scores,inputs=score_evaluation(folder,reference,report,prepared['evaluation_dir'])
            write(report/'reassessment.json',{'completed':True,'request_sha256':signature,'result_kind':'fixed_output_hybrid_not_FM_integration',
                'slow_evaluation':str(entry['folder']),'source_b5_checkpoint_sha256':hashes['best'],
                'slow_checkpoint_sha256':entry['marker']['probe_checkpoint_sha256'],'sampling':SAMPLING,'inputs':inputs,'scores':scores,'training_started':False})
            rows.extend({'model':entry['label']+'_output_hybrid','candidate':Path(entry['marker']['probe_checkpoint']).stem,
                         'epoch':entry['marker']['epoch'],'score_view':view,**score} for view,score in scores.items())
        csv_write(output/'comparison.csv',rows)
        eligible=[r for r in rows if r['model']=='B5' and r['score_view']=='target_lead_qc']
        selection={label:max(eligible,key=lambda r:r[field]) for label,field in (
            ('best_total_among_saved','competition_total_score'),('best_r_among_saved','main_correlation'),('best_bonus_among_saved','task2_rmse_bonus_score'))}
        write(output/'selection.json',{'scope':'only_the_four_existing_candidates_not_all_historical_epochs',
            'checkpoint_files_modified':False,'checkpoint_sha256':hashes,'selected':selection})
        lines=['# 已有权重及慢曲线组合QC重评','','同一批预测同时报告完整集合v2和目标导联QC v3，不混合排名。未训练、未改写旧权重。',
            '','|模型|候选|轮次|评分范围|Task1 r|Task2 r|机器RMSE/μV|体脂秤RMSE/μV|总分/60|',
            '|---|---|---:|---|---:|---:|---:|---:|---:|']
        fields=('task1_r_missing11','task2_r_missing11','ecg_machine_d6_task2_missing_lead_mean_rmse_uV',
                'body_scale_d6_task2_missing_lead_mean_rmse_uV','competition_total_score')
        for row in rows: lines.append('|'+ '|'.join([str(row[k]) for k in ('model','candidate','epoch','score_view')]+[f'{row[k]:.6f}' for k in fields])+'|')
        lines+=['','QC总分最高的现存B5候选：'+selection['best_total_among_saved']['candidate'],
            '仅在现存四份候选中选择，不能恢复已丢失epoch的权重；不会覆盖best.pt。',
            'S50/S50-F是固定原B5快分量＋预测慢曲线的输出组合参考，不是FM条件接入成绩。',
            'FM epoch按原检查点0基序号+1显示；慢CNN沿用其报告中的新增轮次。']
        (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(output/'completed.json',{'completed':True,'request_sha256':signature,'training_started':False,'optimizer_updates':0})
    finally:
        for ref in reference.values(): ref['target']._mmap.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b5-run',required=True)
    parser.add_argument('--config',help='Optional YAML; default original resolved_config.json')
    parser.add_argument('--data-repo',help='Actual original repository/data root when using a separate code worktree')
    parser.add_argument('--prepared',help='Original slow-probe prepared directory: verifies source best.pt and old K16 arrays')
    parser.add_argument('--slow-evaluation',action='append',default=[],help='S50=/path/to/evaluation_best_score, repeat for S50-F')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--generate-missing',action='store_true',help='Generate missing/unverifiable B5 predictions; NEVER train')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())


if __name__=='__main__': main()
