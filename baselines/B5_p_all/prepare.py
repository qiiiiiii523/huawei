"""Read a scored P best and its exported bundle; freeze a NEW paired Huawei-only study."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
from baselines.B5_best import SAMPLING
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle, bundle_slow_beta
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.validate import pair_indices
from baselines.B5_public_slow import PROTOCOL as P_PROTOCOL
from baselines.B5_public_slow.common import verify_entries, read_record, validate_record
from baselines.B5_public_slow.prepare import source_study, compatibility
from . import PROTOCOL, GROUPS
from .common import code_digest
from .multiscale import multiscale_settings, LEGACY_FUNCTIONS_SHA256

DEFAULT_SOURCE = '/home/qht/huawei/results/B5-slow-public/P_refine_v1/seed42_lr1e4'


def audit_source(root):
    root = Path(root).resolve()
    paths = {name: root / name for name in ('shared/study.json', 'P/best.pt', 'P/model.pt', 'P/model.json', 'P/completed.json')}
    old, meta, done = (read(paths[k]) for k in ('shared/study.json', 'P/model.json', 'P/completed.json'))
    cp = torch.load(paths['P/best.pt'], map_location='cpu', weights_only=True)
    bundle = load_bundle(paths['P/model.pt'])
    if (old.get('protocol') != P_PROTOCOL or cp.get('protocol') != P_PROTOCOL or cp.get('group') != 'P' or
            cp.get('study_sha256') != sha(paths['shared/study.json']) or cp.get('stage') != 2 or cp.get('step',0) < 1 or
            meta.get('group') != 'P' or meta.get('stage') != 2 or meta.get('step') != cp['step'] or
            meta.get('training_seed') != old['training']['seed'] or cp.get('training_seed') != old['training']['seed'] or
            done.get('completed') is not True or done.get('group') != 'P' or
            old.get('score_protocol') != SCORE_PROTOCOL or bundle.get('reference_score_protocol') != SCORE_PROTOCOL):
        raise ValueError('Need a completed current-QC P-best and its matching portable model')
    if (meta.get('bundle_sha256') != sha(paths['P/model.pt']) or meta.get('checkpoint_sha256') != sha(paths['P/best.pt']) or
            state_sha(cp['model']) != bundle['slow_state_sha256'] or
            cp.get('FM_state_sha256') != bundle['fm_state_sha256'] or
            bundle['sampling'] != SAMPLING or old.get('sampling') != SAMPLING or
            bundle_slow_beta(bundle) != .75 or old.get('beta') != .75):
        raise ValueError('P checkpoint/exported CNN/FM/sampling identity differs')
    if (bundle['slow'].get('protocol') != 'B5-slow-probe-v1' or bundle['slow']['config'].get('group') != 'F' or
            bundle['slow']['config'].get('context_seconds') != 50 or old.get('mean_weight') != 0. or
            old.get('combo_weight') != .1 or old['training']['learning_rate'] != 1e-4):
        raise ValueError('Retain the current S50-F P-best, original objective and 1e-4 recipe')
    score = cp.get('validation_score') or {}
    keys = ('task1_r_missing11', 'task2_r_missing11', 'main_correlation')
    if (score.get('score_protocol') != SCORE_PROTOCOL or meta.get('score') != score or
            any(not np.isfinite(score.get(k,np.nan)) or not -1 <= score[k] <= 1 for k in keys) or
            abs(score[keys[2]] - (score[keys[0]]+score[keys[1]])/2) > 1e-10 or
            done.get('best',{}).get('score') != score or done['best'].get('step') != cp['step']):
        raise ValueError('Source best/export/complete report does not agree on the selected score')
    return old, bundle, score, paths


def audit_cache(item, bundle):
    root = Path(item['root']).resolve()
    if sha(item['manifest']) != item['sha256']:
        raise ValueError('P Huawei cache manifest changed')
    manifest = read(item['manifest'])
    if (manifest.get('completed') is not True or manifest.get('sampling') != SAMPLING or
            manifest.get('FM_optimizer_updates',0) != 0):
        raise ValueError('Need complete original Heun16/K16 fixed-FM cache')
    cache_study_path, cache_study = source_study(root, manifest)
    source_components = cache_study_path.parent / 'components.pt'
    if sha(source_components) != cache_study['bundle_sha256']:
        raise ValueError('Cache source components changed')
    cached = load_bundle(source_components)
    if (cached['fm_state_sha256'] != bundle['fm_state_sha256'] or
            cached['fm']['scales_sha256'] != bundle['fm']['scales_sha256'] or
            cache_study.get('residual_weight') is None):
        raise ValueError('Wrong frozen control100 FM/scales for the selected P')
    files = {str(cache_study_path):sha(cache_study_path),str(source_components):sha(source_components),
             str(Path(item['manifest']).resolve()):item['sha256']}
    for check in cache_study.get('source_checks',{}).values():
        if sha(check['path']) != check['sha256']:
            raise ValueError('Cached Huawei raw data/QC/demographics changed: '+check['path'])
        files[check['path']] = check['sha256']
    train, val = manifest['train'], manifest['validation']
    if (not train or len({e['target_record_id'] for e in train}) != len(train) or
            {e['subject_id'] for e in train} & {e['subject_id'] for e in val}):
        raise ValueError('Missing/duplicate train records or patient leakage')
    lookup = {e['cache_key']:e for e in val}
    if len(lookup) != len(val) or set(manifest['validation_rows']) != {'task1','task2'}:
        raise ValueError('Invalid task/pair cache identities')
    scales = np.asarray(bundle['fm']['scales']['d12'],np.float32)
    target_sources, quality = {}, {}
    for split in ('train','validation'):
        verify_entries(root,manifest[split])
        for e in manifest[split]:
            record = read_record(root,e)
            validate_record(record,scales,huawei=True,training=split=='train')
            if split=='validation':
                previous=target_sources.setdefault(e['target_record_id'],e)
                if previous is not e:
                    old=read_record(root,previous)
                    for k in ('anchor','numeric','sex','field_mask','age_topcoded','slow_target','target_uV','mask'):
                        if not np.array_equal(record[k],old[k]):
                            raise ValueError('Repeated validation target changes inputs/labels/QC')
                quality[e['cache_key']]=bool(record['mask'][0])
    populations={}
    seen=set()
    for task,rows in manifest['validation_rows'].items():
        devices={}
        for ids in pair_indices(rows):
            row=rows[ids[0]]; key=task+':'+row['pair_id']; entry=lookup[key]; seen.add(key)
            if entry['target_record_id']!=row['target_record_id'] or entry['subject_id']!=row['subject_id']:
                raise ValueError('Validation cache and metadata differ')
            d=devices.setdefault(row['input_type'],{'pairs':0,'eligible_pairs':0,'targets':set()})
            d['pairs']+=1
            if quality[key]:
                d['eligible_pairs']+=1; d['targets'].add(entry['target_record_id'])
        for d in devices.values():d['eligible_unique_targets']=len(d.pop('targets'))
        populations[task]=devices
    if seen!=set(lookup):raise ValueError('Orphan validation cache')
    return manifest,files,populations,cache_study


def prepare(args):
    output,source=Path(args.output).resolve(),Path(args.source_run).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use an empty NEW shared directory')
    if type(args.updates) is not int or args.updates<1 or type(args.validate_every) is not int or args.validate_every<1 or not 0<=args.training_seed<2**32:
        raise ValueError('Need positive update/validation budgets and valid training seed')
    print('Checking selected P best/export and read-only Huawei cache; no public training or FM sampling',flush=True)
    old,bundle,score,paths=audit_source(source)
    item=dict(old['huawei']); cache_root=Path(item['root']).resolve()
    for p in (source,cache_root):
        if output==p or p in output.parents or output in p.parents:raise ValueError('Use output outside all source directories')
    manifest,files,populations,cache_study=audit_cache(item,bundle)
    if old['residual_weight']!=cache_study['residual_weight']:
        raise ValueError('Original F coefficient differs')
    ms=multiscale_settings(cache_root,manifest['train'],.1)
    ms['enabled_bands']=[0,1,2]
    for p in paths.values():files[str(p)]=sha(p)
    cfg={k:old['training'][k] for k in ('learning_rate','weight_decay','batch_records','gradient_clip')}
    if cfg['batch_records']!=2:raise ValueError('Preserve two equal-record microbatches per update')
    cfg.update(updates=args.updates,validate_every=args.validate_every,seed=args.training_seed)
    checked=compatibility(bundle)
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(paths['P/model.pt'],output/'components.pt')
    study={'protocol':PROTOCOL,'score_protocol':SCORE_PROTOCOL,'code_sha256':code_digest(),
           'bundle_sha256':sha(output/'components.pt'),'initial_slow_state_sha256':bundle['slow_state_sha256'],
           'source_files':files,'source_checkpoint':str(paths['P/best.pt']),'source_bundle':str(paths['P/model.pt']),
           'source_training_seed':old['training']['seed'],'source_stage':2,'source_step':read(paths['P/model.json'])['step'],
           'source_code_sha256':old['code_sha256'],'reference_score':score,'training':cfg,'groups':list(GROUPS),
           'huawei':item,'sampling':dict(SAMPLING),'beta':.75,'residual_weight':old['residual_weight'],
           'combo_weight':old['combo_weight'],'mean_weight':0.,'multiscale':ms,
           'legacy_scale_functions_sha256':LEGACY_FUNCTIONS_SHA256,
           'selection':'initial_P_best_or_each_observed_update_by_final_mean_r',
           'optimizer_initialization':'fresh_AdamW_same_all_arms', 'sampler_seed':args.training_seed+200}
    write(output/'study.json',study)
    report={'ready':True,'training_started':False,'optimizer_updates':0,'FM_optimizer_updates':0,
            'public_training_updates':0,'source_bundle':study['source_bundle'],'source_step':study['source_step'],
            'source_training_seed':study['source_training_seed'],'reference_score':score,'training':cfg,
            'train_records':len(manifest['train']),'validation_unique_targets':len({e['target_record_id'] for e in manifest['validation']}),
            'validation_populations':populations,'multiscale_weights':{'control':0.,'all':.1},
            'same_initial_slow_state_sha256':bundle['slow_state_sha256'],**checked}
    write(output/'preflight.json',report)
    print(report,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',default=DEFAULT_SOURCE)
    p.add_argument('--updates',type=int,default=390)
    p.add_argument('--validate-every',type=int,default=78)
    p.add_argument('--training-seed',type=int,default=43)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(min(2,torch.get_num_threads()));prepare(a)


if __name__=='__main__':main()
