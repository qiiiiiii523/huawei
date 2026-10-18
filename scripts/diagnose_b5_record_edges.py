#!/usr/bin/env python3
"""CPU-only regional diagnosis of the OLD frozen initial/B20/C20 predictions."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from diagnose_b5_control100_multiscale import (
    Saved, pairs, key, read, write, sha, corr, moving_average, csv_save,
    mean, ratio, source_header, METHODS, LEADS, RUN, SCORE)

REGIONS = {'first20': (0, 10000), 'middle80': (10000, 50000), 'last20': (50000, 60000)}


def context_layout():
    """Half-open sample/time intervals. Negative or >120s samples use edge repeats."""
    result = []
    for i in range(12):
        start, stop = i * 500 - 1000, i * 500 + 1500
        result.append({'window': i, 'context_start_s': start / 50, 'context_end_s': stop / 50,
                       'real_start_s': max(0, start) / 50, 'real_end_s': min(6000, stop) / 50,
                       'left_repeat_s': max(0, -start) / 50, 'right_repeat_s': max(0, stop-6000) / 50,
                       'output_start_s': i * 10, 'output_end_s': i * 10 + 10,
                       'crop_in_context_samples_50hz': '1000:1500', 'real_output_samples': 500})
    return result


def region_metrics(prediction, target, fixed_fast):
    ps = prediction - fixed_fast
    ys = moving_average(target, 501)  # Smooth full120s BEFORE cropping regions.
    result = []
    for region, (a, b) in REGIONS.items():
        error = ps[a:b] - ys[a:b]
        result.append({'region': region, 'slow_r': corr(ps[a:b], ys[a:b]),
                       'raw_r': corr(prediction[a:b], target[a:b]),
                       'slow_rmse_uV': float(np.sqrt(np.mean(error**2))),
                       'slow_centered_rmse_uV': float(error.std()),
                       'slow_mean_error_uV': float(error.mean()),
                       'slow_target_std_uV': float(ys[a:b].std()),
                       'slow_std_ratio': ratio(ps[a:b].std(), ys[a:b].std())})
    return result


def aggregate(rows):
    identity = ('method', 'task', 'input_type', 'target_record_id', 'subject_id', 'lead', 'region')
    fields = ('slow_r', 'raw_r', 'slow_rmse_uV', 'slow_centered_rmse_uV',
              'slow_mean_error_uV', 'slow_target_std_uV', 'slow_std_ratio')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in identity)].append(row)
    targets = [dict(zip(identity, k), n_pairs=len(v), **{f: mean([r[f] for r in v]) for f in fields})
               for k, v in sorted(groups.items())]
    groups = defaultdict(list)
    for row in targets:
        groups[tuple(row[k] for k in ('method', 'task', 'input_type', 'lead', 'region'))].append(row)
    leads = [dict(zip(('method', 'task', 'input_type', 'lead', 'region'), k), n_targets=len(v),
                  **{f: mean([r[f] for r in v]) for f in fields}) for k, v in sorted(groups.items())]
    groups = defaultdict(list)
    for row in leads:
        groups[tuple(row[k] for k in ('method', 'task', 'input_type', 'region'))].append(row)
    summary = [dict(zip(('method', 'task', 'input_type', 'region'), k),
                    **{f: mean([r[f] for r in v]) for f in fields}) for k, v in sorted(groups.items())]
    return targets, leads, summary


def run(args):
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Keep output outside frozen source run')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a NEW empty output; tee logs beside it')
    refs = {}
    def frozen(path):
        refs[str(path)] = sha(path)
        return read(path)
    study_path, view_path = root/'shared/study.json', root/'shared/cache_view.json'
    study, view = frozen(study_path), frozen(view_path)
    if (study['protocol'] != 'B5-control100-partner-multiscale-reliable-I-v2' or
            study['score_protocol'] != SCORE or study['beta'] != .75 or
            study['sampling'] != {'seed':42, 'solver':'heun', 'steps':16, 'samples':16} or
            not view['completed'] or view['study_sha256'] != sha(study_path)):
        raise ValueError('Expected OLD frozen seed42 study/cache view')
    cache_root = Path(view['validation_root']).resolve()
    cache_json = cache_root/'cache.json'
    refs[str(cache_json)] = sha(cache_json)
    if refs[str(cache_json)] != study['source_files'].get(str(cache_json)):
        raise ValueError('Source manifest changed')
    lookup = {r['cache_key']:r for r in view['validation']}
    if len(lookup) != len(view['validation']):
        raise ValueError('Duplicate cached pair identity')
    output.mkdir(parents=True, exist_ok=True)
    metrics, exclusions, anchor_stats = [], [], []
    seen = set()
    with ExitStack() as stack:
        sources = {}
        for method in METHODS:
            group = 'adapt_multiscale' if method == 'C20' else 'adapt_control100'
            epoch = 0 if method == 'initial' else 20
            folder = root/group/'validation'/f'epoch_{epoch:04d}'
            source_header(folder, group, epoch, study, refs)
            sources[method] = {}
            for task in ('task1','task2'):
                reference = sources['initial'][task].rows if method != 'initial' else None
                sources[method][task] = Saved(folder/task, stack, refs, reference)
        for task in ('task1','task2'):
            initial = sources['initial'][task]
            canonical = {key(r): r for r in view['validation_rows'][task]}
            if len(canonical) != len(initial.rows) or set(canonical) != {key(r) for r in initial.rows}:
                raise ValueError('Study/evaluation windows differ')
            for ids in pairs(initial.rows):
                row = initial.rows[ids[0]]
                if any(any(r[k] != canonical[key(r)][k] for k in ('subject_id','input_type'))
                       for r in (initial.rows[i] for i in ids)):
                    raise ValueError('Canonical subject/device mismatch')
                cache_key = task+':'+row['pair_id']; seen.add(cache_key)
                entry = lookup[cache_key]
                if any(entry[k] != row[k] for k in ('target_record_id','subject_id')):
                    raise ValueError('Cache subject/target mismatch')
                path = (cache_root/entry['file']).resolve()
                if not path.is_relative_to(cache_root) or sha(path) != entry['sha256']:
                    raise ValueError('Frozen cached pair changed')
                refs[str(path)] = entry['sha256']
                with np.load(path, allow_pickle=False) as data:
                    fm, y, q = data['fm_uV'].astype(float), data['target_uV'], data['mask'].astype(bool)
                    if fm.shape != (12,60000) or not np.isfinite(fm).all() or q.shape != (12,):
                        raise ValueError('Need finite fixed FM full120s and lead QC')
                    anchor = data['anchor']
                    if anchor.shape != (1,6000) or not np.isfinite(anchor).all():
                        raise ValueError('S50 anchor must be visible full120s I at50Hz')
                    for name, (a,b) in REGIONS.items():
                        part = anchor[0,a//10:b//10]
                        anchor_stats.append({'task':task,'pair_id':row['pair_id'],'subject_id':row['subject_id'],
                                             'region':name,'input_I_QC_good':bool(q[0]),
                                             'anchor_mean_normalized':float(part.mean()),
                                             'anchor_std_normalized':float(part.std())})
                fixed_fast = fm-moving_average(fm,501)
                for method in METHODS:
                    p, target, mask = sources[method][task].record(ids)
                    if not np.array_equal(target,y) or not np.array_equal(mask,q):
                        raise ValueError('Cached and saved targets/QC differ')
                    if not np.allclose(p[0],fm[0],atol=.002,rtol=1e-6):
                        raise ValueError('Observed I changed')
                    if not q[0]:
                        continue
                    for j, lead in enumerate(LEADS[1:],1):
                        if q[j]:
                            for r in region_metrics(p[j],target[j],fixed_fast[j]):
                                metrics.append({**{k:row[k] for k in ('pair_id','target_record_id','subject_id','input_type')},
                                                'task':task,'method':method,'lead':lead,**r})
                if not q[0]:
                    exclusions.append({'task':task,'pair_id':row['pair_id'],'reason':'bad_input_I'})
                else:
                    for j in range(1,12):
                        if not q[j]:
                            exclusions.append({'task':task,'pair_id':row['pair_id'],'lead':LEADS[j],'reason':'bad_target_lead'})
                print('Checked regional errors:', task, row['target_record_id'], flush=True)
    if seen != set(lookup):
        raise ValueError('Orphan validation cache entries')
    targets, leads, summary = aggregate(metrics)
    for name, rows in (('pair_lead_regions',metrics),('target_lead_regions',targets),('lead_regions',leads),
                       ('region_summary',summary),('S50_context_layout',context_layout()),
                       ('anchor_regions',anchor_stats),('qc_exclusions',exclusions)):
        csv_save(output/(name+'.csv'), rows)
    if any(sha(p) != h for p,h in refs.items()):
        raise ValueError('Sources changed during read-only diagnosis')
    write(output/'edge_audit.json', {'completed':True,'training_started':False,'model_loaded':False,
          'new_FM_inference':False,'source_files':refs,'score_protocol':SCORE,
          'regions':REGIONS,'input_padding':'repeat endpoint,20s each side',
          'S50_window':'50s, stride10s, crop native50Hz [1000:1500]',
          'scoring':'regional metrics are diagnostic; full120s formal scoring unchanged',
          'centered_error':'std of slow prediction minus full-record MA501 target in each region; no test calibration',
          'interpretation':'More edge error alone does not prove padding bug; inspect target variability/input together.'})
    print({'completed':True,'output':str(output),'training_started':False},flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default=RUN)
    p.add_argument('--output',required=True)
    run(p.parse_args())
