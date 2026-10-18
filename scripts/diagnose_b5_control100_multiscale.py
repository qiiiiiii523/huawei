#!/usr/bin/env python3
"""Read-only CPU diagnosis of saved initial/B20/C20 arrays. No torch/model imports.

Reconstruct the actual beta-scaled slow contribution using the SAME fixed FM
cache, verify reliable-I official scores, compare paired targets and bootstrap
patients as clusters. Truth/centering/windows are used ONLY for diagnostics.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.B5_best.common import read, write, sha
from baselines.B5_best.diagnostics_core import corr, moving_average
from ecg12gen.evaluate import (evaluate_record_predictions, write_report,
    correlation_score_from_summaries, write_correlation_score, RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL)

METHODS = ('initial', 'B20', 'C20')
LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
VIEWS = (('task1', 'watch_ecg'), ('task2', 'ecg_machine_d6'), ('task2', 'body_scale_d6'))
RUN = '/home/qht/huawei/results/B5-best-exp-Control100PCC/partner_multiscale_v1'
PROTOCOL = 'B5-control100-multiscale-saved-diagnosis-v1'
SCORE = RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_save(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def ratio(a, b):
    return float(a / b) if b > 1e-10 else None


def key(row):
    return (row['pair_id'], row['target_record_id'], int(row['start_sample_500hz']))


def pairs(rows):
    grouped = defaultdict(list)
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate pair/target/window identities')
    for i, row in enumerate(rows):
        grouped[row['pair_id']].append(i)
    for name in sorted(grouped):
        ids = sorted(grouped[name], key=lambda i: int(rows[i]['start_sample_500hz']))
        if [int(rows[i]['start_sample_500hz']) for i in ids] != list(range(0, 60000, 5000)):
            raise ValueError('Each pair must contain twelve chronological real windows')
        for field in ('target_record_id', 'subject_id', 'input_type'):
            if len({rows[i][field] for i in ids}) != 1:
                raise ValueError('Pair mixes ' + field)
        yield ids


class Saved:
    def __init__(self, folder, stack, references, reference_rows=None):
        self.folder = Path(folder)
        target = self.folder / 'target_uV.npy'
        if not target.is_file():
            refpath = self.folder / 'target_reference.json'
            info = read(refpath)
            target = Path(info['path'])
            if not target.is_absolute():
                target = self.folder / target
            if sha(target) != info['sha256']:
                raise ValueError('Target reference differs from frozen SHA')
            references[str(refpath)] = sha(refpath)
        self.paths = {'prediction': self.folder / 'prediction_uV.npy', 'target': target.resolve(),
                      'mask': self.folder / 'target_quality_mask.npy', 'metadata': self.folder / 'window_metadata.csv'}
        for path in self.paths.values():
            references[str(path)] = sha(path)
        self.rows = csv_read(self.paths['metadata'])
        list(pairs(self.rows))
        self.arrays = {}
        for name in ('prediction', 'target', 'mask'):
            array = np.load(self.paths[name], mmap_mode='r', allow_pickle=False)
            stack.callback(array._mmap.close)
            expected = (len(self.rows), 12) if name == 'mask' else (len(self.rows), 12, 5000)
            if array.shape != expected or (name == 'mask' and not np.isin(array, [0, 1]).all()):
                raise ValueError('Wrong array shape/QC values: ' + str(self.paths[name]))
            self.arrays[name] = array
        self.order = np.arange(len(self.rows))
        if reference_rows is not None:
            lookup = {key(r): i for i, r in enumerate(self.rows)}
            if set(lookup) != {key(r) for r in reference_rows}:
                raise ValueError('Evaluation populations differ')
            self.order = np.array([lookup[key(r)] for r in reference_rows])
            for row, i in zip(reference_rows, self.order):
                if any(row[k] != self.rows[i][k] for k in ('subject_id', 'input_type')):
                    raise ValueError('Patient/device identity mismatch')

    def record(self, ids):
        ids = self.order[ids]
        p, y = [np.concatenate(self.arrays[k][ids], axis=-1).astype(np.float64) for k in ('prediction', 'target')]
        masks = self.arrays['mask'][ids]
        if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.all(masks == masks[0]):
            raise ValueError('Nonfinite waveforms or QC changes within pair')
        return p, y, masks[0].astype(bool)


def bands(slow500):
    """Deployed contribution -> 50Hz block means; approximate training scales."""
    x = np.asarray(slow500, np.float64)
    if x.shape != (60000,):
        raise ValueError('Need a complete 120s lead')
    native = x.reshape(6000, 10).mean(-1)
    five, fifteen = moving_average(native, 251), moving_average(native, 751)
    return (native - five, five - fifteen, fifteen)


def metrics(pred, target, fm, beta):
    fast = fm - moving_average(fm, 501)
    slow = pred - fast  # Not MA501(final prediction).
    ys = moving_average(target, 501)
    error = slow - ys
    out = {'raw_r': corr(pred, target), 'raw_rmse_uV': float(np.sqrt(np.mean((pred-target)**2))),
           'slow_r': corr(slow, ys), 'slow_rmse_uV': float(np.sqrt(np.mean(error**2))),
           'slow_centered_rmse_uV': float(np.std(error)), 'slow_mean_error_uV': float(error.mean()),
           'slow_std_ratio_actual': ratio(slow.std(), ys.std()),
           'slow_std_ratio_raw_CNN': ratio(slow.std()/beta, ys.std()),
           'fixed_fast_r': corr(fast, target - ys),
           'final_fast_r': corr(pred - moving_average(pred, 501), target - ys)}
    for name, p, y in zip(('short', 'middle', 'long'), bands(slow), bands(ys)):
        centered = (p-p.mean()) - (y-y.mean())
        out.update({name+'_r': corr(p, y), name+'_std_ratio': ratio(p.std(), y.std()),
                    name+'_target_std_uV': float(y.std()), name+'_centered_rmse_uV': float(np.sqrt(np.mean(centered**2)))})
    return out


def aggregate(pair_rows):
    identity = ('method', 'task', 'input_type', 'target_record_id', 'subject_id', 'lead')
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[tuple(row[k] for k in identity)].append(row)
    fields = [k for k in pair_rows[0] if k not in (*identity, 'pair_id')]
    target_rows = []
    for keys, rows in sorted(grouped.items()):
        target_rows.append(dict(zip(identity, keys), n_pairs=len(rows), **{k: mean([r[k] for r in rows]) for k in fields}))
    grouped = defaultdict(list)
    for row in target_rows:
        grouped[tuple(row[k] for k in ('method', 'task', 'input_type', 'lead'))].append(row)
    lead_rows = []
    for keys, rows in sorted(grouped.items()):
        out = dict(zip(('method', 'task', 'input_type', 'lead'), keys), n_targets=len(rows))
        for field in fields:
            values = [r[field] for r in rows if r[field] is not None]
            out[field+'_mean'] = mean(values)
            out[field+'_median'] = float(np.median(values)) if values else None
            out[field+'_n_defined'] = len(values)
        lead_rows.append(out)
    return target_rows, lead_rows


def comparisons(target_rows):
    identity = ('task', 'input_type', 'target_record_id', 'subject_id', 'lead')
    grouped = defaultdict(dict)
    for row in target_rows:
        grouped[tuple(row[k] for k in identity)][row['method']] = row
    selected = ('raw_r', 'slow_r', 'slow_std_ratio_actual', 'slow_std_ratio_raw_CNN',
                'slow_mean_error_uV', 'short_r', 'middle_r', 'long_r',
                'short_std_ratio', 'middle_std_ratio', 'long_std_ratio', 'final_fast_r')
    deltas = []
    for keys, rows in sorted(grouped.items()):
        if set(rows) != set(METHODS):
            raise ValueError('Missing paired method for target/lead')
        out = dict(zip(identity, keys), n_pairs=rows['initial']['n_pairs'])
        for field in selected:
            for method in METHODS:
                out[method+'_'+field] = rows[method][field]
            for suffix, a, b in (('C_minus_B', 'C20', 'B20'), ('C_minus_initial', 'C20', 'initial')):
                out[field+'_'+suffix] = (rows[a][field]-rows[b][field]
                                         if rows[a][field] is not None and rows[b][field] is not None else None)
        deltas.append(out)
    return deltas


def target_diagnostics(deltas):
    grouped = defaultdict(list)
    for row in deltas:
        grouped[tuple(row[k] for k in ('task','input_type','target_record_id','subject_id'))].append(row)
    result = []
    for keys, rows in sorted(grouped.items()):
        out = dict(zip(('task','input_type','target_record_id','subject_id'), keys), n_qc_good_leads=len(rows))
        for method in METHODS:
            out[method+'_target_diagnostic_r'] = mean([r[method+'_raw_r'] for r in rows])
        for suffix, a, b in (('C_minus_B', 'C20', 'B20'), ('C_minus_initial', 'C20', 'initial')):
            out[suffix] = (out[a+'_target_diagnostic_r']-out[b+'_target_diagnostic_r']
                           if out[a+'_target_diagnostic_r'] is not None and out[b+'_target_diagnostic_r'] is not None else None)
        result.append(out)
    patients = []
    grouped = defaultdict(list)
    for row in result:
        grouped[row['subject_id']].append(row)
    for patient, rows in sorted(grouped.items()):
        out = {'subject_id': patient, 'n_views': len(rows), 'n_physical_targets': len({r['target_record_id'] for r in rows})}
        for method in METHODS:
            out[method+'_patient_diagnostic_r'] = mean([r[method+'_target_diagnostic_r'] for r in rows])
        for suffix, a, b in (('C_minus_B', 'C20', 'B20'), ('C_minus_initial', 'C20', 'initial')):
            out[suffix] = (out[a+'_patient_diagnostic_r']-out[b+'_patient_diagnostic_r']
                          if out[a+'_patient_diagnostic_r'] is not None and out[b+'_patient_diagnostic_r'] is not None else None)
        patients.append(out)
    return result, patients


def score_arrays(target_rows):
    patients = sorted({r['subject_id'] for r in target_rows})
    sums = np.zeros((3, 3, 11, len(patients)))
    counts = np.zeros_like(sums)
    for row in target_rows:
        if row['raw_r'] is not None:
            idx = (METHODS.index(row['method']), VIEWS.index((row['task'], row['input_type'])),
                   LEADS[1:].index(row['lead']), patients.index(row['subject_id']))
            sums[idx] += row['raw_r']
            counts[idx] += 1
    return patients, sums, counts


def weighted_scores(sums, counts, weights):
    numerator = np.einsum('mvls,ks->kmvl', sums, weights)
    denominator = np.einsum('mvls,ks->kmvl', counts, weights)
    valid = (denominator > 0).all(axis=(1, 2, 3))
    values = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=values, where=denominator > 0)
    views = values.mean(-1)
    return .5*views[:, :, 0] + .25*views[:, :, 1] + .25*views[:, :, 2], valid


def bootstrap(target_rows, repetitions, seed):
    patients, sums, counts = score_arrays(target_rows)
    point, valid = weighted_scores(sums, counts, np.ones((1, len(patients))))
    if not valid[0]:
        raise ValueError('Official hierarchy cannot be reconstructed from target-level raw r')
    result = {'method': 'paired_subject_cluster_percentile', 'seed': seed, 'n_patients': len(patients),
              'point_R': dict(zip(METHODS, map(float, point[0]))), 'training_seed_replication': False,
              'conditional_on_fixed_selected_checkpoints': True, 'post_selection_bias_corrected': False}
    if repetitions == 0 or len(patients) < 2:
        return {**result, 'available': False, 'reason': 'disabled_or_too_few_patients'}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(patients), np.full(len(patients), 1/len(patients)), size=repetitions)
    values, valid = weighted_scores(sums, counts, weights)
    values = values[valid]
    result.update(available=bool(len(values)), requested_draws=repetitions,
                  valid_draws=len(values), invalid_empty_device_or_lead_draws=int((~valid).sum()))
    if not len(values):
        return result
    for name, a, b in (('C_minus_B', 2, 1), ('C_minus_initial', 2, 0)):
        delta = values[:, a]-values[:, b]
        result[name] = {'point_delta_R': float(point[0,a]-point[0,b]),
                        'percentile_95_low': float(np.quantile(delta, .025)),
                        'percentile_95_high': float(np.quantile(delta, .975)),
                        'fraction_draws_positive_NOT_p_value': float(np.mean(delta > 0))}
    return result


def source_header(folder, group, epoch, study, references):
    report_path = folder / 'evaluation.json'
    report = read(report_path)
    references[str(report_path)] = sha(report_path)
    note_path = folder / 'experiment.json'
    note = read(note_path)
    references[str(note_path)] = sha(note_path)
    if (not report.get('completed') or report.get('group') != group or report.get('epoch') != epoch
            or report.get('score', {}).get('score_protocol') != SCORE
            or report.get('sampling') != study['sampling'] or not report.get('FM_frozen')
            or note.get('beta') != study['beta']):
        raise ValueError('Wrong epoch/group/score scope/FM/sampling: ' + str(folder))
    return report['score']


def choose_plots(deltas, limit):
    specs = (('task1','watch_ecg','V3','raw_r_C_minus_initial',True),
             ('task1','watch_ecg','V6','raw_r_C_minus_initial',True),
             ('task2','body_scale_d6','V1','raw_r_C_minus_initial',False),
             ('task2','body_scale_d6','V6','raw_r_C_minus_initial',False),
             ('task2','body_scale_d6','V5','raw_r_C_minus_B',True),
             ('task1','watch_ecg','V5','raw_r_C_minus_B',False))
    selected, seen = [], set()
    for task, device, lead, field, largest in specs:
        candidates = [r for r in deltas if (r['task'],r['input_type'],r['lead']) == (task,device,lead)
                      and r[field] is not None and (r['subject_id'], lead) not in seen]
        if candidates:
            row = sorted(candidates, key=lambda r: (r[field], r['target_record_id']), reverse=largest)[0]
            selected.append({**row, 'selection_field': field, 'selection': 'largest' if largest else 'smallest'})
            seen.add((row['subject_id'], lead))
        if len(selected) >= limit:
            break
    return selected


def plot_case(output, row, waves, target, fm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lead = LEADS.index(row['lead'])
    y = target[lead]
    ys = moving_average(y, 501)
    fast = fm[lead]-moving_average(fm[lead], 501)
    slow = {m: waves[m][lead]-fast for m in METHODS}
    true_bands, pred_bands = bands(ys), {m: bands(v) for m,v in slow.items()}
    time = np.arange(60000)/500
    local_ref = 'B20' if row['selection_field'].endswith('_B') else 'initial'
    gains = []
    for start in range(0, 60000, 5000):
        r1, r0 = corr(waves['C20'][lead,start:start+5000], y[start:start+5000]), corr(waves[local_ref][lead,start:start+5000], y[start:start+5000])
        gains.append(r1-r0 if r1 is not None and r0 is not None else 0.)
    index = int(np.argmax(gains) if row['selection'] == 'largest' else np.argmin(gains))
    start = index*5000
    colors = {'initial':'#888888','B20':'#2878b5','C20':'#cf4d2d'}
    fig, axes = plt.subplots(6, 1, figsize=(15, 17))
    axes[0].plot(time[::10], ys[::10], 'k', label='True slow MA501')
    for m in METHODS:
        axes[0].plot(time[::10], slow[m][::10], color=colors[m], label=m+' actual beta=.75 slow')
    axes[0].set_title('Actual slow contribution; original voltage retained')
    for i, name in enumerate(('Approx <5s changes','Approx 5..15s changes','Approx >15s trend'), 1):
        t = np.arange(6000)/50
        truth = true_bands[i-1]
        axes[i].plot(t, truth-truth.mean(), 'k', label='Target')
        for m in METHODS:
            value = pred_bands[m][i-1]
            axes[i].plot(t, value-value.mean(), color=colors[m], label=m)
        axes[i].set_title(name+'; mean removed FOR DISPLAY ONLY')
    segment = slice(start,start+5000)
    axes[4].plot(time[segment], y[segment], 'k', label='Target ECG')
    for m in METHODS:
        axes[4].plot(time[segment], waves[m][lead,segment], color=colors[m], label=m, alpha=.85)
        axes[5].plot(time[::10], (waves[m][lead]-y)[::10], color=colors[m], label=m+' minus target')
    axes[4].set_title('10s final ECG: truth-selected illustrative window, NOT official scoring')
    axes[5].set_title('Final ECG error over full120s')
    for i, ax in enumerate(axes):
        ax.set_ylabel('uV'); ax.set_xlabel('Time (s)'); ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=8)
        if i != 4:
            for edge in range(10,120,10):
                ax.axvline(edge, color='gray', ls='--', lw=.5, alpha=.3)
    fig.suptitle(f"{row['subject_id']} | {row['target_record_id']} | {row['task']} {row['input_type']} {row['lead']}")
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(output, dpi=130); plt.close(fig)
    return start/500


def run(args):
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Keep diagnostic output separate from the source run')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty output; tee logs beside it')
    if args.bootstrap < 0 or args.max_plots < 1:
        raise ValueError('Invalid bootstrap/plot budget')
    if args.plots:
        import matplotlib  # fail before costly scoring if plotting dependency is missing
    references = {}
    study_path, view_path = root/'shared/study.json', root/'shared/cache_view.json'
    study, view = read(study_path), read(view_path)
    references[str(study_path)], references[str(view_path)] = sha(study_path), sha(view_path)
    if (study.get('protocol') != 'B5-control100-partner-multiscale-reliable-I-v2'
            or study.get('score_protocol') != SCORE or study.get('beta') != .75
            or study.get('sampling') != {'seed':42,'solver':'heun','steps':16,'samples':16}
            or view.get('study_sha256') != sha(study_path) or not view.get('completed')):
        raise ValueError('Expected the completed three-arm, reliable-I study/cache view')
    initialization = []
    for group in ('adapt_control100','adapt_multiscale'):
        path = root/group/'initialization.json'
        item = read(path)
        references[str(path)] = sha(path)
        if (item.get('study_sha256') != sha(study_path) or item.get('cache_sha256') != sha(view_path)
                or item.get('score_protocol') != SCORE or item.get('sampling') != study['sampling']
                or item.get('beta') != study['beta']):
            raise ValueError('Initialization does not identify this study/cache/scoring scope')
        initialization.append(item)
    for field in ('initial_slow_state_sha256','fixed_FM_state_sha256','training'):
        if initialization[0].get(field) != initialization[1].get(field):
            raise ValueError('B/C do not share initialization and training settings')
    cache_root = Path(view['validation_root']).resolve()
    cache_manifest = cache_root/'cache.json'
    if sha(cache_manifest) != study['source_files'].get(str(cache_manifest)):
        raise ValueError('Frozen control100 source cache manifest changed')
    references[str(cache_manifest)] = sha(cache_manifest)
    entries = {r['cache_key']: r for r in view['validation']}
    if len(entries) != len(view['validation']):
        raise ValueError('Duplicate validation pair cache keys')
    for i, entry in enumerate(entries.values(),1):
        path = (cache_root/entry['file']).resolve()
        if not path.is_relative_to(cache_root) or sha(path) != entry['sha256']:
            raise ValueError('Frozen validation cache entry changed: '+str(path))
        references[str(path)] = entry['sha256']
        if i % 25 == 0 or i == len(entries):
            print(f'Checked validation FM cache {i}/{len(entries)}', flush=True)
    folders = {'initial': Path(args.initial_evaluation or root/'adapt_control100/validation/epoch_0000').resolve(),
               'B20': Path(args.b_evaluation or root/'adapt_control100/validation/epoch_0020').resolve(),
               'C20': Path(args.c_evaluation or root/'adapt_multiscale/validation/epoch_0020').resolve()}
    for folder in folders.values():
        if output == folder or folder in output.parents or output in folder.parents:
            raise ValueError('Output must be outside source evaluation directories')
    expected = {m: source_header(folders[m], 'adapt_multiscale' if m=='C20' else 'adapt_control100',
                               0 if m=='initial' else 20, study, references) for m in METHODS}
    output.mkdir(parents=True, exist_ok=True)
    scores, pair_rows, exclusions, locators, seen_cache = [], [], [], {}, set()
    with ExitStack() as stack:
        sources = {}
        for task in ('task1','task2'):
            sources[task] = {}
            for m in METHODS:
                sources[task][m] = Saved(folders[m]/task, stack, references,
                                         sources[task]['initial'].rows if m != 'initial' else None)
            canonical = view['validation_rows'][task]
            # Align cached pair metadata by identity, never by array row position.
            if {key(r) for r in canonical} != {key(r) for r in sources[task]['initial'].rows}:
                raise ValueError('Study cache and saved validation populations differ')
            canonical_lookup = {key(r):r for r in canonical}
            for row in sources[task]['initial'].rows:
                if any(row[k] != canonical_lookup[key(row)][k] for k in ('subject_id','input_type')):
                    raise ValueError('Study and evaluation patient/device metadata differ')
        print('Recomputing raw120s official reliable-I scores; CPU only', flush=True)
        for m in METHODS:
            summaries = {}
            for task in ('task1','task2'):
                saved = sources[task][m]
                overall, detail = evaluate_record_predictions(saved.arrays['prediction'], saved.arrays['target'], task,
                    saved.rows, target_quality_mask=saved.arrays['mask'],
                    input_quality_mask=saved.arrays['mask'][:,0], require_reliable_input=True,
                    input_quality_source='same_record_target_I')
                write_report(output/'verification'/m/task, overall, detail, title='Saved-array reliable-I verification')
                summaries[task] = overall
            score = correlation_score_from_summaries(summaries)
            for field in ('task1_r_missing11','task2_r_missing11','main_correlation'):
                if abs(score[field]-expected[m][field]) > 2e-5:
                    raise ValueError('Saved raw arrays do not reproduce '+m+' reference score')
            write_correlation_score(output/'verification'/m, score)
            scores.append({'method':m, **score})
            print(m, score, flush=True)
        target_checks = {}
        for task, variants in sources.items():
            for n, ids in enumerate(pairs(variants['initial'].rows),1):
                row = variants['initial'].rows[ids[0]]
                cache_key = task+':'+row['pair_id']
                entry = entries[cache_key]; seen_cache.add(cache_key)
                if entry['target_record_id'] != row['target_record_id'] or entry['subject_id'] != row['subject_id']:
                    raise ValueError('Cache pair mixes patient/target identity')
                waves = {}
                y0, q0 = None, None
                for m, saved in variants.items():
                    p, y, q = saved.record(ids)
                    if y0 is not None and (not np.array_equal(y,y0) or not np.array_equal(q,q0)):
                        raise ValueError('Methods use different targets or quality masks')
                    waves[m], y0, q0 = p, y, q
                with np.load(cache_root/entry['file'], allow_pickle=False) as cache:
                    fm = cache['fm_uV'].astype(np.float64)
                    if not np.array_equal(y0,cache['target_uV']) or not np.array_equal(q0,cache['mask']):
                        raise ValueError('Cache and evaluation targets/QC differ')
                if fm.shape != (12,60000) or not np.isfinite(fm).all():
                    raise ValueError('Need finite fixed control100 full120s prediction')
                if any(not np.allclose(waves[m][0],fm[0],atol=.002,rtol=1e-6) for m in METHODS):
                    raise ValueError('Actual observed I differs between saved methods and fixed FM')
                import hashlib
                fingerprint = hashlib.sha256(y0.tobytes()+q0.tobytes()).hexdigest()
                identity = (row['subject_id'], fingerprint)
                if target_checks.setdefault(row['target_record_id'], identity) != identity:
                    raise ValueError('Repeated physical target has changed patient/labels/QC')
                if not q0[0]:
                    exclusions.append({**{k:row[k] for k in ('pair_id','target_record_id','subject_id','input_type')},
                                       'task':task,'lead':'ALL','reason':'actual_input_I_marked_unreliable'})
                    continue
                for j, lead in enumerate(LEADS[1:],1):
                    if not q0[j]:
                        exclusions.append({**{k:row[k] for k in ('pair_id','target_record_id','subject_id','input_type')},
                                           'task':task,'lead':lead,'reason':'target_lead_marked_bad'})
                        continue
                    for m in METHODS:
                        pair_rows.append({'method':m,'task':task,'input_type':row['input_type'],
                            'pair_id':row['pair_id'],'target_record_id':row['target_record_id'],
                            'subject_id':row['subject_id'],'lead':lead, **metrics(waves[m][j],y0[j],fm[j],study['beta'])})
                    locators.setdefault((task,row['input_type'],row['target_record_id'],lead), (ids,cache_key,row['pair_id']))
                print(f'Diagnosed {task} pair {n}: {row["target_record_id"]}', flush=True)
        if seen_cache != set(entries):
            raise ValueError('Unused/orphan cached validation pairs')
        targets, leads = aggregate(pair_rows)
        deltas = comparisons(targets)
        target_summary, patients = target_diagnostics(deltas)
        interval = bootstrap(targets, args.bootstrap, args.seed)
        for row in scores:
            if abs(interval['point_R'][row['method']]-row['main_correlation']) > 1e-10:
                raise ValueError('Target/patient hierarchy does not reproduce official R')
        selected = choose_plots(deltas, args.max_plots) if args.plots else []
        plot_rows = []
        if selected:
            (output/'plots').mkdir()
        for i, row in enumerate(selected):
            ids, cache_key, pair_id = locators[(row['task'],row['input_type'],row['target_record_id'],row['lead'])]
            variants = sources[row['task']]
            waves = {m: v.record(ids)[0] for m,v in variants.items()}
            y = variants['initial'].record(ids)[1]
            with np.load(cache_root/entries[cache_key]['file'], allow_pickle=False) as cache:
                fm = cache['fm_uV'].astype(np.float64)
            name = f'{i+1:02d}_{row["subject_id"]}_{row["lead"]}.png'
            start = plot_case(output/'plots'/name,row,waves,y,fm)
            plot_rows.append({**row,'pair_id':pair_id,'window_start_seconds':start,'plot':'plots/'+name})
    for name, rows in (('score_summary',scores),('pair_lead_metrics',pair_rows),('target_lead_metrics',targets),
                       ('lead_summary',leads),('target_lead_comparison',deltas),('target_diagnostics',target_summary),
                       ('patient_diagnostics',patients),('qc_exclusions',exclusions),('plot_selection',plot_rows)):
        csv_save(output/(name+'.csv'), rows)
    write(output/'patient_bootstrap.json', interval)
    patient_counts = {name:{'improved':sum(r[name] is not None and r[name]>0 for r in patients),
                           'worsened':sum(r[name] is not None and r[name]<0 for r in patients),
                           'ties':sum(r[name] == 0 for r in patients)} for name in ('C_minus_B','C_minus_initial')}
    print('Rechecking source hashes; no source files were written', flush=True)
    if any(sha(path) != checksum for path,checksum in references.items()):
        raise ValueError('A source changed during diagnosis; discard incomplete output')
    write(output/'diagnostic.json', {'completed':True,'protocol':PROTOCOL,'score_protocol':SCORE,
          'run':str(root),'evaluations':{m:str(p) for m,p in folders.items()},'scores':scores,
          'source_files':{p:{'sha256':v} for p,v in references.items()},'script_sha256':sha(__file__),
          'training_started':False,'model_loaded':False,'new_FM_inference':False,'source_arrays_modified':False,
          'beta':study['beta'],'sampling':study['sampling'],'patient_diagnostic_counts':patient_counts,
          'n_reliable_patients':len(patients),'plot_count':len(plot_rows),
          'slow_definition':'saved_final - (fixed_FM - full120s MA501(fixed_FM)); actual beta=.75 contribution',
          'bands_definition':'50Hz block means then MA251/MA751; approximate training scales; not ideal frequency cutoffs',
          'band_QC':'input-I and target-lead QC; constant-label r unavailable; no training variance-floor exclusion',
          'diagnostic_patient_means_are_NOT_official_R':True,'truth_selected_windows_are_NOT_deployable':True})
    report = ['# 第0轮、B20、C20离线诊断','',
              '已按可靠I＋目标导联QC复核完整120秒正式评分。不训练、不加载模型、不重新推理。',
              'score_summary.csv是正式分数；lead_summary.csv是诊断均值/中位数，不能代替正式R。',
              'target_lead_comparison.csv是逐目标/导联配对变化；patient_diagnostics.csv的简单患者均分仅用于定位。',
              'patient_bootstrap.json按患者成组抽样并保持正式宏平均层级；只针对当前已选定权重与验证患者。',
              'bootstrap不等于训练seed复核，重复配对/任务/设备中的同一患者始终一起抽样。',
              '慢贡献从同一固定FM快残差中恢复，不把平滑最终预测冒充独立慢CNN。',
              '分尺度指标描述实际beta=.75贡献在50Hz上的变化，近似训练尺度，不是理想频率分带。',
              '图是正常I/正常目标导联的说明性选例，10秒窗口由真值诊断选取，不代表总体得分。',
              '标签、均值及尺度拆分只用于离线分析，不修改部署预测。',
              '', '## 患者诊断变化数量', str(patient_counts), '', '## 患者成组bootstrap', str(interval)]
    (output/'report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    print({'completed':True,'output':str(output),'patients':len(patients),'plots':len(plot_rows)},flush=True)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default=RUN)
    p.add_argument('--initial-evaluation')
    p.add_argument('--b-evaluation')
    p.add_argument('--c-evaluation')
    p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42,help='Bootstrap RNG only; not model/training/inference seed')
    p.add_argument('--plots',action='store_true')
    p.add_argument('--max-plots',type=int,default=6)
    run(p.parse_args())


if __name__ == '__main__':
    main()
