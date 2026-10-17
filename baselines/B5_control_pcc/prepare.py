"""Verify and reuse completed caches, create a NEW auditable study; no FM inference."""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
import torch
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_best.bundle import load_bundle, verified_source, DEFAULT_PROFILE
from baselines.B5_best import SAMPLING
from baselines.B5.scoring import SCORE_PROTOCOL
from . import PROTOCOL, LEGACY_CACHE_PROTOCOL, GROUPS
from .common import code_digest, slow_model, validation_key, load_cache
from .core import generate

ROOT = Path('/home/qht/huawei/results/B5-best-exp-SlowPCC')
NON_FM_FIELDS = {'anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded', 'target_uV', 'mask', 'slow_target'}


def source_cache(run):
    study_path = run / 'shared/study.json'
    study, manifest = read(study_path), read(run / 'cache/cache.json')
    if (manifest.get('protocol') != LEGACY_CACHE_PROTOCOL or not manifest.get('completed')
            or manifest.get('study_sha256') != sha(study_path)
            or manifest.get('sampling') != SAMPLING
            or manifest.get('FM_optimizer_updates') != 0):
        raise ValueError('Need a complete unchanged Heun16/K16 cache: ' + str(run))
    components = run / 'shared/components.pt'
    if sha(components) != study['bundle_sha256']:
        raise ValueError('Legacy study/component checksum differs')
    bundle = load_bundle(components)
    for split in ('train', 'validation'):
        for index, item in enumerate(manifest[split], 1):
            path = (run / 'cache' / item['file']).resolve()
            if not path.is_relative_to((run / 'cache').resolve()) or sha(path) != item['sha256']:
                raise ValueError('Source cache missing, changed, or outside its root: ' + str(path))
            if index % 25 == 0 or index == len(manifest[split]):
                print(f'Verified {run.name}: {split} {index}/{len(manifest[split])}', flush=True)
    return study, manifest, bundle


def check_train_pair(old_root, old, new_root, new):
    if (old['target_record_id'] != new['target_record_id'] or old['subject_id'] != new['subject_id']
            or old['data_sha256'] != new['data_sha256']):
        raise ValueError('Training condition/target/QC fingerprint differs')
    with np.load(old_root / old['file'], allow_pickle=False) as a, np.load(new_root / new['file'], allow_pickle=False) as b:
        if set(a.files) != NON_FM_FIELDS | {'fm_uV'} or set(a.files) != set(b.files):
            raise ValueError('Unexpected cache fields')
        for key in a.files:
            if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                raise ValueError('Nonfinite training cache')
            if key != 'fm_uV' and not np.array_equal(a[key], b[key]):
                raise ValueError('Two arms differ outside FM: ' + key)
        if a['fm_uV'].shape != (12, 60000) or b['fm_uV'].shape != (12, 60000):
            raise ValueError('FM cache must contain full 120 seconds')
        if not a['mask'][0] or not a['mask'][1:].any():
            raise ValueError('Training requires reliable I and a usable target lead')
        return float(np.max(np.abs(a['fm_uV'] - b['fm_uV'])))


def validation_audit(manifest, root, train_subjects):
    """Audit all cached pairs before runtime scoring applies the reliable-I rule."""
    from baselines.B5_best.validate import pair_indices
    keys = [e['cache_key'] for e in manifest['validation']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate validation cache key')
    lookup = {e['cache_key']: e for e in manifest['validation']}
    if train_subjects & {e['subject_id'] for e in manifest['validation']}:
        raise ValueError('Patient leakage across train and validation')
    audit, target_digests = {}, {}
    seen = set()
    for task, rows in manifest['validation_rows'].items():
        devices = {}
        for ids in pair_indices(rows):
            first = rows[ids[0]]
            key = validation_key(task, first['pair_id'])
            entry = lookup[key]
            if entry['target_record_id'] != first['target_record_id']:
                raise ValueError('Validation pair/target mismatch')
            record = entry['target_record_id']
            previous = target_digests.setdefault(record, entry['data_sha256'])
            if previous != entry['data_sha256']:
                raise ValueError('Repeated target has different I/demographics/labels/QC')
            seen.add(key)
            with np.load(root / entry['file'], allow_pickle=False) as data:
                if (set(data.files) != NON_FM_FIELDS | {'fm_uV'} or
                        any(not np.isfinite(data[k]).all() for k in data.files)):
                    raise ValueError('Invalid validation payload')
                if data['target_uV'].shape != (12, 60000) or data['fm_uV'].shape != (12, 60000):
                    raise ValueError('Validation needs full 120 seconds')
                eligible = bool(data['mask'][0])
            item = devices.setdefault(first['input_type'], {'cached_pairs': 0, 'eligible_pairs': 0,
                                                            'excluded_targets': [], 'targets': set()})
            item['cached_pairs'] += 1
            if eligible:
                item['eligible_pairs'] += 1
                item['targets'].add(record)
            else:
                item['excluded_targets'].append(record)
        for item in devices.values():
            item['eligible_unique_targets'] = len(item.pop('targets'))
            item['excluded_targets'] = sorted(set(item['excluded_targets']))
        audit[task] = devices
    if seen != set(keys) or set(audit) != {'task1', 'task2'}:
        raise ValueError('Orphan cache pairs or incomplete task validation')
    return audit


def prepare(args):
    output = Path(args.output).resolve()
    old_run, new_run = Path(args.old_run).resolve(), Path(args.control100_run).resolve()
    for source in (old_run, new_run):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('New study output must be outside both frozen source runs')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty NEW study output; old source caches stay read-only')
    if args.epochs < 1:
        raise ValueError('Need a positive paired training budget')
    reference = read(args.reference_report)
    if reference.get('score_protocol') != SCORE_PROTOCOL or any(
            not np.isfinite(reference.get(k, np.nan)) or not -1 <= reference[k] <= 1
            for k in ('task1_r_missing11', 'task2_r_missing11', 'main_correlation')):
        raise ValueError('Need a reliable-I reference for this control100 + PCC20 initializer')
    if abs(reference['main_correlation'] - (reference['task1_r_missing11'] + reference['task2_r_missing11']) / 2) > 1e-10:
        raise ValueError('Reference mean r does not match the two tasks')
    print('Checking both completed caches read-only; no FM inference or training', flush=True)
    old, om, ob = source_cache(old_run)
    new, nm, nb = source_cache(new_run)
    _, pinned = verified_source(read(DEFAULT_PROFILE))
    if nb['fm_state_sha256'] != pinned['fm_state_sha256'] or nb['slow_state_sha256'] != pinned['slow_state_sha256']:
        raise ValueError('New cache initializer is not the pinned control100 + PCC20 model')
    if (ob['fm_state_sha256'] == nb['fm_state_sha256'] or ob['fm']['scales_sha256'] != nb['fm']['scales_sha256']
            or old['manifests'] != new['manifests'] or old['source_slow_config'] != new['source_slow_config']):
        raise ValueError('Need different frozen FM partners with identical scales/data/loss preparation')
    if (old['combo_weight'] != new['combo_weight'] or new['combo_weight'] <= 0
            or old['residual_weight'] != new['residual_weight'] or new['mean_weight'] != 0
            or old['beta'] != .75 or new['beta'] != .75):
        raise ValueError('Original loss or beta settings differ')
    a = {e['target_record_id']: e for e in om['train']}
    b = {e['target_record_id']: e for e in nm['train']}
    if len(a) != len(om['train']) or len(b) != len(nm['train']) or set(a) != set(b):
        raise ValueError('Training record sets are not identical and unique')
    if len(a) != old['train_records'] or len(b) != new['train_records']:
        raise ValueError('Completed cache is missing training records declared by its source study')
    differences = [check_train_pair(old_run / 'cache', a[k], new_run / 'cache', b[k]) for k in sorted(a)]
    if not any(v > 0 for v in differences):
        raise ValueError('Two TRAIN FM caches are identical; no partner ablation is possible')
    audit = validation_audit(nm, new_run / 'cache', {e['subject_id'] for e in nm['train']})
    training = copy.deepcopy(new['training'])
    training.update(epochs=args.epochs)
    if training['seed'] != 42 or training['validate_every'] != 5:
        raise ValueError('Preserve paired seed42 and validation every five epochs')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(new_run / 'shared/components.pt', output / 'components.pt')
    source_files = {}
    for run in (old_run, new_run):
        for name in ('shared/study.json', 'shared/components.pt', 'cache/cache.json'):
            path = run / name
            source_files[str(path)] = sha(path)
    ref_path = Path(args.reference_report).resolve()
    source_files[str(ref_path)] = sha(ref_path)
    study = {'protocol': PROTOCOL, 'score_protocol': SCORE_PROTOCOL, 'sampling': dict(SAMPLING),
             'beta': .75, 'training': training, 'residual_weight': new['residual_weight'],
             'residual_width_native': new['residual_width_native'], 'combo_weight': new['combo_weight'],
             'mean_weight': 0., 'bundle_sha256': sha(output / 'components.pt'),
             'source_files': source_files, 'code_sha256': code_digest(), 'reference_score': reference,
             'paired_difference': 'TRAIN FM prediction used by unchanged final-waveform PCC only',
             'validation_FM': 'control100 for BOTH arms', 'source_code_executed': False}
    write(output / 'study.json', study)
    view = {'completed': True, 'study_sha256': sha(output / 'study.json'),
            'train_sources': {GROUPS[0]: {'root': str(old_run / 'cache'), 'entries': [a[k] for k in sorted(a)]},
                              GROUPS[1]: {'root': str(new_run / 'cache'), 'entries': [b[k] for k in sorted(b)]}},
            'validation_root': str(new_run / 'cache'), 'validation': nm['validation'],
            'validation_rows': nm['validation_rows']}
    write(output / 'cache_view.json', view)
    print('Cache views prepared. Checking slow-CNN CPU inference; no optimizer updates', flush=True)
    model = slow_model(nb, torch.device('cpu')).eval()
    with np.load(new_run / 'cache' / nm['train'][0]['file'], allow_pickle=False) as first:
        inputs = {k: torch.from_numpy(first[k].copy()[None]) for k in ('anchor','numeric','sex','field_mask','age_topcoded')}
    with torch.no_grad():
        prediction = generate(model, inputs)
    if prediction.shape != (1, 11, 6000) or not torch.isfinite(prediction).all():
        raise ValueError('Synthetic/read-only full-window inference preflight failed')
    status = {'ready': True, 'training_started': False, 'optimizer_updates': 0, 'FM_optimizer_updates': 0,
              'cache_regenerated': False, 'source_caches_modified': False, 'train_records': len(a),
              'same_initial_slow_state_sha256': state_sha(model.state_dict()),
              'validation_audit': audit, 'score_protocol': SCORE_PROTOCOL,
              'reference_main_correlation': reference['main_correlation'], 'groups': list(GROUPS)}
    write(output / 'preflight.json', status)
    print(status, flush=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run', default=str(ROOT / 'combo_pcc_v1'))
    p.add_argument('--control100-run', default=str(ROOT / 'control100_pcc20_cache_v1'))
    p.add_argument('--reference-report', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    prepare(args)


if __name__ == '__main__':
    main()
