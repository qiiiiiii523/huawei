"""Collect a read-only B7-T diagnostic bundle; never train or use a CUDA device."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def copy(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def finite(value):
    return float(value) if np.isfinite(value) else None


def select_samples(manifest, scores=None):
    """Two training subjects and low/median/high validation pairs per group."""
    scores = scores or {}
    selected = []
    groups = sorted({r['group'] for r in manifest['records']})
    for group in groups:
        rows = [r for r in manifest['records'] if r['group'] == group and r['input_reliable']]
        subjects = set()
        for row in rows:
            if row['split'] == 'train' and row['subject_id'] not in subjects:
                subjects.add(row['subject_id'])
                selected.append(row)
                if len(subjects) == 2:
                    break
        valid = [r for r in rows if r['split'] == 'validation']
        def rank(row):
            value = scores.get(row['record_id'])
            return (float(value) if value is not None and np.isfinite(value) else 0., row['record_id'])
        valid.sort(key=rank)
        if valid:
            selected.extend(valid[i] for i in sorted({0, len(valid)//2, len(valid)-1}))
    return selected


def waveform_stats(path, scales):
    with np.load(path, allow_pickle=False) as data:
        anchor = data['anchor_uV'].astype('float64')
        target = data['target_uV'].astype('float64')
        mask = data['lead_mask'].astype(bool)
        stats = {'keys': list(data.files), 'anchor_shape': list(anchor.shape),
                 'target_shape': list(target.shape), 'lead_mask': mask.tolist(),
                 'seconds_at_500Hz': target.shape[-1]/500,
                 'anchor_I_max_abs_difference_uV': finite(np.nanmax(np.abs(anchor-target[:1]))),
                 'std_uV': [finite(np.nanstd(row)) for row in target],
                 'normalized_std': [finite(np.nanstd(row)/scale) for row, scale in zip(target, scales)],
                 'abs_p99_uV': [finite(np.nanpercentile(np.abs(row), 99)) for row in target]}
        residuals = {'III': target[2]-(target[1]-target[0]),
                     'aVR': target[3]+(target[0]+target[1])/2,
                     'aVL': target[4]-(target[0]-target[1]/2),
                     'aVF': target[5]-(target[1]-target[0]/2)}
        stats['derived_limb_residual_rms_uV'] = {
            name: finite(np.sqrt(np.nanmean(value**2))) for name, value in residuals.items()}
        return stats


def collect_samples(manifest_path, destination, scores=None):
    manifest = read(manifest_path)
    full = destination / 'manifest_original.json'
    copy(manifest_path, full)
    rows = select_samples(manifest, scores)
    exported, statistics = [], []
    for index, row in enumerate(rows):
        source = Path(row['path'])
        if not source.is_absolute():
            source = manifest_path.parent / source
        name = f'records/{index:04d}.npz'
        if not copy(source, destination / name):
            raise FileNotFoundError(source)
        new = dict(row, path=name)
        new.pop('resolved_path', None)
        exported.append(new)
        statistics.append({'record_id': row['record_id'], 'split': row['split'], 'group': row['group'],
                           'source_path': str(source), 'stats': waveform_stats(source, manifest['scales_uV'])})
    subset = dict(manifest, records=exported, diagnostic_subset=True,
                  diagnostic_original_manifest=str(manifest_path.resolve()))
    write(destination / 'manifest_subset.json', subset)
    write(destination / 'sample_stats.json', statistics)
    return {'source': str(manifest_path), 'selected': len(exported),
            'record_counts': dict(Counter(f"{r['split']}/{r['group']}" for r in manifest['records'])),
            'subject_counts': {split: len({r['subject_id'] for r in manifest['records'] if r['split'] == split})
                               for split in ('train', 'validation', 'test')},
            'scales_uV': manifest['scales_uV']}


def collect_run(source, destination):
    if not source.is_dir():
        return
    for pattern in ('*.json', '*.log', 'configs/*.json', 'logs/*.log'):
        for path in source.glob(pattern):
            copy(path, destination / path.relative_to(source))
    for stage in ('P', 'A', 'Q', 'B'):
        directory = source / stage
        for pattern in ('*.json', '*.jsonl'):
            for path in directory.glob(pattern):
                copy(path, destination / stage / path.name)


def export_ema(source, destination):
    import torch
    value = torch.load(source, map_location='cpu', weights_only=True)
    required = ('architecture_id', 'format_version', 'generated_leads', 'model_config', 'ema',
                'sampling', 'loss', 'scales_uV', 'stage', 'update', 'best_r')
    if any(k not in value for k in required):
        raise ValueError(f'Incomplete diagnostic checkpoint: {source}')
    thin = {k: v for k, v in value.items() if k not in ('model', 'optimizer', 'rng', 'identity')}
    thin.update(diagnostic_only=True, resume_supported=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(thin, destination)
    identity = value.get('identity', {})
    write(destination.with_suffix('.json'), {'source': str(source), 'update': value['update'],
          'best_r': value['best_r'], 'initialization': value.get('initialization'),
          'identity': {k: v for k, v in identity.items() if k != 'record_sha256'},
          'hashed_records_in_original_checkpoint': len(identity.get('record_sha256', {})),
          'diagnostic_only': True, 'resume_supported': False})


def collect(args):
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    # All originals are read-only; only this newly created destination is written.
    args.output.mkdir(parents=True)
    report = {'CPU_only': True, 'training_started': False, 'raw_XML_included': False,
              'source_run': str(args.run), 'missing': [], 'data': {}, 'weights': []}
    collect_run(args.run, args.output / 'runs/original_T')
    for pattern in ('*.json', '*.log', 'logs/*.log'):
        for path in args.new_run.glob(pattern):
            copy(path, args.output / 'runs/compact_suite' / path.relative_to(args.new_run))
    for name in ('g6_19m', 'w144_g5_21m'):
        collect_run(args.new_run / name / 'seed42', args.output / 'runs' / name)
    for root, name in ((args.repo, 'original_T'), (args.new_repo, 'compact_T')):
        if not root.is_dir():
            continue
        for folder, pattern in (('baselines/B7', '*.py'), ('configs', 'b7_*.json'), ('scripts', 'b7t*.py')):
            for path in (root / folder).glob(pattern):
                copy(path, args.output / 'code' / name / path.relative_to(root))
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        write(args.output / 'code' / name / 'revision.json', {'commit': result.stdout.strip(), 'error': result.stderr.strip()})
        diff = subprocess.run(['git', '-C', str(root), 'diff', 'HEAD', '--', 'baselines/B7', 'configs'],
                              capture_output=True, text=True)
        (args.output / 'code' / name / 'working_tree.diff').write_text(diff.stdout, encoding='utf-8')
    for name in ('subject_split.csv', 'raw_record_manifest.csv', 'device_interpretation_qc.csv',
                 'pair_manifest_task1.csv', 'pair_manifest_task2.csv'):
        if not copy(args.metadata / name, args.output / 'metadata' / name):
            report['missing'].append(str(args.metadata / name))
    for name in ('huawei_index.json', 'huawei_index_audit.json'):
        copy(args.cache_run / 'data' / name, args.output / 'data' / name)
    for stage in ('P', 'Q', 'A', 'B'):
        source = args.run / stage / 'best_r.pt'
        if not source.is_file():
            raise FileNotFoundError(f'Expected completed original stage: {source}')
        export_ema(source, args.output / 'weights' / f'{stage}_ema_diagnostic.pt')
        report['weights'].append(stage)
        print(f'[weights] {stage}: EMA exported for diagnosis, original unchanged', flush=True)
    for dataset, stage in (('public10', 'P'), ('ptb40', 'Q'), ('huawei', 'B')):
        manifest = args.cache_run / 'data' / dataset / 'manifest.json'
        if dataset == 'ptb40' and (args.cache_run / 'ptb_recovery/data/manifest.json').is_file():
            manifest = args.cache_run / 'ptb_recovery/data/manifest.json'
        # Use the actual manifest identified by this original run when recorded.
        provenance = args.run / 'p_final_only.json'
        if provenance.is_file():
            cache_key = {'public10': 'public10', 'ptb40': 'public40', 'huawei': 'huawei40'}[dataset]
            item = read(provenance).get('caches', {}).get(cache_key)
            if item:
                manifest = Path(item['manifest'])
        validation = args.run / stage / ('validation_0010000.json' if stage == 'P' else 'validation_0001000.json')
        scores = {r['record_id']: r.get('r') for r in read(validation).get('records', [])} if validation.is_file() else {}
        report['data'][dataset] = collect_samples(manifest, args.output / 'data' / dataset, scores)
        print(f"[data] {dataset}: selected {report['data'][dataset]['selected']} real records", flush=True)
    write(args.output / 'bundle_report.json', report)
    files = sorted(p for p in args.output.rglob('*') if p.is_file())
    with (args.output / 'SHA256SUMS').open('w', encoding='utf-8') as out:
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out.write(f'{digest}  {path.relative_to(args.output).as_posix()}\n')
    print(f'Diagnostic directory ready: {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('/home/qht/huawei_B7_T'))
    parser.add_argument('--new-repo', type=Path, default=Path('/home/qht/huawei_B7_T_compact'))
    parser.add_argument('--run', type=Path, default=Path('/home/qht/huawei_B7_T/results/B7-T/auto_bs32_final_only_seed42'))
    parser.add_argument('--new-run', type=Path, default=Path('/home/qht/huawei_B7_T_compact/results/B7-T-compact-suite'))
    parser.add_argument('--cache-run', type=Path, default=Path('/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42'))
    parser.add_argument('--metadata', type=Path, default=Path('/home/qht/huawei/metadata'))
    args = parser.parse_args()
    # Prevent torch imports/loads from discovering or initializing a CUDA device.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    collect(args)


if __name__ == '__main__':
    main()
