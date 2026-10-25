"""Six independent Huawei fine-tunes from Q40 / P->Q75; no Huawei parent weights."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import signal

from .adapt120 import execute
from .checkpoint import file_sha256, load_checkpoint, model_from_checkpoint
from .compact import PRESETS, ROOT
from .config import ModelConfig, read_json
from .data import load_record, read_manifest
from .evaluate import SCORE_PROTOCOL
from .evaluate_modes import data_identity, write
from .pipeline import Job, acquire_lock, command_for, validate_artifact
from .train import code_sha, validate_settings

VARIANTS = {'H120': (120, 120, 90, 0), 'H10': (10, 10, 10, 0),
            'H40C10': (40, 10, 10, 10)}


def recipes(preset, batch_size=16, eval_batch_size=4, updates=1000, q_updates=1000):
    base = read_json(ROOT/'configs/b7_t_120'/preset/'b7_huawei120.json')
    result = {}
    for key, (window, core, pcc, mae) in {'Q75': (75, 75, 75, 0), **VARIANTS}.items():
        cfg = copy.deepcopy(base)
        cfg['stage'] = 'public75' if key == 'Q75' else 'huawei'+str(window)
        cfg['sampling'].update(window_seconds=window, core_seconds=core)
        cfg['loss'].update(pcc_core_seconds=pcc, mae_core_seconds=mae)
        total = q_updates if key == 'Q75' else updates
        cfg['training'].update(batch_size=batch_size, updates=total, validate_every=total,
                               checkpoint_every=500, boundary_probability=0.)
        cfg['validation'] = {'batch_size': eval_batch_size, 'initial': False}
        validate_settings(cfg)
        result[key] = cfg
    return result


def pin_config(cfg, source):
    value = load_checkpoint(source)
    cfg = copy.deepcopy(cfg)
    if ModelConfig.from_dict(value['model_config']) != ModelConfig.from_dict(cfg['model']):
        raise ValueError('Public parent architecture does not match the selected preset')
    cfg['initialization'] = {'source_stage': value['stage'], 'checkpoint_sha256': file_sha256(source)}
    validate_settings(cfg)
    return cfg


def verify_public(directory, stage, model_config, scales):
    directory = Path(directory)
    done = read_json(directory/'completed.json')
    cfg = read_json(directory/'config.json')
    value = load_checkpoint(directory/'best_r.pt')
    if (done.get('completed') is not True or done.get('update') != cfg['training']['updates']
            or value['best_r'] != done.get('best_r') or value['stage'] != stage
            or cfg['stage'] != stage or value.get('identity', {}).get('config') != cfg
            or ModelConfig.from_dict(value['model_config']) != model_config
            or list(value['scales_uV']) != list(scales) or value.get('score_protocol') != SCORE_PROTOCOL):
        raise ValueError(f'{directory}: expected completed matching {stage} public EMA')
    model_from_checkpoint(value)  # Strict load; old public code hashes may differ.
    return {'path': str((directory/'best_r.pt').resolve()),
            'sha256': file_sha256(directory/'best_r.pt'), 'stage': stage,
            'update': value['update'], 'best_r': value['best_r']}


def ptb75_manifest(path):
    """Reference the existing full continuous NPZ files; never extend or copy waveforms."""
    original = read_manifest(path)
    rows, rejected, counts, subjects = [], [], Counter(), {}
    for row in original['records']:
        if row['source'] != 'ptbdb':
            raise ValueError('Q75 accepts the PTBDB cache only')
        anchor, _, mask, _ = load_record(row)
        if anchor.shape[-1] < 37500 or row.get('real_samples', anchor.shape[-1]) < 37500:
            rejected.append(row['record_id'])
            continue
        row = dict(row, path=row['resolved_path'])
        row.pop('resolved_path', None)
        rows.append(row)
        counts[row['split']] += 1
        subjects.setdefault(row['split'], set()).add(row['subject_id'])
    for split in ('train', 'validation'):
        if not any(r['split'] == split and r['input_reliable'] for r in rows):
            raise ValueError(f'No reliable real75s PTBDB {split} records')
    result = {k: v for k, v in original.items() if k != 'records'}
    result['records'] = rows
    result['window_selection'] = {'real_seconds_minimum': 75, 'padding': False,
        'source_manifest_sha256': file_sha256(path), 'counts': dict(counts),
        'patients': {k: len(v) for k, v in subjects.items()}, 'excluded_short': rejected}
    return result


def persist(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f'Frozen experiment changed: {path}; use a new output')
    write(path, value)


def job_for(name, cfg_path, manifest, source, output, gpu):
    directory = output/name
    return Job(name, (), 'train', ['--config', str(cfg_path), '--manifest', str(manifest),
        '--output', str(directory), '--device', 'cuda', '--init-checkpoint', str(source)],
        directory/'completed.json', gpu)


def run(args):
    for key in ('source_run', 'huawei_manifest', 'ptb_manifest', 'output'):
        setattr(args, key, Path(getattr(args, key)).resolve())
    if args.output == args.source_run or args.source_run in args.output.parents or args.output in args.source_run.parents:
        raise ValueError('New experiments must use a separate output, outside the source run')
    cfgs = recipes(args.preset, args.batch_size, args.eval_batch_size, args.updates, args.q_updates)
    huawei = read_manifest(args.huawei_manifest)
    public75 = ptb75_manifest(args.ptb_manifest)
    if public75['scales_uV'] != huawei['scales_uV']:
        raise ValueError('Huawei/PTB must reuse the same frozen P scales')
    mc = ModelConfig.from_dict(cfgs['Q75']['model'])
    parents = {name: verify_public(args.source_run/name, stage, mc, huawei['scales_uV'])
               for name, stage in (('P', 'public10'), ('Q', 'public40'))}
    # Q40 must genuinely descend directly from this P, not a different public experiment.
    q = load_checkpoint(parents['Q']['path'])
    if q['initialization'].get('sha256') != parents['P']['sha256']:
        raise ValueError('Q40 was not initialized from the selected P EMA')
    audit = public75['window_selection']
    print('[Q75-data] '+json.dumps(audit, ensure_ascii=False), flush=True)
    names = ['Q40_'+v for v in VARIANTS]+['Q75']+['Q75_'+v for v in VARIANTS]
    print('[direct-suite] 6 independent Huawei jobs + Q75; max_train=1 per controller; '
          'full-window FM; central10 MAE/PCC only for H40C10', flush=True)
    if args.dry_run:
        print(json.dumps({'jobs': names, 'parents': parents, 'recipes': cfgs}, ensure_ascii=False, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output/'.direct_suite.lock')
    try:
        (args.output/'configs').mkdir(exist_ok=True)
        (args.output/'logs').mkdir(exist_ok=True)
        manifest75 = args.output/'ptb75_manifest.json'
        persist(manifest75, public75)
        read_manifest(manifest75)
        ptb = read_manifest(args.ptb_manifest)
        spec = {'code_sha256': code_sha(), 'preset': args.preset, 'gpu': args.gpu,
            'parents': parents, 'recipes': cfgs, 'max_train': 1,
            'huawei_data': data_identity(args.huawei_manifest, huawei),
            'ptb_data': data_identity(args.ptb_manifest, ptb), 'ptb75_audit': audit,
            'jobs': names, 'no_Huawei_parent_weights': True}
        persist(args.output/'suite_spec.json', spec)
        states = dict.fromkeys(names, 'pending')
        results = {}

        def save():
            write(args.output/'pipeline_status.json', {'jobs': states, 'results': results,
                'success': all(v == 'done' for v in states.values()), 'max_train': 1})

        save()
        for name in names:
            try:
                if name.startswith('Q75_') and states['Q75'] != 'done':
                    states[name] = 'blocked'
                    save()
                    continue
                if name == 'Q75':
                    source, recipe, manifest = Path(parents['P']['path']), cfgs['Q75'], manifest75
                else:
                    parent, variant = name.split('_', 1)
                    source = Path(parents['Q']['path']) if parent == 'Q40' else args.output/'Q75/best_r.pt'
                    recipe, manifest = cfgs[variant], args.huawei_manifest
                cfg = pin_config(recipe, source)
                expected_stage = 'public10' if name == 'Q75' else ('public40' if name.startswith('Q40_') else 'public75')
                if cfg['initialization']['source_stage'] != expected_stage:
                    raise ValueError('Wrong public source stage')
                cfg_path = args.output/'configs'/(name+'.json')
                persist(cfg_path, cfg)
                job = job_for(name, cfg_path, manifest, source, args.output, args.gpu)
                states[name] = 'training_or_final_validation'
                save()
                if not job.artifact.exists():
                    print(f'[START] {name}; log: {args.output/"logs"/(name+".log")}', flush=True)
                    execute(command_for(job), args.output/'logs'/(name+'.log'), args.gpu)
                result = validate_artifact(job)
                trained = load_checkpoint(job.artifact.parent/'last.pt')
                if trained['initialization'].get('sha256') != file_sha256(source):
                    raise ValueError('Completed run did not descend from its pinned public EMA')
                results[name] = {**result, 'initialization': trained['initialization'],
                                 'sampling': trained['sampling'], 'loss': trained['loss']}
                states[name] = 'done'
                print(f'[DONE] {name}: r={result["best_r"]:.6f}', flush=True)
            except Exception as exc:
                states[name] = 'failed'
                results[name] = {'error': str(exc)}
                print(f'[FAILED] {name}: {exc}; other independent jobs continue', flush=True)
            save()
        return 0 if all(v == 'done' for v in states.values()) else 1
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=tuple(PRESETS), required=True)
    p.add_argument('--source-run', type=Path, required=True, help='Existing compact run containing completed P and Q')
    p.add_argument('--huawei-manifest', type=Path, required=True)
    p.add_argument('--ptb-manifest', type=Path, required=True, help='Existing full continuous PTB cache, not cropped windows')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='1')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=4)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--q-updates', type=int, default=1000)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
