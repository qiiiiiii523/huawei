"""Continue completed A40/B40 EMAs on real full120 records, then compare inference modes."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .checkpoint import file_sha256, load_checkpoint, model_from_checkpoint
from .compact import PRESETS, ROOT
from .config import ModelConfig, read_json
from .data import read_manifest
from .evaluate import SCORE_PROTOCOL
from .evaluate_modes import data_identity, identity_for, validate_reports, write
from .pipeline import Job, acquire_lock, command_for, fingerprint, validate_artifact
from .train import validate_settings

PRESET_NAMES = ('original_t_15m', *PRESETS)


def recipe(args):
    cfg = read_json(ROOT/'configs/b7_t_120'/args.preset/'b7_huawei120.json')
    cfg['training']['batch_size'] = args.batch_size
    cfg['training']['updates'] = args.updates
    cfg['training']['validate_every'] = args.updates
    cfg['validation']['batch_size'] = args.eval_batch_size
    validate_settings(cfg)
    return cfg


def training_job(args, route, config_path):
    directory = args.output / (route+'120')
    return Job(route+'120', (), 'train', ['--config', str(config_path), '--manifest', str(args.manifest),
        '--output', str(directory), '--device', 'cuda', '--init-checkpoint',
        str(args.source_run/route/'best_r.pt')], directory/'completed.json', args.gpu)


def verify_source(args, route, cfg):
    directory = args.source_run/route
    done = read_json(directory/'completed.json')
    old_cfg = read_json(directory/'config.json')
    checkpoint_path = directory/'best_r.pt'
    value = load_checkpoint(checkpoint_path)
    manifest = read_manifest(args.manifest)
    identity = value.get('identity', {})
    current_data = data_identity(args.manifest, manifest)
    if (done.get('completed') is not True or done.get('update') != old_cfg['training']['updates']
            or not math.isfinite(done.get('best_r', float('nan'))) or value['best_r'] != done['best_r']
            or value['stage'] != 'huawei40' or old_cfg['stage'] != 'huawei40'
            or identity.get('config') != old_cfg or value['scales_uV'] != manifest['scales_uV']
            or ModelConfig.from_dict(value['model_config']) != ModelConfig.from_dict(cfg['model'])
            or any(identity.get(k) != v for k, v in current_data.items())
            or value.get('score_protocol') != SCORE_PROTOCOL):
        raise ValueError(f'{route}40 source identity/model/scales/completion changed')
    model_from_checkpoint(value)  # Strict CPU state-dict load; source code hash may differ by design.
    return {'checkpoint': str(checkpoint_path), 'sha256': file_sha256(checkpoint_path),
            'source_code_sha256': identity['code_sha256'], 'best_r_40s': value['best_r'],
            'source_update': value['update'], **current_data}


def wait_for_source(args, route):
    while not (args.source_run/route/'completed.json').is_file():
        candidates = [(args.source_run/'pipeline_status.json', route),
                      (args.source_run.parent.parent/'pipeline_status.json', args.source_run.parent.name+':'+route)]
        for path, key in candidates:
            if path.is_file() and read_json(path).get('jobs', {}).get(key) in ('failed', 'blocked'):
                raise RuntimeError(f'{route}40 failed or blocked; inspect its source log before adapting')
        if not args.wait_for_source:
            raise RuntimeError(f'{route}40 not complete; use --wait-for-source or finish it first')
        print(f'[waiting] {route}40 must complete before {route}120; no GPU training started for this route', flush=True)
        time.sleep(30)


def execute(command, log_path, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    env.setdefault('OMP_NUM_THREADS', '4')
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f'exit={code}; inspect {log_path}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run(args):
    args.output, args.source_run, args.manifest = [Path(x).resolve() for x in
                                                (args.output, args.source_run, args.manifest)]
    if args.output == args.source_run or args.source_run in args.output.parents or args.output in args.source_run.parents:
        raise ValueError('Use a separate output outside the existing 40s run')
    cfg = recipe(args)
    config_path = args.output/'configs/b7_huawei120.json'
    jobs = [training_job(args, route, config_path) for route in args.routes]
    if args.dry_run:
        for job in jobs:
            print(json.dumps({'route': job.name, 'source_40s': str(args.source_run/job.name[0]/'best_r.pt'),
                'training': cfg['training'], 'loss': cfg['loss'], 'sampling': cfg['sampling'],
                'command': command_for(job), 'comparison_modes': ['120/120', '40/10', '40/20']}, ensure_ascii=False))
        return 0
    if not args.source_run.is_dir():
        raise FileNotFoundError(args.source_run)
    read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output/'.adapt120.lock')
    try:
        config_path.parent.mkdir(exist_ok=True)
        if config_path.exists() and read_json(config_path) != cfg:
            raise ValueError('120s config changed; use a new output')
        write(config_path, cfg)
        args.config_dir = config_path.parent
        # No P/Q jobs exist here; model and data identity remain locked on resume.
        args.data_root = args.manifest.parent
        args.metadata = args.source_run
        args.ptb_root = args.manifest.parent
        spec = {'fingerprint': fingerprint(args), 'source_run': str(args.source_run),
                'manifest_sha256': file_sha256(args.manifest), 'routes': args.routes,
                'gpu': args.gpu, 'max_train': 1, 'source_kind': 'completed_same_data_A40_B40_EMA'}
        if (args.output/'adaptation_spec.json').exists() and read_json(args.output/'adaptation_spec.json') != spec:
            raise ValueError('Adaptation code/data/source/plan changed; use a new output')
        write(args.output/'adaptation_spec.json', spec)
        logs = args.output/'logs'
        logs.mkdir(exist_ok=True)
        status = {route+'120': 'pending' for route in args.routes}
        results = {}
        def save():
            write(args.output/'pipeline_status.json', {'jobs': status, 'results': results,
                'success': all(s == 'done' for s in status.values()), 'max_train': 1})
        save()
        # Finish all selected 40s parents before beginning this serial 120s experiment.
        for route in args.routes:
            status[route+'120'] = 'waiting_for_40s'
            save()
            try:
                wait_for_source(args, route)
                status[route+'120'] = 'ready'
            except Exception as exc:
                status[route+'120'] = 'failed'
                results[route+'120'] = {'error': str(exc)}
            save()
        for route, job in zip(args.routes, jobs):
            if status[job.name] == 'failed':
                continue
            try:
                source = verify_source(args, route, cfg)
                source_path = args.output/(route+'40_source.json')
                if source_path.exists() and read_json(source_path) != source:
                    raise ValueError('Parent 40s checkpoint or data changed')
                write(source_path, source)
                status[job.name] = 'training_or_final_validation'
                save()
                if not job.artifact.exists():
                    print(f'[START] {job.name}; log: {logs/(job.name+".log")}', flush=True)
                    execute(command_for(job), logs/(job.name+'.log'), args.gpu)
                result = validate_artifact(job)
                comparison = job.artifact.parent/'comparison'
                status[job.name] = 'comparing_inference'
                save()
                if not (comparison/'completed.json').exists():
                    execute([sys.executable, '-u', '-m', 'baselines.B7.evaluate_modes',
                        '--checkpoint', str(job.artifact.parent/'best_r.pt'), '--manifest', str(args.manifest),
                        '--output', str(comparison), '--batch-size', str(args.eval_batch_size), '--device', 'cuda'],
                        logs/(job.name+'_comparison.log'), args.gpu)
                _, identity = identity_for(job.artifact.parent/'best_r.pt', args.manifest, args.eval_batch_size)
                scores = validate_reports(comparison, identity)
                results[job.name] = {**result, 'inference_r': scores, 'source_40s_r': source['best_r_40s']}
                status[job.name] = 'done'
                print(f'[DONE] {job.name}: {scores}', flush=True)
            except Exception as exc:
                status[job.name] = 'failed'
                results[job.name] = {'error': str(exc)}
                print(f'[FAILED] {job.name}: {exc}; independent route continues', flush=True)
            save()
        return 0 if all(s == 'done' for s in status.values()) else 1
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESET_NAMES, required=True)
    parser.add_argument('--source-run', type=Path, required=True, help='Completed 40s run containing A/B')
    parser.add_argument('--manifest', type=Path,
        default=Path('/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42/data/huawei/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routes', nargs='+', choices=('A', 'B'), default=['A', 'B'])
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=4, help='Concurrent ODE trajectories, K remains16')
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--wait-for-source', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(args.routes)) != len(args.routes):
        parser.error('Routes must not repeat')
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
