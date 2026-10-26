"""Train the slow branch with scheduled EMA validation, selection and exact resume."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from .config import LossConfig, ModelConfig, read_json
from .experiment import build_optimizer, load_experiment
from .train_data import SlowDataset, file_sha, read_manifest
from .train_eval import FrozenFast, evaluate
from .train_loss import flow_loss

ARCHITECTURE = 'FM-Unet-slow-seven-level-v1'


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def pack_rng():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact resume needs same visible CUDA device count')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def move(batch, device):
    return {k: move(v, device) if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


def source_sha():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def load_statistics(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    scales = np.asarray(value['scale_uV'], np.float64)
    noise = np.asarray(value['noise_std7'], np.float64)
    if scales.shape != (12,) or noise.shape != (7,) or not np.isfinite(scales).all() or not np.isfinite(noise).all():
        raise ValueError('Expected finite frozen scale_uV[12] and noise_std7[7]')
    if (scales <= 0).any() or (noise <= 0).any():
        raise ValueError('Scale/noise must be positive')
    return scales.tolist(), noise.tolist()


def train(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    raw_config = read_json(args.config)
    seed = raw_config['training']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config, model = load_experiment(args.config)
    training = config['training']
    output = Path(args.output).resolve()
    if args.resume and args.init_checkpoint:
        raise ValueError('Choose exact resume OR new-stage EMA initialization')
    if not args.resume and output.exists():
        raise FileExistsError('New experiment needs a new output directory')
    if args.resume and not (output/'last.pt').is_file():
        raise FileNotFoundError('Resume needs this run last.pt')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    scales, noise = load_statistics(args.statistics)
    manifest = read_manifest(args.manifest)
    if args.demographics_csv:
        if not args.fast_code:
            raise ValueError('Demographics encoding requires the existing --fast-code B5 repository')
        import sys
        sys.path.insert(0, str(Path(args.fast_code).resolve()))
        from baselines.B5.metadata import DemographicsTable
        table = DemographicsTable(args.demographics_csv)
        for row in manifest['records']:
            if row['source'] == 'huawei':
                row['metadata'] = table.get(row['subject_id'])
    sources = {r['source'] for r in manifest['records']}
    expected = {'public10': {'ptbxl', 'chapman'}, 'public75': {'ptbdb'}, 'huawei120': {'huawei'}}
    if sources != expected[config['stage']]:
        raise ValueError(f'Stage requires sources {expected[config["stage"]]}, got {sources}')
    if config['stage'] == 'public10' and manifest.get('public_qc_version') != 'B7-public-QC-v2':
        raise ValueError('public10 requires corrected QC-v2 cache')
    if not any(r['split'] == 'validation' and r['input_reliable'] for r in manifest['records']):
        raise ValueError('Reliable validation records required')
    if config['stage'] == 'huawei120' and (not args.fast_code or not args.fast_checkpoint):
        raise ValueError('Huawei final full-wave validation requires --fast-code and --fast-checkpoint')
    print(f'[train] stage={config["stage"]} parameters={sum(p.numel() for p in model.parameters())} '
          f'device={device} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} '
          f'batch={training["batch_size"]} microbatch=none accumulation=1 '
          f'updates={training["updates"]} validation={config["validation"]["mode"]} validation_updates={validation_steps(config)}', flush=True)
    print('[preflight] checking immutable cached data and patient splits; no ODE generation', flush=True)
    dataset = SlowDataset(manifest, config, scales, args.cache_dir)
    # Hash held-out records too, so exact resume cannot silently change the final evaluation set.
    all_hashes = dict(dataset.record_hashes)
    for row in manifest['records']:
        if row['split'] == 'validation':
            all_hashes[row['record_id']] = file_sha(row['resolved_path'])
    identity = dict(config=config, manifest_sha=file_sha(args.manifest), code_sha=source_sha(),
                    records=all_hashes, scale_uV=scales, noise_std7=noise,
                    statistics_sha=file_sha(args.statistics), device_type=device.type,
                    demographics_sha=file_sha(args.demographics_csv) if args.demographics_csv else None,
                    fast_checkpoint_sha=file_sha(args.fast_checkpoint) if args.fast_checkpoint else None,
                    fast_code=str(Path(args.fast_code).resolve()) if args.fast_code else None)
    model = model.to(device)
    optimizer = build_optimizer(model, config)
    initialization = {'kind': 'random', 'seed': seed, 'old_network_weights_loaded': False}
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        previous = {'public75': 'public10', 'huawei120': 'public75'}.get(config['stage'])
        if (initial.get('architecture_id') != ARCHITECTURE or initial.get('stage') != previous
                or not (initial.get('validation_completed') or initial.get('final_validated'))
                or ModelConfig.from_dict(initial['model_config']) != model.config
                or initial['scale_uV'] != scales or initial['noise_std7'] != noise):
            raise ValueError('Stage initialization requires completed previous-stage matching EMA/statistics')
        model.load_state_dict(initial['ema'], strict=True)
        initialization = dict(kind='previous_stage_ema', stage=initial['stage'], step=initial['step'],
                              checkpoint_sha=file_sha(args.init_checkpoint), optimizer_reset=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    step, final_validated = 0, False
    validated_steps, best_r, best_step = [], -float('inf'), None
    if args.resume:
        old = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if old.get('architecture_id') != ARCHITECTURE or old['identity'] != identity:
            raise ValueError('Resume changed code/config/data/statistics/device/fast identity; start a new run')
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        ema = {k: v.to(device) for k, v in old['ema'].items()}
        step, final_validated, initialization = old['step'], old['final_validated'], old['initialization']
        validated_steps = old.get('validated_steps', [step] if final_validated else [])
        best_r, best_step = old.get('best_r', -float('inf')), old.get('best_step')
        restore_rng(old['rng'])
        print(f'[resume] restored update={step} final_validated={final_validated}', flush=True)
    output.mkdir(parents=True, exist_ok=args.resume)
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    def payload():
        return dict(architecture_id=ARCHITECTURE, model_config=model.config.to_dict(), stage=config['stage'],
                    native_hz=model.rate, input_seconds=model.input_seconds, output_seconds=model.output_seconds,
                    scale_uV=scales, noise_std7=noise, model=model.state_dict(), ema=ema,
                    optimizer=optimizer.state_dict(), step=step, seed=seed, rng=pack_rng(), identity=identity,
                    initialization=initialization, final_validated=final_validated, validated_steps=validated_steps,
                    validation_completed=step in validated_steps, best_r=best_r, best_step=best_step)

    def run_validation():
        nonlocal final_validated, best_r, best_step
        if step in validated_steps:
            return
        print(f'[validation] START EMA update={step}; Heun16/K16', flush=True)
        training_rng = pack_rng()
        evaluated = copy.deepcopy(model).eval()
        evaluated.load_state_dict(ema, strict=True)
        fast = FrozenFast(args.fast_code, args.fast_checkpoint, device) if config['stage'] == 'huawei120' else None
        # Restore the pre-validation RNG, including randomness consumed by B5 initialization.
        try:
            metrics, rows = evaluate(evaluated, manifest, config, scales, noise, fast=fast,
                                     fast_cache=Path(args.cache_dir)/'fixed_fast')
        finally:
            restore_rng(training_rng)
        value = float(metrics['r'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation r cannot select a checkpoint')
        (output/f'validation_{step:07d}.json').write_text(
            json.dumps(dict(update=step, metrics=metrics, records=rows), indent=2), encoding='utf-8')
        won = value > best_r
        if won:
            best_r, best_step = value, step
        validated_steps.append(step)
        final_validated = step == training['updates']
        atomic_save(output/f'step_{step:07d}.pt', payload())
        if won:
            atomic_save(output/'best.pt', payload())
        atomic_save(output/'last.pt', payload())
        if final_validated:
            atomic_save(output/'final.pt', payload())
            (output/'validation_final.json').write_text(json.dumps(dict(metrics=metrics, records=rows), indent=2), encoding='utf-8')
            exported = payload()
            inference = {k: exported[k] for k in ('ema', 'model_config', 'native_hz', 'input_seconds',
                                                  'output_seconds', 'scale_uV', 'noise_std7', 'step', 'seed')}
            atomic_save(output/'inference.pt', inference)
        print('[validation] '+json.dumps(dict(update=step, metrics=metrics, new_best=won, best_step=best_step)), flush=True)
        del evaluated, fast
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not args.resume:
        atomic_save(output/'last.pt', payload())
    elif step in validation_steps(config) and step not in validated_steps:
        run_validation()  # Recover a checkpoint saved before an interrupted scheduled validation.
    loss_config = LossConfig(**config['loss'])
    started = time.monotonic()
    model.train()
    for step in range(step+1, training['updates']+1):
        batch = move(default_collate([dataset[i] for i in dataset.indices(training['batch_size'])]), device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = flow_loss(model, batch, loss_config, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for key, value in model.state_dict().items():
                ema[key].mul_(training['ema_decay']).add_(value, alpha=1-training['ema_decay'])
        if step == 1 or step % 50 == 0:
            report = dict(update=step, budget=training['updates'], loss=float(loss.detach()),
                          **{k: float(v.detach()) for k, v in parts.items()},
                          elapsed_seconds=time.monotonic()-started)
            print('[train] '+json.dumps(report), flush=True)
            with (output/'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(report)+'\n')
        due = step in validation_steps(config)
        if step % training['checkpoint_every'] == 0 or due or step == args.stop_after:
            atomic_save(output/'last.pt', payload())
            print(f'[checkpoint] update={step}; validation_due={due}', flush=True)
        if due:
            run_validation()
        if step == args.stop_after and step < training['updates']:
            return
    if not final_validated:
        run_validation()
    selected = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
    atomic_save(output/'best_inference.pt', {k: selected[k] for k in ('ema', 'model_config', 'native_hz',
        'input_seconds', 'output_seconds', 'scale_uV', 'noise_std7', 'step', 'seed')})
    (output/'completed.json').write_text(json.dumps(dict(stage=config['stage'], update=step,
        completed=True, final_validated=True, selection='best_scheduled_ema', best_r=best_r,
        best_step=best_step, validated_steps=validated_steps)), encoding='utf-8')


def validation_steps(config):
    return config['validation'].get('updates', [config['training']['updates']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--statistics', default=str(Path(__file__).with_name('best.pt')))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--fast-code')
    parser.add_argument('--fast-checkpoint')
    parser.add_argument('--demographics-csv', help='Original observed patient demographics; Huawei only')
    parser.add_argument('--stop-after', type=int, help='Testing/recovery drill: save and return after this update')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
