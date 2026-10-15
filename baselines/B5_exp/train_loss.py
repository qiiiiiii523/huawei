"""Same CNN and optimizer; C/M/F at S30, or original-loss S50 context ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import seed_all, device_from_name
from . import PROTOCOL
from .core import Records, load_prepared, sha, json_write, generate, device_batch, correlations, macro
from .check_loss import initializer, implementation_hashes
from .train import state_hash
from .losses import LOSS_PROTOCOL, paired_objective
from .loss_metrics import Reference


def default_group():
    return json.loads((Path(__file__).resolve().parents[2] / 'configs/experiments/b5_exp_loss_group.json').read_text(encoding='utf-8'))['group']


def configuration(args, prepared):
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    if calibration.get('protocol') != LOSS_PROTOCOL or not calibration.get('ready'):
        raise ValueError('Run the common check_loss preflight first')
    if calibration['prepared_sha256'] != sha(prepared / 'prepared.json') or calibration['init_checkpoint_sha256'] != sha(args.init_checkpoint):
        raise ValueError('Shared calibration data/initializer changed')
    if calibration['calibration_split'] != 'train_only_first_two_records':
        raise ValueError('Expected training-only calibration')
    if calibration['implementation_sha256'] != implementation_hashes():
        raise ValueError('Implementation changed after calibration; keep identical paired code')
    seconds = getattr(args, 'context_seconds', 30)
    if seconds not in (30, 50):
        raise ValueError('Continuation supports 30 or 50 second input contexts')
    if seconds == 50 and (args.group != 'C' or not calibration.get('s50_compatibility', {}).get('ready')):
        raise ValueError('S50 requires original loss (group C) and the check_s50 compatibility preflight')
    weights = calibration['weights']
    if not all(np.isfinite(weights[k]) and .01 <= weights[k] <= 10 for k in ('mean', 'residual')):
        raise ValueError('Invalid calibrated weights')
    config = {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'group': args.group,
              'context_seconds': seconds, 'seed': 42, 'epochs': args.epochs,
              'batch_records': args.batch_records, 'validate_every': args.validate_every,
              'learning_rate': .0001, 'weight_decay': .01, 'huber_weight': 1., 'pcc_weight': .1,
              'model_channels': 48, 'prepared': str(prepared), 'prepared_sha256': calibration['prepared_sha256'],
              'init_checkpoint': str(Path(args.init_checkpoint).resolve()),
              'init_checkpoint_sha256': calibration['init_checkpoint_sha256'],
              'initial_weights_sha256': calibration['initial_weights_sha256'], 'source_epoch': calibration['source_epoch'],
              'calibration_sha256': sha(args.calibration), 'calibrated_weights': weights,
              'implementation_sha256': calibration['implementation_sha256'],
              'mean_weight': weights['mean'] if args.group == 'M' else 0.,
              'residual_weight': weights['residual'] if args.group == 'F' else 0.,
              'residual_width_native': calibration['residual_width_native'],
              'selection': ['slow_r_unique_quality_good_50Hz', 'competition_total_score'],
              'score_protocol': calibration['score_protocol'], 'epoch_zero_is_candidate': True,
              'optimizer_initialization': 'fresh_not_source_optimizer', 'FM_training_started': False}
    if calibration.get('s50_compatibility'):
        from .context_audit import canonical, verify_control_files
        proof = calibration['s50_compatibility']
        config['context_extension_protocol'] = calibration['s50_compatibility']['protocol']
        config['control_reference'] = calibration['s50_compatibility']['control_reference']
        if canonical(config) != proof['control_canonical_config']:
            raise ValueError('Context experiment changed initialization, data, loss or optimization from C control')
        verify_control_files(proof['control_reference'])
    return config


@torch.no_grad()
def validation(model, loader, device, reference, seconds=30):
    model.eval()
    predicted, targets, masks = [], [], []
    for batch in loader:
        inputs = {k: batch[k] for k in ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')}
        predicted.append(generate(model, device_batch(inputs, device), seconds).cpu().numpy())
        targets.append(batch['target'].numpy())
        masks.append(batch['mask'].numpy())
    p, y, mask = np.concatenate(predicted), np.concatenate(targets), np.concatenate(masks)
    if not np.isfinite(p).all():
        raise FloatingPointError('Nonfinite generated slow curve')
    learned = reference.restored(p)
    _, diagnostic = reference.diagnostics(learned, mask)
    report = {'slow_r_unique_quality_good_50Hz': macro(correlations(p, y), mask),
              'slow_rmse_normalized_unique_quality_good_50Hz': macro(np.sqrt(np.mean((p - y) ** 2, axis=-1)), mask),
              'hybrid_score': reference.score(learned), 'baseline_score': reference.baseline_score,
              'diagnostics': diagnostic}
    if not np.isfinite(report['slow_r_unique_quality_good_50Hz']):
        raise ValueError('Undefined validation slow r selector')
    return report


def update_bests(best, validation_metrics, epoch):
    selectors = {'slow_r': validation_metrics['slow_r_unique_quality_good_50Hz'],
                 'score': validation_metrics['hybrid_score']['competition_total_score']}
    changed = []
    for name, value in selectors.items():
        if not np.isfinite(value):
            raise ValueError('Undefined validation selector: ' + name)
        if value > best[name]['value']:
            best[name] = {'value': float(value), 'epoch': epoch}
            changed.append(name)
    return changed


def run(args):
    if min(args.epochs, args.batch_records, args.validate_every) < 1:
        raise ValueError('Positive training budget, batch and validation interval required')
    prepared, info = load_prepared(args.prepared)
    config = configuration(args, prepared)
    output = Path(args.output).resolve()
    device = device_from_name(args.device)
    seed_all(42, True)
    model, _ = initializer(args.init_checkpoint, prepared)
    if state_hash(model.state_dict()) != config['initial_weights_sha256']:
        raise ValueError('Initializer weights changed after calibration')
    model.to(device)
    # C/M/F ignore all source optimizer/RNG states, then start the same new stream.
    seed_all(42, True)
    generator = torch.Generator().manual_seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
    train = Records(prepared / 'train.npz')
    val = Records(prepared / 'validation.npz')
    train_loader = DataLoader(train, batch_size=args.batch_records, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_records, shuffle=False, num_workers=0)
    reference = Reference(prepared, info)
    start, best = 0, {key: {'value': -float('inf'), 'epoch': None} for key in ('slow_r', 'score')}
    best_payloads = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('experiment_protocol') != LOSS_PROTOCOL or checkpoint['config'] != config:
            raise ValueError('Exact resume must retain group, initialization, calibration and full budget')
        if checkpoint.get('output_directory') != str(output):
            raise ValueError('Resume into the original run output directory')
        for name in ('resolved_config.json', 'history.jsonl', 'initial_evaluation.json'):
            if not (output / name).is_file():
                raise FileNotFoundError('Keep original resume artifacts: ' + name)
        best_payloads = checkpoint['best_payloads']
        for key, name in (('slow_r', 'best_slow_r.pt'), ('score', 'best_score.pt')):
            candidate = best_payloads[key]
            if candidate['config'] != config or candidate['epoch'] != checkpoint['best'][key]['epoch']:
                raise ValueError('Best candidate does not match resume checkpoint ledger')
            # Recover interrupted multi-file saves from the authoritative last.pt.
            needs_repair = True
            try:
                existing = torch.load(output / name, map_location='cpu', weights_only=True)
                needs_repair = (existing['config'] != candidate['config'] or existing['epoch'] != candidate['epoch'] or
                                state_hash(existing['model']) != state_hash(candidate['model']))
            except Exception:
                # A candidate alias is a recoverable cache; last.pt remains authoritative.
                pass
            if needs_repair:
                atomic_save(output / name, candidate)
        if not 0 <= checkpoint['epoch'] <= args.epochs:
            raise ValueError('Resume epoch outside original budget')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        generator.set_state(checkpoint['loader_rng'])
        restore_rng(checkpoint['rng'])
        start, best = checkpoint['epoch'], checkpoint['best']
        # A crash may leave one extra history line; resume state is authoritative.
        lines = [json.loads(line) for line in (output / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not any(line['epoch'] == start for line in lines):
            raise ValueError('Resume checkpoint epoch missing from history')
        lines = [line for line in lines if line['epoch'] <= start]
        (output / 'history.jsonl').write_text(''.join(json.dumps(line, allow_nan=False) + '\n' for line in lines), encoding='utf-8')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Use empty --output, or resume the original last.pt')
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'resolved_config.json', config)
    if config.get('context_extension_protocol'):
        json_write(output / 'context_preflight.json', json.loads(Path(args.calibration).read_text(encoding='utf-8')))

    def save_epoch(epoch, record, changed):
        candidate = {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'config': config,
                     'epoch': epoch, 'source_epoch': config['source_epoch'],
                     'model': {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()},
                     'validation': record.get('validation')}
        for key in changed:
            best_payloads[key] = candidate
        payload = {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'config': config,
                   'output_directory': str(output), 'epoch': epoch, 'source_epoch': config['source_epoch'],
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': pack_rng(),
                   'loader_rng': generator.get_state(), 'best': best,
                   'validation': record.get('validation'), 'best_payloads': best_payloads}
        # History is committed first; last.pt represents a complete epoch boundary.
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        for key in changed:
            atomic_save(output / ('best_slow_r.pt' if key == 'slow_r' else 'best_score.pt'), best_payloads[key])
        atomic_save(output / 'last.pt', payload)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    print(json.dumps({'phase': 'start', 'group': args.group, 'source_epoch': config['source_epoch'],
                      'context_seconds': config['context_seconds'],
                      'additional_epochs': args.epochs, 'start_epoch': start, 'device': str(device),
                      'initial_weights_sha256': config['initial_weights_sha256'],
                      'optimizer_initialization': 'fresh' if not args.resume else 'exact_resume'}), flush=True)
    if not args.resume:
        report = validation(model, val_loader, device, reference, config['context_seconds'])
        changed = update_bests(best, report, 0)
        record = {'epoch': 0, 'phase': 'common_initializer', 'validation': report, 'best': best}
        json_write(output / 'initial_evaluation.json', report)
        save_epoch(0, record, changed)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        totals, seen = {}, 0
        for batch in train_loader:
            b = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            p = generate(model, b, config['context_seconds'])
            loss, parts = paired_objective(p, b['target'], b['mask'], args.group,
                                          config['calibrated_weights'], config['residual_width_native'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            n = len(b['anchor'])
            for key, value in {'loss': float(loss.detach()), **parts}.items():
                totals[key] = totals.get(key, 0.) + n * value
            seen += n
        if seen == 0:
            raise ValueError('Empty training set')
        record = {'epoch': epoch, 'source_epoch': config['source_epoch'],
                  'train': {k: v / seen for k, v in totals.items()}}
        changed = []
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            record['validation'] = validation(model, val_loader, device, reference, config['context_seconds'])
            changed = update_bests(best, record['validation'], epoch)
        record['best'] = best
        save_epoch(epoch, record, changed)
    json_write(output / 'completed.json', {'completed': True, 'group': args.group,
                                         'source_epoch': config['source_epoch'], 'additional_epochs': args.epochs,
                                         'best': best, 'FM_training_started': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--group', choices=('C', 'M', 'F'), default=default_group())
    parser.add_argument('--context-seconds', type=int, choices=(30, 50), default=30)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-records', type=int, default=2)
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--resume')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if not args.execute_training:
        parser.error('Training requires --execute-training; check_loss never trains')
    run(args)


if __name__ == '__main__':
    main()
