"""Standalone slow predictor; never initializes/updates the FM model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import seed_all, device_from_name
from . import PROTOCOL
from .core import (SlowCNN, Records, generate, objective, device_batch, load_prepared,
                   json_write, sha, correlations, macro)


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validate(model, loader, seconds, device):
    model.eval()
    scores, masks, rmses = [], [], []
    for batch in loader:
        b = device_batch(batch, device)
        p = generate(model, b, seconds).cpu().numpy()
        y = batch['target'].numpy()
        scores.append(correlations(p, y))
        masks.append(batch['mask'].numpy())
        rmses.append(np.sqrt(np.mean((p - y) ** 2, axis=-1)))
    return {'slow_r_unique_quality_good_50Hz': macro(np.concatenate(scores), np.concatenate(masks)),
            'slow_rmse_normalized_unique_quality_good_50Hz': macro(np.concatenate(rmses), np.concatenate(masks))}


def run(args):
    prepared, info = load_prepared(args.prepared)
    if args.epochs < 1 or args.batch_records < 1 or args.validate_every < 1:
        raise ValueError('Positive epochs, record batch and validation interval required')
    device = device_from_name(args.device)
    seed_all(42, True)
    model = SlowCNN().to(device)
    initial = state_hash(model.state_dict())
    output = Path(args.output).resolve()
    config = {'protocol': PROTOCOL, 'context_seconds': args.context_seconds, 'seed': 42,
              'epochs': args.epochs, 'batch_records': args.batch_records, 'validate_every': args.validate_every,
              'learning_rate': .0001, 'weight_decay': .01, 'huber_weight': 1., 'pcc_weight': .1,
              'model_channels': 48, 'prepared_sha256': sha(prepared / 'prepared.json'),
              'prepared': str(prepared), 'initial_weights_sha256': initial,
              'selection': 'slow_r_unique_quality_good_50Hz', 'FM_training_started': False}
    generator = torch.Generator().manual_seed(42)
    train = Records(prepared / 'train.npz')
    val = Records(prepared / 'validation.npz')
    train_loader = DataLoader(train, batch_size=args.batch_records, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_records, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    start, best, best_epoch = 0, -float('inf'), None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        old = checkpoint['config']
        if {k: v for k, v in config.items() if k != 'epochs'} != {k: v for k, v in old.items() if k != 'epochs'}:
            raise ValueError('Resume changed context/data/model/optimization settings')
        if args.epochs < old['epochs'] or checkpoint['epoch'] >= args.epochs:
            raise ValueError('Resume epochs must exceed completed epoch and not reduce budget')
        if not (output / 'resolved_config.json').is_file() or (checkpoint['best_epoch'] is not None and not (output / 'best.pt').is_file()):
            raise ValueError('Resume into the original output directory; keep any existing best.pt')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        restore_rng(checkpoint['rng'])
        generator.set_state(checkpoint['loader_rng'])
        start, best, best_epoch = checkpoint['epoch'], checkpoint['best'], checkpoint['best_epoch']
    elif output.exists() and any((output / name).exists() for name in ('best.pt', 'last.pt', 'history.jsonl')):
        raise FileExistsError('Use fresh run output or --resume its last.pt')
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'resolved_config.json', config)
    print(json.dumps({'phase': 'start', 'context_seconds': args.context_seconds, 'initial_weights_sha256': initial,
                      'parameters': sum(p.numel() for p in model.parameters()), 'train_records': len(train),
                      'epochs': args.epochs, 'start_epoch': start, 'device': str(device)}), flush=True)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        totals = {'loss': 0., 'huber': 0., 'pcc': 0.}
        seen = 0
        for batch in train_loader:
            b = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = generate(model, b, args.context_seconds)
            loss, parts = objective(prediction, b['target'], b['mask'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            n = len(b['anchor'])
            for key, value in {'loss': float(loss.detach()), **parts}.items():
                totals[key] += n * value
            seen += n
        record = {'epoch': epoch, 'train': {k: v / seen for k, v in totals.items()}}
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            record['validation'] = validate(model, val_loader, args.context_seconds, device)
            value = record['validation']['slow_r_unique_quality_good_50Hz']
            if not np.isfinite(value):
                raise ValueError('Undefined validation selector')
            improved = value > best
            if improved:
                best, best_epoch = value, epoch
        else:
            improved = False
        record.update(best_slow_r=best if np.isfinite(best) else None, best_epoch=best_epoch)
        payload = {'protocol': PROTOCOL, 'config': config, 'epoch': epoch, 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'rng': pack_rng(), 'loader_rng': generator.get_state(),
                   'best': best, 'best_epoch': best_epoch}
        if improved:
            atomic_save(output / 'best.pt', payload)
        atomic_save(output / 'last.pt', payload)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
    json_write(output / 'completed.json', {'completed': True, 'epochs': args.epochs, 'best_epoch': best_epoch,
                                         'best_slow_r': best, 'FM_training_started': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--context-seconds', type=int, choices=(10, 30), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-records', type=int, default=2)
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--resume')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if not args.execute_training:
        parser.error('Training requires explicit --execute-training; prepare/tests never train')
    run(args)


if __name__ == '__main__':
    main()
