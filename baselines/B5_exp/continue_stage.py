"""Extend a completed S50/S50-F slow-CNN run in a NEW directory, never train FM."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import seed_all, device_from_name
from . import PROTOCOL
from .check_loss import implementation_hashes
from .core import SlowCNN, Records, load_prepared, sha, json_write, device_batch, generate, csv_write
from .losses import LOSS_PROTOCOL, paired_objective
from .stage_metrics import QCReference, SCORE_PROTOCOL, SAMPLING, validate
from .train import state_hash

STAGE_PROTOCOL = 'B5-slow-stage-mean-r-v1'
SOURCE_CANDIDATES = ('best_slow_r', 'best_score', 'last')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def implementation():
    base = Path(__file__).resolve().parent
    result = implementation_hashes()
    for name in ('continue_stage.py', 'stage_metrics.py', 'stage_evaluator.py'):
        result['baselines/B5_exp/' + name] = hashlib.sha256((base/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


def audit_source(source, prepared):
    """Only a completed, untouched legacy 20-epoch run is a stage source."""
    source = Path(source).resolve()
    last = torch.load(source/'last.pt', map_location='cpu', weights_only=True)
    cfg = last['config']
    expected = {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'context_seconds': 50,
                'epochs': 20, 'seed': 42, 'model_channels': 48, 'learning_rate': 1e-4,
                'weight_decay': .01, 'huber_weight': 1., 'pcc_weight': .1, 'mean_weight': 0.,
                'prepared_sha256': sha(prepared/'prepared.json')}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Unsupported source run: ' + key)
    if cfg['group'] not in ('C', 'F') or cfg['residual_width_native'] != 51:
        raise ValueError('Only original S50 or fixed-F S50-F can continue')
    active = 0. if cfg['group'] == 'C' else cfg['calibrated_weights']['residual']
    if cfg['residual_weight'] != active or (cfg['group'] == 'F' and active != 2.5890171538829843):
        raise ValueError('Changed F loss coefficient')
    if last.get('experiment_protocol') != LOSS_PROTOCOL or last['epoch'] != 20:
        raise ValueError('Use the completed legacy epoch-20 last.pt, not a best checkpoint')
    if cfg['implementation_sha256'] != implementation_hashes():
        raise ValueError('Legacy implementation differs from this source checkpoint; use its matching branch')
    if read(source/'resolved_config.json') != cfg:
        raise ValueError('Source resolved configuration differs from last.pt')
    done = read(source/'completed.json')
    if not done.get('completed') or done.get('additional_epochs') != 20 or done.get('group') != cfg['group']:
        raise ValueError('Source has not completed its original 20 epochs')
    history = [json.loads(line) for line in (source/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [r['epoch'] for r in history] != list(range(21)):
        raise ValueError('Source history must contain each original epoch exactly once')
    if last.get('output_directory') != str(source):
        raise ValueError('Source path differs from the saved run; do not move the old run')
    if set(('optimizer', 'rng', 'loader_rng', 'best_payloads')) - last.keys():
        raise ValueError('last.pt lacks complete continuation state')
    if len(last['optimizer']['param_groups']) != 1:
        raise ValueError('Unexpected optimizer groups')
    opt = last['optimizer']['param_groups'][0]
    if opt['lr'] != cfg['learning_rate'] or opt['weight_decay'] != cfg['weight_decay']:
        raise ValueError('Optimizer settings differ from source configuration')
    names = ['last.pt', 'best_slow_r.pt', 'best_score.pt', 'resolved_config.json',
             'history.jsonl', 'completed.json', 'initial_evaluation.json']
    for name in ('context_preflight.json', 'pair_preflight.json'):
        if (source/name).exists():
            names.append(name)
    calibration = source / ('pair_preflight.json' if cfg['group'] == 'F' else 'context_preflight.json')
    if sha(calibration) != cfg['calibration_sha256']:
        raise ValueError('Source preflight snapshot changed')
    if sha(cfg['init_checkpoint']) != cfg['init_checkpoint_sha256']:
        raise ValueError('Original S30 initializer changed')
    for key in ('control_reference', 's50_f_control_reference'):
        if cfg.get(key):
            for name, expected_sha in cfg[key]['files'].items():
                if sha(Path(cfg[key]['path'])/name) != expected_sha:
                    raise ValueError('Original paired control changed: ' + name)
    candidates = {}
    for name in SOURCE_CANDIDATES:
        cp = torch.load(source/(name+'.pt'), map_location='cpu', weights_only=True)
        if cp.get('protocol') != PROTOCOL or cp['config'] != cfg or not 0 <= cp['epoch'] <= 20:
            raise ValueError('Source candidate config/epoch mismatch: ' + name)
        if name != 'last':
            key = 'slow_r' if name == 'best_slow_r' else 'score'
            authoritative = last['best_payloads'][key]
            if cp['epoch'] != authoritative['epoch'] or state_hash(cp['model']) != state_hash(authoritative['model']):
                raise ValueError('Source best alias differs from last.pt ledger: ' + name)
        candidates[name] = cp
    files = {str((source/name).resolve()): sha(source/name) for name in names}
    return last, candidates, files


def stage_config(old, total_epochs):
    if total_epochs <= old['epochs']:
        raise ValueError('--total-epochs must exceed the completed source budget (20)')
    cfg = copy.deepcopy(old)
    cfg.update(epochs=total_epochs, selection=['main_correlation'], score_protocol=SCORE_PROTOCOL,
               optimizer_initialization='restore_source_last', continuation_protocol=STAGE_PROTOCOL)
    return cfg


def restore_training(checkpoint, model, optimizer, generator, device):
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Match the original train_loss continuation behavior for optimizer tensors.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    generator.set_state(checkpoint['loader_rng'].cpu())
    restore_rng(checkpoint['rng'])


def eligible_output(output, source):
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Stage output must be separate from the read-only source run')


def candidate(model, cfg, epoch, report, origin):
    return {'protocol': PROTOCOL, 'experiment_protocol': LOSS_PROTOCOL, 'continuation_protocol': STAGE_PROTOCOL,
            'config': cfg, 'epoch': epoch, 'source_epoch': cfg['source_epoch'],
            'model': {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'validation': report, 'origin': origin}


def score_value(report):
    score = report['hybrid_score']
    value = score['main_correlation']
    if score.get('score_protocol') != SCORE_PROTOCOL or not np.isfinite(value):
        raise ValueError('Only finite QC mean-r may select best_r.pt')
    return float(value)


def repair_best(path, best):
    """Do not rewrite an unchanged alias: final-evaluation SHA stays stable on resume."""
    try:
        existing = torch.load(path, map_location='cpu', weights_only=True)
        keys = ('config', 'epoch', 'origin', 'validation', 'stage_sha256')
        if all(existing.get(k) == best.get(k) for k in keys) and state_hash(existing['model']) == state_hash(best['model']):
            return
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError):
        pass
    atomic_save(path, best)


def run(args):
    prepared, info = load_prepared(args.prepared)
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    eligible_output(output, source)
    print('Auditing completed legacy source and original prepared data (no source writes).', flush=True)
    original, old_candidates, files = audit_source(source, prepared)
    cfg = stage_config(original['config'], args.total_epochs)
    manifest = {'protocol': STAGE_PROTOCOL, 'source_run': str(source), 'source_files': files,
                'prepared': str(prepared), 'prepared_sha256': sha(prepared/'prepared.json'),
                'output': str(output), 'config': cfg, 'implementation_sha256': implementation(),
                'sampling': SAMPLING, 'score_protocol': SCORE_PROTOCOL,
                'FM_training_started': False, 'source_training_epoch': 20}
    reference = QCReference(prepared, info)
    device = device_from_name(args.device)
    seed_all(42, True)
    model = SlowCNN(cfg['model_channels']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator()
    train_loader = DataLoader(Records(prepared/'train.npz'), batch_size=cfg['batch_records'],
                              shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(Records(prepared/'validation.npz'), batch_size=cfg['batch_records'],
                            shuffle=False, num_workers=0)
    history_path = output/'history.jsonl'
    last_path = output/'checkpoints/last.pt'
    resuming_training = bool(args.resume and last_path.exists())
    if args.resume:
        if read(output/'stage.json') != manifest:
            raise ValueError('Stage resume changed source, data, code, output, or total budget')
    if resuming_training:
        cp = torch.load(last_path, map_location='cpu', weights_only=True)
        if cp.get('continuation_protocol') != STAGE_PROTOCOL or cp['config'] != cfg or cp.get('stage_sha256') != sha(output/'stage.json'):
            raise ValueError('Wrong stage checkpoint')
        start, best = cp['epoch'], cp['best_payload']
        if not 20 <= start <= args.total_epochs:
            raise ValueError('Stage checkpoint epoch outside budget')
        lines = [json.loads(s) for s in history_path.read_text(encoding='utf-8').splitlines() if s.strip()]
        lines = [r for r in lines if r['epoch'] <= start]
        if [r['epoch'] for r in lines] != list(range(20, start+1)):
            raise ValueError('Stage history incomplete or duplicated')
        history_path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in lines), encoding='utf-8')
        repair_best(output/'checkpoints/best_r.pt', best)
    else:
        if not args.resume and output.exists() and any(output.iterdir()):
            raise FileExistsError('Use a NEW stage output directory, or --resume this exact stage')
        if args.resume and history_path.exists():
            # Initial stage save was interrupted before authoritative last.pt.
            rows = [json.loads(s) for s in history_path.read_text(encoding='utf-8').splitlines() if s.strip()]
            if any(r['epoch'] != 20 for r in rows):
                raise ValueError('Training history exists but authoritative stage last.pt is missing')
            history_path.write_text('', encoding='utf-8')
        output.mkdir(parents=True, exist_ok=True)
        json_write(output/'stage.json', manifest)
        json_write(output/'resolved_config.json', cfg)
        rescored, best = [], None
        # Do not carry the legacy score ledger into the new r-only rule.
        cache = {}
        for name, cp in old_candidates.items():
            print('Reassessing legacy candidate under QC mean-r: ' + name, flush=True)
            digest = state_hash(cp['model'])
            model.load_state_dict(cp['model'], strict=True)
            if digest not in cache:
                cache[digest] = validate(model, val_loader, device, reference)
            report = cache[digest]
            item = candidate(model, cfg, cp['epoch'], report, 'legacy:' + name)
            rescored.append({'candidate': name, 'epoch': cp['epoch'], 'state_sha256': digest,
                             'checkpoint_sha256': sha(source/(name+'.pt')), 'validation': report})
            if best is None or score_value(report) > score_value(best['validation']):
                best = item
        json_write(output/'source_candidates_qc.json', rescored)
        cp, start = original, original['epoch']
        best['stage_sha256'] = sha(output/'stage.json')
        atomic_save(output/'checkpoints/best_r.pt', best)
    # Extra candidate evaluation must not consume the source RNG stream.
    restore_training(cp, model, optimizer, generator, device)

    def save(epoch, record):
        record['best'] = {'epoch': best['epoch'], 'origin': best['origin'],
                          'main_correlation': score_value(best['validation'])}
        with history_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        payload = candidate(model, cfg, epoch, record.get('validation'), 'stage')
        payload.update(output_directory=str(output), stage_sha256=sha(output/'stage.json'),
                       optimizer=optimizer.state_dict(), rng=pack_rng(), loader_rng=generator.get_state(),
                       best_payload=best)
        # Commit authoritative last first; best alias is repaired on resume.
        atomic_save(last_path, payload)
        atomic_save(output/'checkpoints/best_r.pt', best)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

    if not resuming_training:
        save(start, {'epoch': start, 'phase': 'stage_start_source_last',
                     'validation': cache[state_hash(original['model'])]})
    for epoch in range(start+1, args.total_epochs+1):
        model.train()
        totals, seen = {}, 0
        for batch in train_loader:
            b = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = generate(model, b, cfg['context_seconds'])
            loss, parts = paired_objective(prediction, b['target'], b['mask'], cfg['group'],
                                          cfg['calibrated_weights'], cfg['residual_width_native'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            n = len(b['anchor'])
            for key,value in {'loss': float(loss.detach()), **parts}.items():
                totals[key] = totals.get(key, 0.) + n*value
            seen += n
        if not seen:
            raise ValueError('Empty training set')
        record = {'epoch': epoch, 'train': {k:v/seen for k,v in totals.items()}}
        if epoch % cfg['validate_every'] == 0 or epoch == args.total_epochs:
            report = validate(model, val_loader, device, reference)
            record['validation'] = report
            if score_value(report) > score_value(best['validation']):
                best = candidate(model, cfg, epoch, report, 'stage')
                best['stage_sha256'] = sha(output/'stage.json')
        save(epoch, record)
    # Prove no legacy weights, reports, or audit ledgers were overwritten.
    if any(sha(path) != value for path,value in files.items()):
        raise ValueError('Source run changed during continuation; results cannot be compared safely')
    json_write(output/'completed.json', {'completed': True, 'continuation_protocol': STAGE_PROTOCOL,
        'source_epoch': 20, 'total_epochs': args.total_epochs, 'additional_epochs': args.total_epochs-20,
        'score_protocol': SCORE_PROTOCOL, 'best_epoch': best['epoch'], 'best_origin': best['origin'],
        'best_score': best['validation']['hybrid_score'], 'FM_training_started': False})
    old_scores = read(output/'source_candidates_qc.json')
    old_best = max(old_scores, key=lambda item: score_value(item['validation']))
    rows = [('原现存候选最佳', old_best['epoch'], old_best['validation']['hybrid_score']),
            ('含续训候选最佳', best['epoch'], best['validation']['hybrid_score'])]
    text = ('# 慢 CNN 阶段续训\n\n仅比较相同 QC 下的组合平均 r，RMSE 不参与选模。\n\n'
            '| 候选范围 | 轮次 | Task1 r | Task2 r | 平均 r |\n|---|---:|---:|---:|---:|\n')
    for label,epoch,score in rows:
        text += (f"| {label} | {epoch} | {score['task1_r_missing11']:.8f} | "
                 f"{score['task2_r_missing11']:.8f} | {score['main_correlation']:.8f} |\n")
    (output/'report.md').write_text(text, encoding='utf-8')


def evaluate(args):
    stage = Path(args.stage).resolve()
    manifest = read(stage/'stage.json')
    if manifest.get('protocol') != STAGE_PROTOCOL or manifest['implementation_sha256'] != implementation():
        raise ValueError('Stage code/protocol changed')
    for path,digest in manifest['source_files'].items():
        if sha(path) != digest:
            raise ValueError('Source run changed after continuation: ' + path)
    prepared, info = load_prepared(manifest['prepared'])
    if sha(prepared/'prepared.json') != manifest['prepared_sha256']:
        raise ValueError('Stage prepared data changed')
    checkpoint = stage/'checkpoints'/(args.candidate+'.pt')
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (cp.get('continuation_protocol') != STAGE_PROTOCOL or cp['config'] != manifest['config'] or
            cp.get('stage_sha256') != sha(stage/'stage.json')):
        raise ValueError('Not a candidate from this stage')
    output = stage/'evaluations'/args.candidate
    request = {'checkpoint_sha256': sha(checkpoint), 'stage_sha256': sha(stage/'stage.json')}
    if (output/'request.json').exists():
        if read(output/'request.json') != request:
            raise ValueError('Evaluation candidate changed; use a new stage output')
        if (output/'evaluation.json').exists():
            done = read(output/'evaluation.json')
            if done.get('completed') and all(sha(output/name) == digest for name,digest in done['files'].items()):
                print('Existing exact evaluation verified: ' + str(output), flush=True)
                return
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Unrecognized evaluation output')
    output.mkdir(parents=True, exist_ok=True)
    json_write(output/'request.json', request)
    reference = QCReference(prepared, info)
    device = device_from_name(args.device)
    model = SlowCNN(cp['config']['model_channels']).to(device)
    model.load_state_dict(cp['model'], strict=True)
    loader = DataLoader(Records(prepared/'validation.npz'), batch_size=cp['config']['batch_records'], num_workers=0)
    report = validate(model, loader, device, reference, output=output)
    if cp.get('validation') is not None and abs(score_value(report)-score_value(cp['validation'])) > 2e-5:
        raise ValueError('Evaluation does not reproduce the saved QC mean-r selector')
    score = report['hybrid_score']
    csv_write(output/'correlation_score.csv', [{k:v for k,v in score.items() if k != 'summaries'}])
    marker = {'protocol': PROTOCOL, 'continuation_protocol': STAGE_PROTOCOL, 'completed': True,
              'group': cp['config']['group'], 'context_seconds': 50, 'epoch': cp['epoch'],
              'probe_checkpoint': str(checkpoint), 'probe_checkpoint_sha256': sha(checkpoint),
              'prepared_sha256': manifest['prepared_sha256'], 'B5_checkpoint_sha256': info['checkpoint_sha256'],
              'target_used_at_inference': False, 'score_protocol': SCORE_PROTOCOL, 'validation': report,
              'files': {str(p.relative_to(output)).replace('\\','/'):sha(p) for p in output.rglob('*')
                        if p.is_file() and p.name != 'evaluation.json'}}
    json_write(output/'evaluation.json', marker)
    print(json.dumps({'completed': True, 'epoch': cp['epoch'],
                      **{k:v for k,v in score.items() if k != 'summaries'}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    train = sub.add_parser('train')
    train.add_argument('--source-run', required=True, help='Original completed checkpoints directory; read only')
    train.add_argument('--prepared', required=True)
    train.add_argument('--output', required=True, help='New stage directory, never the legacy run')
    train.add_argument('--total-epochs', type=int, default=40)
    train.add_argument('--device', default='cuda')
    train.add_argument('--resume', action='store_true', help='Recover an interrupted NEW stage, same budget')
    train.add_argument('--execute-training', action='store_true')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--stage', required=True)
    evaluation.add_argument('--candidate', choices=('best_r', 'last'), required=True)
    evaluation.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.action == 'train':
        if not args.execute_training:
            parser.error('Actual training requires --execute-training')
        run(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
