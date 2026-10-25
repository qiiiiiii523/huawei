"""Compare full120, window40/core10 and window40/core20 on identical EMA weights."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

from .checkpoint import file_sha256, load_checkpoint, model_from_checkpoint
from .config import SamplingConfig
from .data import read_manifest
from .evaluate import SCORE_PROTOCOL, aggregate, evaluate
from .train import code_sha

MODES = {'full120': (120, 120), 'window40_core10': (40, 10), 'window40_core20': (40, 20)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def data_identity(manifest_path, manifest):
    return {'manifest_sha256': file_sha256(manifest_path),
            'record_sha256': {r['record_id']: file_sha256(r['resolved_path']) for r in manifest['records']}}


def identity_for(checkpoint_path, manifest_path, batch_size):
    manifest = read_manifest(manifest_path)
    return manifest, {'checkpoint_sha256': file_sha256(checkpoint_path),
        **data_identity(manifest_path, manifest), 'code_sha256': code_sha(),
        'batch_size': batch_size, 'modes': {k: list(v) for k, v in MODES.items()}}


def validate_reports(output, identity):
    output = Path(output)
    if read(output / 'spec.json') != identity:
        raise ValueError('Comparison checkpoint/data/code/settings changed')
    scores = {}
    for mode in MODES:
        report = read(output / f'{mode}.json')
        if report['identity'] != identity or report['metrics'] != aggregate(report['records'], 'huawei120'):
            raise ValueError('Comparison report identity or aggregate is inconsistent')
        if report['sampling'] != asdict(SamplingConfig(window_seconds=MODES[mode][0], core_seconds=MODES[mode][1])):
            raise ValueError('Comparison sampling changed; K16/Heun16 must remain fixed')
        scores[mode] = report['metrics']['r']
    done = read(output / 'completed.json')
    if (done.get('completed') is not True or done.get('identity') != identity or done.get('scores') != scores
            or done.get('score_protocol') != SCORE_PROTOCOL or done.get('best_mode') != max(scores, key=scores.get)):
        raise ValueError('Comparison completion does not match reports')
    return scores


def compare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from .pipeline import Job, acquire_lock, validate_artifact
    lock = acquire_lock(output / '.comparison.lock')
    try:
        if type(args.batch_size) is not int or args.batch_size < 1:
            raise ValueError('Positive comparison trajectory batch required')
        manifest, identity = identity_for(args.checkpoint, args.manifest, args.batch_size)
        if (output / 'spec.json').exists() and read(output / 'spec.json') != identity:
            raise ValueError('Comparison identity changed; use a new output')
        write(output / 'spec.json', identity)
        if (output / 'completed.json').exists():
            return validate_reports(output, identity)
        checkpoint = load_checkpoint(args.checkpoint)
        sc = SamplingConfig(**checkpoint['sampling'])
        if (checkpoint['stage'] != 'huawei120' or sc != SamplingConfig(window_seconds=120, core_seconds=120)
                or checkpoint['scales_uV'] != manifest['scales_uV']):
            raise ValueError('Expected the trained full120 EMA with matching frozen scales')
        # Reuse the successful final training evaluation, never rerun it just for this table.
        directory = Path(args.checkpoint).resolve().parent
        job = Job('full120', (), 'train', ['--config', str(directory/'config.json'),
                  '--manifest', str(Path(args.manifest).resolve())], directory/'completed.json', None)
        validate_artifact(job)
        original = read(directory / f"validation_{checkpoint['update']:07d}.json")
        if original['metrics']['r'] != checkpoint['best_r']:
            raise ValueError('Final full120 report is not this EMA checkpoint')
        if original['metrics'] != aggregate(original['records'], 'huawei120'):
            raise ValueError('Final full120 report aggregate changed')
        if not (output / 'full120.json').exists():
            write(output / 'full120.json', {**original, 'identity': identity, 'sampling': asdict(sc),
                  'reused_training_final_evaluation': True})
        model = None
        for mode, (window, core) in MODES.items():
            report_path = output / f'{mode}.json'
            if report_path.exists():
                report = read(report_path)
                if report['identity'] != identity or report['metrics'] != aggregate(report['records'], 'huawei120'):
                    raise ValueError('Cached comparison report changed')
                continue
            if model is None:
                model = model_from_checkpoint(checkpoint, args.device)
            config = replace(sc, window_seconds=window, core_seconds=core)
            print(f'[compare] {mode}: K16 Heun16, trajectory_batch={args.batch_size}', flush=True)
            started = time.monotonic()
            metrics, rows = evaluate(model, manifest, config, 'huawei120', batch_size=args.batch_size)
            write(report_path, {'identity': identity, 'sampling': asdict(config), 'metrics': metrics,
                                'records': rows, 'seconds': time.monotonic()-started})
        scores = {mode: read(output/f'{mode}.json')['metrics']['r'] for mode in MODES}
        write(output / 'completed.json', {'completed': True, 'identity': identity, 'scores': scores,
              'score_protocol': SCORE_PROTOCOL, 'best_mode': max(scores, key=scores.get)})
        return validate_reports(output, identity)
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    print(json.dumps(compare(parser.parse_args()), ensure_ascii=False))


if __name__ == '__main__':
    main()
