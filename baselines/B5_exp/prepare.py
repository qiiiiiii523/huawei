"""Freeze train-only labels and test-visible I from the original B5 run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from baselines.B5.checkpoint import load_checkpoint, checkpoint_preprocessor
from baselines.B5.data import (HuaweiTrainDataset, HuaweiValidationDataset, common_config,
                               demographics_table, demographics_path, preprocessing_config)
from ecg12gen.evaluate import COMPETITION_SCORE_PROTOCOL, TARGET_RECORD_AGGREGATION
from . import BASE_COMMIT, PROTOCOL
from .core import POINTS, downsample, slow, sha, json_write, csv_read


def group_windows(rows):
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row['target_record_id']].append(i)
    for target, indices in grouped.items():
        indices.sort(key=lambda i: int(rows[i]['start_sample_500hz']))
        if [int(rows[i]['start_sample_500hz']) for i in indices] != list(range(0, POINTS, 5000)):
            raise ValueError(f'Expected exactly 12 chronological windows: {target}')
        if len({rows[i]['subject_id'] for i in indices}) != 1:
            raise ValueError('Target mixes subjects')
    return grouped


def pack(records, demographics, scales):
    keys = ('anchor', 'target', 'mask', 'numeric', 'sex', 'field_mask', 'age_topcoded')
    result = {k: [] for k in keys}
    for row in records:
        raw, anchor = row['raw'], row['anchor']
        if raw.shape != (12, POINTS) or anchor.shape != (1, POINTS) or not np.isfinite(raw).all() or not np.isfinite(anchor).all():
            raise ValueError('Expected finite 120-second records')
        if not np.array_equal(anchor, raw[:1]):
            raise ValueError('This probe requires synchronous same-record I')
        result['anchor'].append((downsample(anchor) / scales[:1, None]).astype(np.float32))
        # Labels are filtered before any cropping, absolute means retained.
        result['target'].append((downsample(slow(raw[1:])) / scales[1:, None]).astype(np.float32))
        result['mask'].append(np.asarray(row['mask'][1:], dtype=bool))
        meta = demographics.get(row['subject_id'])
        for key in ('numeric', 'sex', 'field_mask', 'age_topcoded'):
            result[key].append(meta[key])
    return {key: np.stack(value) for key, value in result.items()}


def prepare(checkpoint_path, evaluation, output):
    checkpoint_path, evaluation, output = map(lambda p: Path(p).resolve(), (checkpoint_path, evaluation, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty --output directory; prepared data are frozen')
    run = checkpoint_path.parent
    original_config = run / 'resolved_config.json'
    config = json.loads(original_config.read_text(encoding='utf-8-sig'))
    repo = Path(__file__).resolve().parents[2]
    config['repository_root'] = str(repo)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get('cache_protocol') != 'independent-record-context-v1':
        raise ValueError('Requires the original independent-record-context B5 run')
    if checkpoint.get('evaluation_protocol') not in ('machine_body_equal_weight_score_v1',
                                                    'machine_body_equal_weight_target_record_score_v2'):
        raise ValueError('Unsupported source B5 evaluation protocol')
    source = checkpoint['manifests']
    common = common_config(config)
    demographics = demographics_table(config)
    preprocessing = checkpoint_preprocessor(checkpoint, preprocessing_config(config))
    scales = preprocessing.scale_uV_by_source['d12'].astype(np.float32)
    sources = {'strict_index': repo / 'metadata/d12_strict_pretrain_index.csv',
               'demographics_csv': demographics_path(config),
               'subject_split_csv': common.path('subject_split_csv'),
               'device_interpretation_qc_csv': common.path('device_interpretation_qc_csv')}
    for task in ('task1', 'task2'):
        directory = common.path(task + '_output')
        sources[task + '_pair_manifest'] = common.path(task + '_pair_manifest_csv')
        sources[task + '_window_metadata'] = directory / (task + '_window_metadata.csv')
        for split in ('train', 'validation'):
            sources[task + '_' + split + '_target'] = directory / (task + '_' + split + '_target.npy')
    checks = {}
    for key, path in sources.items():
        print('Checking original B5 data: ' + key, flush=True)
        actual = sha(path)
        if source.get(key) != actual:
            raise ValueError(f'Original-run data changed: {key}, {path}')
        checks[key] = {'path': str(path), 'sha256': actual}

    train_dataset = HuaweiTrainDataset(config, preprocessing, demographics)
    if train_dataset.manifest_digest != source['train']:
        raise ValueError('Training sample selection differs from original B5')
    training = []
    for target, indices in sorted(group_windows(train_dataset.rows).items()):
        samples = [train_dataset.base[i] for i in indices]
        masks = [np.asarray(sample.target_quality_mask, bool) for sample in samples]
        if not all(np.array_equal(masks[0], mask) for mask in masks) or not masks[0][0]:
            raise ValueError('Inconsistent QC or unusable training I')
        raw = np.concatenate([sample.Y_12lead for sample in samples], axis=1)
        training.append({'target_record_id': target, 'subject_id': train_dataset.rows[indices[0]]['subject_id'],
                         'raw': raw, 'anchor': raw[:1], 'mask': masks[0]})

    provenance = json.loads((evaluation / 'inference_sampling.json').read_text(encoding='utf-8-sig'))
    if (Path(provenance['checkpoint']).resolve() != checkpoint_path or provenance.get('context_used') is not False or
            provenance['sampling'] != {'seed': 42, 'solver': 'heun', 'steps': 16, 'samples': 16}):
        raise ValueError('Evaluation must be the same B5 best.pt with Heun16/K16/seed42 and no context')
    validation, pairs, eval_files = {}, {}, {}
    task1_targets = set()
    for task in ('task1', 'task2'):
        folder = evaluation / task
        rows = csv_read(folder / 'window_metadata.csv')
        target_array = np.load(folder / 'target_uV.npy', mmap_mode='r', allow_pickle=False)
        prediction = np.load(folder / 'prediction_uV.npy', mmap_mode='r', allow_pickle=False)
        if target_array.shape != (len(rows), 12, 5000) or prediction.shape != target_array.shape:
            raise ValueError('Invalid original evaluation array shape')
        overall = csv_read(folder / 'overall_metrics.csv')[0]
        if overall.get('evaluation_view') != 'raw_uV' or overall.get('prediction_unit') != 'uV':
            raise ValueError('Original evaluation must contain raw uV')
        adapter = HuaweiValidationDataset(config, task, preprocessing, demographics)
        if adapter.manifest_digest != source[task]:
            raise ValueError('Validation selection differs from original B5')
        adapter_map = {(r['pair_id'], r['start_sample_500hz']): i for i, r in enumerate(adapter.rows)}
        grouped = defaultdict(list)
        for i, row in enumerate(rows):
            grouped[row['pair_id']].append(i)
        if len(adapter_map) != len(rows):
            raise ValueError('Evaluation omits or duplicates validation windows')
        pairs[task] = []
        for pair, indices in sorted(grouped.items()):
            indices.sort(key=lambda i: int(rows[i]['start_sample_500hz']))
            if [int(rows[i]['start_sample_500hz']) for i in indices] != list(range(0, POINTS, 5000)):
                raise ValueError('Validation pair is not a complete 120-second record')
            first = rows[indices[0]]
            for field in ('target_record_id', 'subject_id', 'input_type'):
                if any(rows[i][field] != first[field] for i in indices):
                    raise ValueError('Validation pair mixes identities')
            raw = np.concatenate([np.asarray(target_array[i]) for i in indices], axis=1)
            samples = [adapter.base[adapter_map[(pair, rows[i]['start_sample_500hz'])]] for i in indices]
            if not all(np.array_equal(samples[0].target_quality_mask, s.target_quality_mask) for s in samples):
                raise ValueError('Validation record mixes QC masks')
            current = np.concatenate([s.Y_12lead for s in samples], axis=1)
            anchor = np.concatenate([s.anchor_i_ecg for s in samples], axis=1)
            if not np.array_equal(current, raw):
                raise ValueError('Original evaluation targets differ from current cache')
            item = {'target_record_id': first['target_record_id'], 'subject_id': first['subject_id'],
                    'raw': raw, 'anchor': anchor, 'mask': np.asarray(samples[0].target_quality_mask, bool)}
            old = validation.setdefault(first['target_record_id'], item)
            if (old['subject_id'] != item['subject_id'] or not np.array_equal(old['raw'], raw) or
                    not np.array_equal(old['anchor'], anchor) or not np.array_equal(old['mask'], item['mask'])):
                raise ValueError('Repeated target does not have identical input/target/QC')
            if task == 'task1':
                task1_targets.add(first['target_record_id'])
            pairs[task].append({'pair_id': pair, 'target_record_id': first['target_record_id'],
                                'subject_id': first['subject_id'], 'input_type': first['input_type'], 'indices': indices})
        for name in ('target_uV.npy', 'prediction_uV.npy', 'window_metadata.csv', 'overall_metrics.csv'):
            eval_files[task + '/' + name] = sha(folder / name)
        del prediction, target_array
    values = [validation[k] for k in sorted(validation)]
    if {r['subject_id'] for r in training} & {r['subject_id'] for r in values}:
        raise ValueError('Training/validation subject overlap')
    metadata = {'train': [{k: r[k] for k in ('target_record_id', 'subject_id')} for r in training],
                'validation': [{k: r[k] for k in ('target_record_id', 'subject_id')} |
                               {'shared_with_task1': r['target_record_id'] in task1_targets,
                                'input_I_good': bool(r['mask'][0])} for r in values], 'pairs': pairs}
    train_pack, val_pack = pack(training, demographics, scales), pack(values, demographics, scales)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'train.npz', **train_pack)
    np.savez_compressed(output / 'validation.npz', **val_pack)
    json_write(output / 'records.json', metadata)
    if not task1_targets.issubset(validation):
        raise ValueError('Task1 targets must be covered by the unique Task2 target set')
    info = {'protocol': PROTOCOL, 'base_commit': BASE_COMMIT, 'completed': True, 'training_started': False,
            'source_evaluation_protocol': checkpoint['evaluation_protocol'],
            'applied_score_protocol': COMPETITION_SCORE_PROTOCOL,
            'applied_target_aggregation': TARGET_RECORD_AGGREGATION,
            'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha(checkpoint_path),
            'source_config': str(original_config), 'source_config_sha256': sha(original_config),
            'evaluation_dir': str(evaluation), 'evaluation_files': eval_files,
            'source_checks': checks, 'scale_uV': scales.tolist(), 'demographics': demographics.audit(),
            'train_records': len(training), 'validation_unique_targets': len(values),
            'validation_shared': len(task1_targets), 'validation_exclusive': len(values) - len(task1_targets),
            'train_validation_subject_overlap': 0, 'slow_width': 501, 'native_rate_hz': 50,
            'target_means_removed': False, 'input_source': 'same-record synchronous I; verified against B5 adapter',
            'files': {name: sha(output / name) for name in ('train.npz', 'validation.npz', 'records.json')}}
    json_write(output / 'prepared.json', info)
    print(json.dumps({k: info[k] for k in ('completed', 'training_started', 'train_records', 'validation_unique_targets',
                                         'validation_shared', 'validation_exclusive')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.checkpoint, args.evaluation_dir, args.output)


if __name__ == '__main__':
    main()
