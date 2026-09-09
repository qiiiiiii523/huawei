"""Prepare train-fitted Task1 joint-anchor scales; never trains on validation.

Place in huawei/scripts. Requires the joint-anchor main data interfaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/b0_joint_anchor/task1/preprocessing')
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty directory: {output}')

    cfg = ECGDataConfig.from_yaml(ROOT / 'configs/common.yaml')
    split_path = cfg.path('subject_split_csv')
    split_rows = read_csv(split_path)
    subject_split = {r['subject_id']: r['split'] for r in split_rows}
    if len(subject_split) != len(split_rows):
        raise ValueError('Duplicate subject IDs in split file')
    strict = StrictD12PretrainDataset(cfg, 'd12_i_pretrain')
    train = JointAnchorDataset(cfg, 'task1', 'train')
    validation = JointAnchorDataset(cfg, 'task1', 'validation')
    if not len(strict) or not len(train) or not len(validation):
        raise ValueError('Strict train, joint train and validation must be nonempty')

    strict_subjects = set()
    strict_ids = set()
    d12 = np.empty((len(strict), 12, 5000), dtype=np.float32)
    for i, sample in enumerate(strict):
        sid = str(sample.meta['subject_id'])
        strict_id = str(sample.meta['strict_id'])
        if sample.split != 'train' or subject_split.get(sid) != 'train':
            raise ValueError(f'Non-training subject in strict index: {sid}')
        if strict_id in strict_ids:
            raise ValueError(f'Duplicate strict ID: {strict_id}')
        strict_ids.add(strict_id)
        strict_subjects.add(sid)
        if sample.Y_12lead.shape != (12, 5000) or not np.isfinite(sample.Y_12lead).all():
            raise ValueError('Invalid strict target waveform')
        if not np.array_equal(sample.X_ecg, sample.Y_12lead[:1]):
            raise ValueError('Strict input is not same-window d12 I')
        d12[i] = sample.Y_12lead

    watch = np.empty((len(train), 1, 5000), dtype=np.float32)
    counts = {}
    for split, dataset in [('train', train), ('validation', validation)]:
        subjects = set()
        for i, sample in enumerate(dataset):
            sid = str(sample.subject_id)
            if sample.split != split or subject_split.get(sid) != split:
                raise ValueError(f'Joint subject split mismatch: {sid}')
            subjects.add(sid)
            if sample.context_source_type != 'watch_ecg' or sample.anchor_source_type != 'ecg_machine_i':
                raise ValueError('Unexpected Task1 context or anchor source')
            if sample.context_ecg.shape != (1, 5000):
                raise ValueError('Expected a single watch context lead')
            if not np.isfinite(sample.context_ecg).all() or not np.isfinite(sample.Y_12lead).all():
                raise ValueError('Non-finite joint waveform')
            if not np.array_equal(sample.anchor_i_ecg, sample.Y_12lead[:1]):
                raise ValueError('Joint anchor is not exactly same-window target I')
            mask = np.asarray(sample.anchor_lead_mask)
            if mask.shape != (12,) or not mask[0] or mask[1:].any():
                raise ValueError('Only target-time I may be observed')
            if split == 'train':
                watch[i] = sample.context_ecg
        counts[split] = {'windows': len(dataset), 'subjects': sorted(subjects)}
    validation_subjects = set(counts['validation']['subjects'])
    if validation_subjects & (strict_subjects | set(counts['train']['subjects'])):
        raise ValueError('Training/validation subject overlap')

    print('PASS: split checks and strict/joint anchor identity checks', flush=True)
    print(f'Fitting scales using {len(strict)} strict train D12 and {len(train)} train watch windows...', flush=True)
    pre_cfg = PreprocessingConfig.from_yaml(cfg.path('preprocessing_config'))
    preprocessor = ECGPreprocessor.fit(pre_cfg, {'d12': d12, 'watch_ecg': watch})
    del d12, watch
    scales = preprocessor.scale_uV_by_source
    if not np.array_equal(scales['ecg_machine_i'], scales['d12'][:1]):
        raise ValueError('Machine-I scale must equal training D12-I scale')
    # Validate the normalized anchor identity without fitting on validation.
    for sample in validation:
        anchor = preprocessor.transform_window(sample.anchor_i_ecg, 'ecg_machine_i').model_signal
        target = preprocessor.transform_d12_target(sample.Y_12lead).model_signal
        if not np.array_equal(anchor, target[:1]):
            raise ValueError('Preprocessed anchor differs from target I')

    output.mkdir(parents=True, exist_ok=True)
    scale_path = output / 'preprocessing_scales.npz'
    np.savez_compressed(scale_path, **scales)
    sources = {
        'common.yaml': ROOT / 'configs/common.yaml',
        'preprocessing.yaml': cfg.path('preprocessing_config'),
        'subject_split.csv': split_path,
        'd12_strict_pretrain_index.csv': ROOT / 'metadata/d12_strict_pretrain_index.csv',
        'task1_pair_manifest.csv': cfg.path('task1_pair_manifest_csv'),
        'task1_window_metadata.csv': cfg.path('task1_output') / 'task1_window_metadata.csv',
        'training_protocol.yaml': cfg.path('training_protocol_config'),
        'context_fusion_protocol.yaml': ROOT / 'configs/context_fusion_protocol.yaml',
        'losses.yaml': ROOT / 'configs/losses.yaml',
        'prepare_b0_joint_task1.py': Path(__file__),
    }
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for name, source in sources.items():
        data = source.read_bytes()
        (snapshot / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    summary = {
        'completed': True, 'task_id': 'task1', 'protocol': 'joint_anchor',
        'git_head': git_head(), 'fit_split': 'train',
        'd12_scale_source': 'deduplicated_strict_train_index',
        'watch_scale_source': 'eligible_task1_joint_train_context',
        'strict_train_windows': len(strict), 'strict_train_subjects': len(strict_subjects),
        'joint_train_windows': len(train), 'joint_validation_windows': len(validation),
        'joint_train_subjects': len(counts['train']['subjects']),
        'joint_validation_subjects': len(counts['validation']['subjects']),
        'train_validation_subject_overlap': 0,
        'validation_used_to_fit_scales': False,
        'scale_uV_by_source': {k: v.tolist() for k, v in scales.items()},
        'scale_sha256': hashlib.sha256(scale_path.read_bytes()).hexdigest(),
        'source_sha256': hashes,
    }
    (output / 'preparation_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: frozen machine-I scale equals D12-I scale', flush=True)
    print(f'Strict train: {len(strict)}; joint train: {len(train)}; joint validation: {len(validation)}')
    print(f'Completed. Results: {output}')


if __name__ == '__main__':
    main()
