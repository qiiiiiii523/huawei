"""Check paired configs/checkpoint/data and freeze initialization. Never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .config import ModelConfig, load_config, resolve_path
from .objective import assert_pair_configs, file_sha256, slow_settings


def freeze_snapshot(source: Path, destination: Path, expected_digest: str) -> None:
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError('Snapshot must have a different path than the original checkpoint')
    if destination.exists():
        if file_sha256(destination) != expected_digest:
            raise FileExistsError('Initializer snapshot already exists with different contents; never overwrite it')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation. Interrupted copies fail the next digest check rather than being reused.
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024*1024)
    if file_sha256(destination) != expected_digest:
        raise ValueError('Checkpoint changed during snapshot; use a stable source and a new snapshot path')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-config', default='configs/experiments/b5_slow_control.yaml')
    parser.add_argument('--slow-config', default='configs/experiments/b5_slow_trend.yaml')
    parser.add_argument('--init-checkpoint', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Use a new preflight report path')
    control, slow = load_config(args.control_config), load_config(args.slow_config)
    assert_pair_configs(control, slow)
    source = args.init_checkpoint.resolve()
    digest = file_sha256(source)
    from .checkpoint import load_checkpoint, scales_digest, scales_payload
    from .data import common_config, load_preprocessor
    from .runtime import build_datasets
    checkpoint = load_checkpoint(source)
    if checkpoint['architecture_hash'] != ModelConfig.from_dict(control['model']).fingerprint:
        raise ValueError('Initializer architecture/conditions differ from paired configs')
    preprocessor = load_preprocessor(control)
    if checkpoint['scales_sha256'] != scales_digest(scales_payload(preprocessor)):
        raise ValueError('Initializer and current frozen scales differ; do not refit')
    for config in (control, slow):
        output = resolve_path(config, 'output_dir')
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Paired training requires new outputs: {output}')
    train, validation = build_datasets(control, preprocessor)
    manifests = {'train': train.manifest_digest, **{name:dataset.manifest_digest for name,dataset in validation.items()}}
    manifests['demographics_csv'] = file_sha256(resolve_path(control, 'demographics'))
    common = common_config(control)
    for key in ('subject_split_csv','device_interpretation_qc_csv'):
        manifests[key] = file_sha256(common.path(key))
    if checkpoint['stage'] != 'finetune' or checkpoint['manifests'] != manifests:
        raise ValueError('This experiment expects the same Huawei fine-tuned checkpoint and data population')
    if file_sha256(source) != digest:
        raise ValueError('Source checkpoint changed during preflight')
    freeze_snapshot(source, args.snapshot, digest)
    report = {'ready':True, 'training_started':False, 'model_inference_started':False,
              'source_checkpoint':str(source), 'source_completed_epoch':int(checkpoint['epoch'])+1,
              'source_checkpoint_sha256':digest, 'snapshot':str(args.snapshot.resolve()),
              'scales_sha256':checkpoint['scales_sha256'], 'manifests':manifests,
              'train_windows':len(train), 'validation_windows':{name:len(ds) for name,ds in validation.items()},
              'control_slow_settings':slow_settings(control['loss']),
              'experiment_slow_settings':slow_settings(slow['loss']),
              'note':'Both runs use --init-checkpoint, fresh optimizer, identical settings except slow-trend weight.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
