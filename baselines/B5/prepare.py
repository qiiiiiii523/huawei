"""Read-only Huawei fine-tuning preflight; no downloads or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .data import HuaweiTrainDataset, HuaweiValidationDataset, demographics_table, fit_huawei_scales, load_preprocessor, demographics_path


def preflight(config: dict[str, Any], fit_scales: bool = False) -> dict[str, Any]:
    from . import CACHE_PROTOCOL, EVALUATION_PROTOCOL
    from baselines.B5_best import SCORE_PROTOCOL
    from .sampling_settings import inference_sampling
    from ecg12gen.evaluate import QC_COMPETITION_SCORE_PROTOCOL as COMPETITION_SCORE_PROTOCOL
    report = {'cache_protocol': CACHE_PROTOCOL, 'evaluation_protocol': EVALUATION_PROTOCOL, 'context_used': False, 'training_started': False, 'downloads_started': False, 'stage': config['stage'], 'errors': []}
    report.update({'sampling': dict(config['sampling']), 'final_sampling': inference_sampling(),
                   'sampling_matches_final_default': config['sampling'] == inference_sampling(),
                   'selection_metric': config['validation']['selection_metric'],
                   'metric_aggregation_protocol': COMPETITION_SCORE_PROTOCOL, 'score_protocol': SCORE_PROTOCOL,
                   'output_dir': str(resolve_path(config, 'output_dir'))})
    try:
        if fit_scales:
            path = resolve_path(config, 'scales')
            if path.exists():
                raise FileExistsError('Scale artifact already exists; reuse the frozen file')
            preprocessor, audit = fit_huawei_scales(config)
            path.parent.mkdir(parents=True, exist_ok=True)
            preprocessor.save(path)
            report['scale_fit'] = audit
        else:
            preprocessor = load_preprocessor(config)
        report['scales'] = {key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}
        demographics = demographics_table(config)
        report['demographics'] = demographics.audit()
        report['demographics_path'] = str(demographics.path.resolve())
        from .metadata import read_csv, FIELDS
        from .objective import file_sha256
        report['demographics_csv_rows'] = len(read_csv(demographics.path))
        report['demographics_sha256'] = file_sha256(demographics.path)

        def coverage(rows):
            subjects = sorted({row['subject_id'] for row in rows})
            masks = [demographics.get(subject)['field_mask'] for subject in subjects]
            return {'dataset_subjects': len(subjects),
                    'subjects_with_any_observed_field': sum(bool(mask.any()) for mask in masks),
                    'subjects_with_all_fields': sum(bool(mask.all()) for mask in masks),
                    'field_coverage': {field: sum(bool(mask[i]) for mask in masks) for i,field in enumerate(FIELDS)}}
        train = HuaweiTrainDataset(config, preprocessor, demographics)
        first = train[0]
        report['huawei_train'] = {'windows': len(train), 'subjects': len({row['subject_id'] for row in train.rows}),
            'manifest_sha256': train.manifest_digest, 'anchor_shape': list(first['anchor'].shape),
            'target_shape': list(first['target'].shape), 'demographics_coverage': coverage(train.rows)}
        report['huawei_validation'] = {}
        for task in config['validation']['tasks']:
            dataset = HuaweiValidationDataset(config, task, preprocessor, demographics)
            dataset[0]
            report['huawei_validation'][task] = {'windows': len(dataset),
                'pairs': len({row['pair_id'] for row in dataset.rows}), 'manifest_sha256': dataset.manifest_digest,
                'devices': {source: len({row['pair_id'] for row in dataset.rows if row['input_type'] == source})
                            for source in sorted({row['input_type'] for row in dataset.rows})},
                'expected_windows_per_pair': 12, 'demographics_coverage': coverage(dataset.rows)}
    except (OSError, ValueError) as exc:
        report['errors'].append(f'Huawei/scales: {exc}')
    report['ready'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fit-scales', action='store_true', help='Only when the frozen train-only scale file does not exist')
    parser.add_argument('--data-root', help='Override the user-info data directory')
    parser.add_argument('--huawei-data-root', help='Override directory containing task1_record_context_v3/task2_record_context_v3')
    parser.add_argument('--scales', help='Override the existing frozen scale artifact')
    parser.add_argument('--report', help='Optional JSON report output')
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('data_root', 'huawei_data_root', 'scales'):
        value = getattr(args, key)
        if value:
            config['paths'][key] = str(Path(value).resolve())
    report = preflight(config, args.fit_scales)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded+'\n', encoding='utf-8')
    if not report['ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
