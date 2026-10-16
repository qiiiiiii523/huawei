"""QC-v3 aggregation, with ONLY mean r selecting slow-CNN stage candidates."""
from __future__ import annotations

import numpy as np

from .core import csv_write, json_write, sha
from .loss_metrics import Reference
from .stage_evaluator import evaluate_record_predictions

SCORE_PROTOCOL = 'mean-r-target-record-qc-v1'
SAMPLING = {'seed': 42, 'solver': 'heun', 'steps': 16, 'samples': 16}


def mean_r(r1, r2):
    values = np.asarray([r1, r2], float)
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.000001):
        raise ValueError('Both task r values must be finite and within [-1,1]')
    return {'task1_r_missing11': float(r1), 'task2_r_missing11': float(r2),
            'main_correlation': float(values.mean()), 'score_protocol': SCORE_PROTOCOL,
            'selection_metric': 'main_correlation'}


class QCReference(Reference):
    def __init__(self, prepared, info):
        self.quality = None
        # Require unchanged raw data/QC/demographics, not just the old prepared files.
        for key, item in info['source_checks'].items():
            if sha(item['path']) != item['sha256']:
                raise ValueError('Prepared source changed: ' + key)
        import json
        from pathlib import Path
        metadata = json.loads((Path(prepared) / 'records.json').read_text(encoding='utf-8'))
        with np.load(Path(prepared) / 'validation.npz', allow_pickle=False) as data:
            mask = data['mask']
            if mask.shape != (len(metadata['validation']), 11) or not np.isin(mask, [0, 1]).all():
                raise ValueError('Missing original per-target 11-lead validation QC')
            self.target_quality = {r['target_record_id']: np.r_[True, mask[i]].astype(bool)
                                   for i, r in enumerate(metadata['validation'])}
        if len(self.target_quality) != len(metadata['validation']):
            raise ValueError('Prepared validation targets must be unique')
        super().__init__(prepared, info)
        # Synchronous I is never scored; the original missing-11 QC is unchanged.

    def prediction(self, task, learned):
        if learned is None:
            return self.stores[task]['prediction']
        output = self.fast[task].copy()
        for pair in self.metadata['pairs'][task]:
            curve = learned[self.ids[pair['target_record_id']]]
            output[pair['indices'], 1:] += curve.reshape(11, 12, 5000).transpose(1, 0, 2)
        return output

    def score(self, learned, output=None):
        summaries = {}
        for task, store in self.stores.items():
            prediction = self.prediction(task, learned)
            quality = np.stack([self.target_quality[r['target_record_id']] for r in self.rows[task]])
            summary, details = evaluate_record_predictions(prediction, store['target'], task,
                self.rows[task], target_quality_mask=quality)
            # Bonus is retired. Keep voltage-error diagnostics only.
            summary = {k: v for k, v in summary.items() if 'bonus' not in k and k != 'competition_total_score'}
            summary.update(score_protocol=SCORE_PROTOCOL, checkpoint_selection_metric='main_correlation')
            summaries[task] = summary
            if output is not None:
                folder = output / 'learned_slow_hybrid' / task
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / 'prediction_uV.npy', prediction)
                np.save(folder / 'target_quality_mask.npy', quality)
                csv_write(folder / 'window_metadata.csv', self.rows[task])
                csv_write(folder / 'per_lead_metrics.csv', details)
                json_write(folder / 'overall_metrics.json', summary)
        result = mean_r(summaries['task1']['r_missing11'], summaries['task2']['r_missing11'])
        result['summaries'] = summaries
        return result


def validate(model, loader, device, reference, seconds=50, output=None):
    """Predictor receives I and demographics ONLY; targets enter metrics afterwards."""
    import torch
    from .core import generate, device_batch, correlations, macro
    from baselines.B5.checkpoint import pack_rng, restore_rng
    state = pack_rng()
    try:
        model.eval()
        predicted, targets, masks = [], [], []
        with torch.no_grad():
            for batch in loader:
                visible = {k: batch[k] for k in ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')}
                predicted.append(generate(model, device_batch(visible, device), seconds).cpu().numpy())
                targets.append(batch['target'].numpy())
                masks.append(batch['mask'].numpy())
        p, y, mask = np.concatenate(predicted), np.concatenate(targets), np.concatenate(masks)
        if not np.isfinite(p).all():
            raise FloatingPointError('Nonfinite predicted slow curves')
        learned = reference.restored(p)
        report = {'slow_r_unique_quality_good_50Hz': macro(correlations(p, y), mask),
                  'slow_rmse_normalized_unique_quality_good_50Hz': macro(np.sqrt(np.mean((p-y)**2, axis=-1)), mask),
                  'hybrid_score': reference.score(learned, output), 'baseline_score': reference.baseline_score}
        if not np.isfinite(report['slow_r_unique_quality_good_50Hz']):
            raise ValueError('Undefined diagnostic slow r')
        if output is not None:
            rows, summary = reference.diagnostics(learned, mask)
            csv_write(output / 'record_lead_metrics.csv', rows)
            csv_write(output / 'wave_summary.csv', summary)
            np.save(output / 'predicted_slow_normalized_50Hz.npy', p)
            # Retain fixed views of raw peaks, fast remainder and the 40s seam.
            from .loss_metrics import plot_fast_details
            records = []
            for i, meta in enumerate(reference.metadata['validation']):
                pairs = [q for q in reference.metadata['pairs']['task2'] if q['target_record_id'] == meta['target_record_id']]
                pair = next((q for q in pairs if q['input_type'] == 'ecg_machine_d6'), pairs[0])
                ids = pair['indices']
                records.append({'baseline': np.concatenate(reference.stores['task2']['prediction'][ids], axis=1),
                                'target': np.concatenate(reference.stores['task2']['target'][ids], axis=1),
                                'learned_slow': learned[i]})
            json_write(output / 'plot_manifest.json', plot_fast_details(output, records, reference.metadata))
        return report
    finally:
        restore_rng(state)
