"""Read-only B5 reference and common whole-record scoring for C/M/F."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecg12gen.evaluate import evaluate_record_predictions, competition_score, COMPETITION_SCORE_PROTOCOL
from .core import sha, csv_read, slow, upsample, correlations, macro, LEADS
from .evaluate import hybrid_wave


class Reference:
    def __init__(self, prepared, info):
        self.info = info
        self.metadata = json.loads((Path(prepared) / 'records.json').read_text(encoding='utf-8'))
        if COMPETITION_SCORE_PROTOCOL != 'correlation50_bonus10_target_record_v2':
            raise ValueError('This paired experiment requires the committed B5-exp v2 score protocol')
        evaluation = Path(info['evaluation_dir'])
        for name, expected in info['evaluation_files'].items():
            if sha(evaluation / name) != expected:
                raise ValueError('Original B5 evaluation changed: ' + name)
        if sha(info['checkpoint']) != info['checkpoint_sha256']:
            raise ValueError('Original B5 checkpoint changed')
        self.stores = {task: {kind: np.load(evaluation / task / (kind + '_uV.npy'), mmap_mode='r', allow_pickle=False)
                              for kind in ('prediction', 'target')} for task in ('task1', 'task2')}
        self.rows = {task: csv_read(evaluation / task / 'window_metadata.csv') for task in self.stores}
        self.ids = {r['target_record_id']: i for i, r in enumerate(self.metadata['validation'])}
        self.fast = {}
        # Precompute fixed B5 fast arrays once per run, preserving the original I head.
        for task, store in self.stores.items():
            fast = np.array(store['prediction'], dtype=np.float32)
            for pair in self.metadata['pairs'][task]:
                indices = pair['indices']
                full = np.concatenate([store['prediction'][j] for j in indices], axis=1).astype(float)
                full[1:] -= slow(full[1:])
                fast[indices] = full.reshape(12, 12, 5000).transpose(1, 0, 2)
            self.fast[task] = fast
        self.baseline_score = self.score(None)

    def score(self, learned):
        summaries = {}
        for task, store in self.stores.items():
            if learned is None:
                prediction = store['prediction']
            else:
                prediction = self.fast[task].copy()
                for pair in self.metadata['pairs'][task]:
                    indices = pair['indices']
                    s = learned[self.ids[pair['target_record_id']]]
                    prediction[indices, 1:] += s.reshape(11, 12, 5000).transpose(1, 0, 2)
            summaries[task], _ = evaluate_record_predictions(prediction, store['target'], task, self.rows[task])
        a, b = summaries['task1'], summaries['task2']
        return competition_score(a['r_missing11'], b['r_missing11'],
                                 b['ecg_machine_d6_task2_missing_lead_mean_rmse_uV'],
                                 b['body_scale_d6_task2_missing_lead_mean_rmse_uV'])

    def restored(self, native):
        scale = np.asarray(self.info['scale_uV'], float)[1:, None]
        return [upsample(p * scale) for p in native]

    def diagnostics(self, learned, masks):
        """Unique targets, representative machine pair; never modifies official score."""
        representative = {}
        for pair in self.metadata['pairs']['task2']:
            key = pair['target_record_id']
            if key not in representative or pair['input_type'] == 'ecg_machine_d6':
                representative[key] = pair
        rows = []
        for i, meta in enumerate(self.metadata['validation']):
            pair = representative[meta['target_record_id']]
            indices = pair['indices']
            store = self.stores['task2']
            target = np.concatenate([store['target'][j] for j in indices], axis=1).astype(float)
            baseline = np.concatenate([store['prediction'][j] for j in indices], axis=1).astype(float)
            combined = hybrid_wave(baseline, learned[i])
            target_slow = slow(target[1:])
            truth_fast = target[1:] - target_slow
            for method, raw in (('B5', baseline), ('hybrid', combined)):
                curve = slow(raw[1:])
                fast = raw[1:] - curve
                predicted_slow = slow(baseline[1:]) if method == 'B5' else learned[i]
                r_raw, r_fast, r_slow = (correlations(raw[1:], target[1:]),
                                       correlations(fast, truth_fast), correlations(predicted_slow, target_slow))
                error = raw[1:] - target[1:]
                seams = np.arange(5000, 60000, 5000)
                seam_error = error[:, seams] - error[:, seams - 1]
                for lead_index, lead in enumerate(LEADS):
                    peaks = np.abs(truth_fast[lead_index]) >= np.quantile(np.abs(truth_fast[lead_index]), .99)
                    rows.append({**meta, 'lead': lead, 'quality': 'good' if masks[i, lead_index] else 'flagged',
                                 'method': method, 'raw_r': r_raw[lead_index], 'fast_r': r_fast[lead_index],
                                 'slow_r': r_slow[lead_index],
                                 'raw_rmse_uV': np.sqrt(np.mean(error[lead_index] ** 2)),
                                 'fast_rmse_uV': np.sqrt(np.mean((fast[lead_index] - truth_fast[lead_index]) ** 2)),
                                 'slow_rmse_uV': np.sqrt(np.mean((predicted_slow[lead_index] - target_slow[lead_index]) ** 2)),
                                 'absolute_record_mean_error_uV': abs(np.mean(predicted_slow[lead_index] - target_slow[lead_index])),
                                 'fast_top1pct_target_amplitude_mae_uV': np.mean(np.abs(fast[lead_index, peaks] - truth_fast[lead_index, peaks])),
                                 'seam_error_step_mae_uV': np.mean(np.abs(seam_error[lead_index]))})
        summaries = []
        fields = ('raw_r', 'fast_r', 'slow_r', 'raw_rmse_uV', 'fast_rmse_uV', 'slow_rmse_uV',
                  'absolute_record_mean_error_uV', 'fast_top1pct_target_amplitude_mae_uV', 'seam_error_step_mae_uV')
        for cohort in ('all_unique', 'shared', 'exclusive'):
            for quality in ('all', 'good'):
                for method in ('B5', 'hybrid'):
                    selected = [r for r in rows if r['method'] == method and
                                (quality == 'all' or r['quality'] == 'good') and
                                (cohort == 'all_unique' or r['shared_with_task1'] == (cohort == 'shared'))]
                    if not selected:
                        continue
                    item = {'cohort': cohort, 'quality': quality, 'method': method}
                    for field in fields:
                        means = [np.mean([r[field] for r in selected if r['lead'] == lead and np.isfinite(r[field])])
                                 for lead in LEADS if any(r['lead'] == lead and np.isfinite(r[field]) for r in selected)]
                        item[field] = float(np.mean(means)) if means else float('nan')
                    summaries.append(item)
        return rows, summaries


def plot_fast_details(output, records, metadata):
    """Fixed 37.5-42.5s views, including the 40s seam; same records across groups."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = [i for i, meta in enumerate(metadata['validation']) if not meta['shared_with_task1']]
    if not candidates:
        return []
    candidates.sort(key=lambda i: correlations(records[i]['baseline'][[9]], records[i]['target'][[9]])[0])
    selected = sorted({candidates[0], candidates[len(candidates) // 2], candidates[-1]})
    folder = Path(output) / 'plots'
    folder.mkdir(exist_ok=True)
    manifest = []
    sl = slice(18750, 21250)
    time = np.arange(60000)[sl] / 500
    for i in selected:
        item = records[i]
        combined = hybrid_wave(item['baseline'], item['learned_slow'])
        for lead, index in (('V2', 7), ('V3', 8), ('V4', 9)):
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            truth = item['target'][index]
            for label, raw in (('Target', truth), ('B5', item['baseline'][index]), ('Hybrid', combined[index])):
                axes[0].plot(time, raw[sl], label=label, lw=.8)
                axes[1].plot(time, (raw - slow(raw[None])[0])[sl], label=label, lw=.8)
            axes[2].plot(time, (combined[index] - truth)[sl], label='Hybrid minus target', lw=.8)
            for ax, title in zip(axes, ('Raw voltage', 'Diagnostic fast remainder', 'Raw prediction error')):
                ax.set_title(title)
                ax.set_ylabel('uV')
                ax.axvline(40, ls=':', color='gray')
                ax.grid(alpha=.2)
                ax.legend()
            axes[-1].set_xlabel('Time (s)')
            fig.suptitle(metadata['validation'][i]['target_record_id'] + ' | ' + lead)
            fig.tight_layout()
            path = folder / f'record_{i:02d}_{lead}_fast_detail.png'
            fig.savefig(path, dpi=120)
            plt.close(fig)
            manifest.append({'path': str(path.relative_to(output)), 'target_record_id': metadata['validation'][i]['target_record_id'],
                             'selection': 'same exclusive low/median/high baseline V4 r; fixed 37.5-42.5s detail'})
    return manifest
