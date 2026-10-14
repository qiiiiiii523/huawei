"""Slow-curve evaluation and learned (NOT target-assisted) replacement of B5 slow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg12gen.evaluate import evaluate_record_predictions, competition_score, write_competition_score, write_report
from baselines.B5.runtime import device_from_name
from . import PROTOCOL
from .core import (SlowCNN, Records, generate, device_batch, load_prepared, sha, json_write,
                   csv_write, csv_read, upsample, slow, correlations, macro, LEADS)


def hybrid_wave(baseline, predicted_slow):
    if baseline.shape != (12, 60000) or predicted_slow.shape != (11, 60000):
        raise ValueError('Invalid full-record shape')
    output = baseline.astype(np.float64).copy()
    output[1:] = baseline[1:] - slow(baseline[1:]) + predicted_slow
    # Preserve the original B5 I head; no true I replacement or target correction.
    return output


def plot_examples(output, records, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = [i for i, r in enumerate(metadata['validation']) if not r['shared_with_task1']]
    if not candidates:
        return []
    candidates.sort(key=lambda i: correlations(records[i]['baseline'][[9]], records[i]['target'][[9]])[0])
    selected = sorted({candidates[0], candidates[len(candidates) // 2], candidates[-1]})
    plots = output / 'plots'
    plots.mkdir()
    manifest = []
    for i in selected:
        item, meta = records[i], metadata['validation'][i]
        time = np.arange(60000) / 500
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        axes[0].plot(time, slow(item['target'][:1])[0], label='Input I slow (test-visible)', lw=.8)
        axes[1].plot(time, slow(item['target'][[9]])[0], label='Target V4 slow (label only)', color='black', lw=.8)
        axes[1].plot(time, slow(item['baseline'][[9]])[0], label='B5 V4 slow', lw=.8)
        axes[1].plot(time, item['learned_slow'][8], label='Learned V4 slow', lw=.8)
        axes[2].plot(time, item['target'][9], label='Target raw V4', color='black', lw=.6)
        axes[2].plot(time, item['baseline'][9], label='B5 raw V4', alpha=.5, lw=.6)
        axes[2].plot(time, hybrid_wave(item['baseline'], item['learned_slow'])[9], label='B5 fast + learned slow', alpha=.7, lw=.6)
        for ax in axes:
            ax.legend(loc='upper right')
            ax.set_ylabel('uV')
            ax.grid(alpha=.2)
            for seam in range(10, 120, 10):
                ax.axvline(seam, color='gray', ls=':', lw=.5)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(meta['target_record_id'] + ' | absolute voltage retained | diagnostic hybrid')
        fig.tight_layout()
        path = plots / f'record_{i:02d}_V4.png'
        fig.savefig(path, dpi=120)
        plt.close(fig)
        manifest.append({'path': str(path.relative_to(output)), 'target_record_id': meta['target_record_id'],
                         'selection': 'exclusive targets: low/median/high baseline V4 r'})
    return manifest


def run(args):
    prepared, info = load_prepared(args.prepared)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['config']['prepared_sha256'] != sha(prepared / 'prepared.json'):
        raise ValueError('Probe checkpoint belongs to different prepared data')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty evaluation output directory')
    evaluation = Path(info['evaluation_dir'])
    for name, value in info['evaluation_files'].items():
        print('Checking original evaluation: ' + name, flush=True)
        if sha(evaluation / name) != value:
            raise ValueError('Original B5 evaluation changed since prepare: ' + name)
    if sha(info['checkpoint']) != info['checkpoint_sha256']:
        raise ValueError('Original B5 best.pt changed since prepare')
    device = device_from_name(args.device)
    model = SlowCNN().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    metadata = json.loads((prepared / 'records.json').read_text(encoding='utf-8'))
    dataset = Records(prepared / 'validation.npz')
    predicted = []
    # The model receives ONLY the explicitly listed input fields. No labels/masks.
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0):
            inputs = {k: batch[k] for k in ('anchor', 'numeric', 'sex', 'field_mask', 'age_topcoded')}
            predicted.extend(generate(model, device_batch(inputs, device), checkpoint['config']['context_seconds']).cpu().numpy())
    scale = np.asarray(info['scale_uV'], float)
    learned = [upsample(p * scale[1:, None]) for p in predicted]
    if not all(np.isfinite(p).all() for p in learned):
        raise FloatingPointError('Nonfinite learned slow prediction')
    stores = {task: {kind: np.load(evaluation / task / (kind + '_uV.npy'), mmap_mode='r', allow_pickle=False)
                     for kind in ('prediction', 'target')} for task in ('task1', 'task2')}
    # Choose one representative original B5 pair per target. All target inputs were
    # checked identical in prepare; retain the pair-specific B5 prediction in scoring.
    target_map = {}
    for pair in metadata['pairs']['task2']:
        if pair['target_record_id'] not in target_map or pair['input_type'] == 'ecg_machine_d6':
            target_map[pair['target_record_id']] = pair
    index_map = {r['target_record_id']: i for i, r in enumerate(metadata['validation'])}
    records, lead_rows = [], []
    masks = dataset.arrays['mask']
    for i, meta in enumerate(metadata['validation']):
        pair = target_map[meta['target_record_id']]
        ids = pair['indices']
        raw = np.concatenate([stores['task2']['target'][j] for j in ids], axis=1).astype(float)
        baseline = np.concatenate([stores['task2']['prediction'][j] for j in ids], axis=1).astype(float)
        target_slow = slow(raw[1:])
        baseline_r = correlations(slow(baseline[1:]), target_slow)
        learned_r = correlations(learned[i], target_slow)
        baseline_rmse = np.sqrt(np.mean((slow(baseline[1:]) - target_slow) ** 2, axis=-1))
        learned_rmse = np.sqrt(np.mean((learned[i] - target_slow) ** 2, axis=-1))
        records.append({'target': raw, 'baseline': baseline, 'learned_slow': learned[i],
                        'baseline_r': baseline_r, 'learned_r': learned_r,
                        'baseline_rmse': baseline_rmse, 'learned_rmse': learned_rmse})
        for j, lead in enumerate(LEADS):
            lead_rows.append({**meta, 'lead': lead, 'quality': 'good' if masks[i, j] else 'flagged',
                              'B5_slow_r': baseline_r[j], 'learned_slow_r': learned_r[j],
                              'B5_slow_rmse_uV': baseline_rmse[j], 'learned_slow_rmse_uV': learned_rmse[j],
                              'B5_slow_mean_error_uV': float((slow(baseline[1:])[j] - target_slow[j]).mean()),
                              'learned_slow_mean_error_uV': float((learned[i][j] - target_slow[j]).mean())})
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / 'slow_record_lead.csv', lead_rows)
    summaries = []
    for cohort in ('all_unique', 'shared', 'exclusive'):
        indices = [i for i, r in enumerate(metadata['validation']) if cohort == 'all_unique' or
                   r['shared_with_task1'] == (cohort == 'shared')]
        if not indices:
            continue
        for quality in ('all', 'good'):
            mask = masks[indices] if quality == 'good' else None
            for method in ('B5', 'learned'):
                prefix = 'baseline' if method == 'B5' else 'learned'
                rs = np.stack([records[i][prefix + '_r'] for i in indices])
                valid = np.ones_like(rs, bool) if mask is None else mask
                summaries.append({'cohort': cohort, 'quality': quality, 'method': method, 'n_targets': len(indices),
                                  'slow_r_500Hz': macro(rs, mask),
                                  'n_undefined_eligible_record_lead_r': int((valid & ~np.isfinite(rs)).sum()),
                                  'slow_mean_lead_rmse_uV': macro(np.stack([records[i][prefix + '_rmse'] for i in indices]), mask)})
    csv_write(output / 'slow_summary.csv', summaries)
    metrics = {'baseline': {}, 'learned_slow_hybrid': {}}
    for task in ('task1', 'task2'):
        original_p, original_y = stores[task]['prediction'], stores[task]['target']
        rows = csv_read(evaluation / task / 'window_metadata.csv')
        folder = output / 'learned_slow_hybrid' / task
        folder.mkdir(parents=True)
        hybrid = np.lib.format.open_memmap(folder / 'prediction_uV.npy', mode='w+', dtype=np.float32, shape=original_p.shape)
        for pair in metadata['pairs'][task]:
            ids = pair['indices']
            p = np.concatenate([original_p[j] for j in ids], axis=1)
            h = hybrid_wave(p, learned[index_map[pair['target_record_id']]])
            hybrid[ids] = h.reshape(12, 12, 5000).transpose(1, 0, 2)
        hybrid.flush()
        for method, array in (('baseline', original_p), ('learned_slow_hybrid', hybrid)):
            summary, details = evaluate_record_predictions(array, original_y, task, rows)
            summary['result_kind'] = method
            summary['target_used_at_inference'] = False
            write_report(output / method / task, summary, details,
                         title='B5 reference' if method == 'baseline' else 'Learned slow replacement diagnostic (not FM integration)')
            metrics[method][task] = summary
        del hybrid
    score_rows = []
    for method, tasks in metrics.items():
        score = competition_score(tasks['task1']['r_missing11'], tasks['task2']['r_missing11'],
                                  tasks['task2']['ecg_machine_d6_task2_missing_lead_mean_rmse_uV'],
                                  tasks['task2']['body_scale_d6_task2_missing_lead_mean_rmse_uV'])
        write_competition_score(output / method, score)
        score_rows.append({'method': method, **score})
    csv_write(output / 'raw_score_comparison.csv', score_rows)
    plot_manifest = plot_examples(output, records, metadata) if not args.no_plots else []
    json_write(output / 'plot_manifest.json', plot_manifest)
    result = {'protocol': PROTOCOL, 'completed': True, 'context_seconds': checkpoint['config']['context_seconds'],
              'probe_checkpoint': str(Path(args.checkpoint).resolve()), 'probe_checkpoint_sha256': sha(args.checkpoint),
              'epoch': checkpoint['epoch'], 'prepared_sha256': sha(prepared / 'prepared.json'),
              'B5_checkpoint_sha256': info['checkpoint_sha256'], 'target_used_at_inference': False,
              'FM_model_modified': False, 'slow_filter_width': 501,
              'slow_diagnostic_reference_pair': 'machine pair preferred per unique target',
              'warning': 'Learned replacement is a diagnostic hybrid, not an integrated FM model. Raw scoring uses latest B5 target-equal aggregation within each device; it may differ from the original source report.',
              'scores': score_rows}
    json_write(output / 'evaluation.json', result)
    print(json.dumps({'completed': True, 'epoch': checkpoint['epoch'], 'context_seconds': result['context_seconds'],
                      'scores': score_rows}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-plots', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
