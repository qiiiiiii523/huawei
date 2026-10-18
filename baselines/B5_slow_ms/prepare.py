"""Create a paired study from an explicitly selected best slow checkpoint; no training."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from baselines.B5.checkpoint import atomic_save
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best import SAMPLING
from baselines.B5_best.bundle import load_bundle, validate_bundle
from baselines.B5_best.common import read, write, sha, state_sha
from baselines.B5_control_pcc.prepare import source_cache, validation_audit, paired_training_config, NON_FM_FIELDS
from baselines.B5_control_pcc.core import generate, FIELDS
from . import PROTOCOL, GROUPS, CORRECTION
from .common import code_digest, make_model

RESULTS = Path('/home/qht/huawei/results')
SOURCE_PROTOCOLS = ('B5-control100-partner-multiscale-reliable-I-v2',
                    'B5-control100-scale-ablation-reliable-I-v3')


def selected_initializer(source_run, checkpoint_path, cache_run):
    """Audit historical artifacts as data, without executing their frozen study loader."""
    source_path = source_run / 'shared/study.json'
    source, view = read(source_path), read(source_run / 'shared/cache_view.json')
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (source.get('protocol') not in SOURCE_PROTOCOLS or cp.get('protocol') != source['protocol'] or
            cp.get('group') != 'adapt_multiscale' or cp.get('study_sha256') != sha(source_path) or
            cp.get('cache_sha256') != sha(source_run / 'shared/cache_view.json') or
            cp.get('sampling_config') != SAMPLING or cp.get('score_protocol') != SCORE_PROTOCOL or
            source.get('score_protocol') != SCORE_PROTOCOL or source.get('beta') != .75 or
            source.get('sampling') != SAMPLING or view.get('completed') is not True or
            view.get('study_sha256') != sha(source_path)):
        raise ValueError('Source must be an explicit completed reliable-I ALL checkpoint and matching study/cache')
    if (cp.get('epoch') != cp.get('best_epoch') or not isinstance(cp.get('epoch'), int) or
            cp.get('epoch', -1) < 1):
        raise ValueError('Select the trained ALL best.pt, not initializer or last.pt with a different best epoch')
    score = cp.get('validation_score') or {}
    fields = ('task1_r_missing11', 'task2_r_missing11', 'main_correlation')
    if (score.get('score_protocol') != SCORE_PROTOCOL or
            any(not np.isfinite(score.get(k, np.nan)) or not -1 <= score[k] <= 1 for k in fields) or
            abs(score['main_correlation'] - (score[fields[0]] + score[fields[1]]) / 2) > 1e-10 or
            not np.isfinite(cp.get('best_r', np.nan)) or
            abs(score['main_correlation'] - cp.get('best_r', np.nan)) > 1e-10):
        raise ValueError('Selected best has no valid full-record correlation score')
    source_files = {str(source_path): sha(source_path), str(checkpoint_path): sha(checkpoint_path),
                    str(source_run / 'shared/cache_view.json'): sha(source_run / 'shared/cache_view.json')}
    for name, expected in source['source_files'].items():
        if sha(name) != expected:
            raise ValueError('Historical source changed: ' + name)
        source_files[name] = expected
    components = source_run / 'shared/components.pt'
    if sha(components) != source['bundle_sha256']:
        raise ValueError('Historical components changed')
    bundle = load_bundle(components)
    source_files[str(components)] = sha(components)
    _, manifest, cached_bundle = source_cache(cache_run)
    if (bundle['fm_state_sha256'] != cached_bundle['fm_state_sha256'] or
            cp.get('fixed_FM_state_sha256') != bundle['fm_state_sha256'] or
            bundle['fm']['scales_sha256'] != cached_bundle['fm']['scales_sha256']):
        raise ValueError('Selected slow checkpoint must use the control100 FM and same scales')
    train = view['train_sources']['adapt_multiscale']
    key = lambda e: e['target_record_id']
    if (Path(train['root']).resolve() != (cache_run / 'cache').resolve() or
            Path(view['validation_root']).resolve() != (cache_run / 'cache').resolve() or
            sorted(train['entries'], key=key) != sorted(manifest['train'], key=key) or
            view['validation'] != manifest['validation'] or view['validation_rows'] != manifest['validation_rows']):
        raise ValueError('Selected source was not trained/evaluated with this exact control100 cache')
    for relative in ('shared/study.json', 'shared/components.pt', 'cache/cache.json'):
        p = cache_run / relative
        source_files[str(p)] = sha(p)
    bundle = copy.deepcopy(bundle)
    bundle['slow']['model'] = {k: v.detach().cpu().clone() for k, v in cp['model'].items()}
    bundle['slow_state_sha256'] = state_sha(bundle['slow']['model'])
    bundle['reference_qc_r'] = {k: score[k] for k in fields[:2]}
    bundle['reference_score_protocol'] = SCORE_PROTOCOL
    bundle['provenance'].update(stable_reference=False, slow_sha256=sha(checkpoint_path),
                                slow_component='ALL-best-source', source_updated_slow_epoch=cp['epoch'])
    validate_bundle(bundle)
    return bundle, manifest, source, score, source_files, cp['epoch']


def prepare(args):
    source_run, cache_run, output = (Path(p).resolve() for p in (args.source_run, args.control100_run, args.output))
    checkpoint = Path(args.source_checkpoint).resolve() if args.source_checkpoint else source_run / 'adapt_multiscale/best.pt'
    for p in (source_run, cache_run):
        if output == p or p in output.parents or output in p.parents:
            raise ValueError('Use a separate new study, outside the frozen source runs')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty NEW shared output; do not rerun prepare after success')
    bundle, manifest, source, score, files, epoch = selected_initializer(source_run, checkpoint, cache_run)
    training = paired_training_config(source['training'], args.epochs, args.training_seed, args.learning_rate)
    entries = sorted(manifest['train'], key=lambda e: e['target_record_id'])
    if (len({e['target_record_id'] for e in entries}) != len(entries) or
            len(entries) != source.get('train_records', len(entries))):
        raise ValueError('Training records must be unique and complete')
    for e in entries:
        with np.load(cache_run / 'cache' / e['file'], allow_pickle=False) as a:
            if (set(a.files) != NON_FM_FIELDS | {'fm_uV'} or not a['mask'][0] or not a['mask'][1:].any() or
                    a['anchor'].shape != (1, 6000) or a['slow_target'].shape != (11, 6000) or
                    a['target_uV'].shape != (12, 60000) or a['fm_uV'].shape != (12, 60000) or
                    any(not np.isfinite(a[k]).all() for k in a.files)):
                raise ValueError('Invalid reliable-I training cache: ' + e['file'])
    audit = validation_audit(manifest, cache_run / 'cache', {e['subject_id'] for e in entries})
    with np.load(cache_run / 'cache' / entries[0]['file'], allow_pickle=False) as a:
        inputs = {k: torch.from_numpy(a[k].copy()[None]) for k in FIELDS}
    torch.manual_seed(args.training_seed)
    control = make_model(bundle, 'control', torch.device('cpu')).eval()
    residual = make_model(bundle, 'residual', torch.device('cpu')).eval()
    with torch.no_grad():
        original, added = generate(control, inputs), generate(residual, inputs)
    difference = float((original - added).abs().max())
    if difference != 0 or not torch.isfinite(added).all():
        raise ValueError('Residual branch must exactly reproduce the selected best at initialization')
    output.mkdir(parents=True, exist_ok=True)
    atomic_save(output / 'components.pt', bundle)
    study = {'protocol': PROTOCOL, 'score_protocol': SCORE_PROTOCOL, 'sampling': dict(SAMPLING), 'beta': .75,
             'training': training, 'correction': copy.deepcopy(CORRECTION), 'extra_scale_loss_weight': 0.,
             'residual_weight': source['residual_weight'], 'residual_width_native': source['residual_width_native'],
             'combo_weight': source['combo_weight'], 'mean_weight': 0., 'reference_score': score,
             'initial_slow_state_sha256': bundle['slow_state_sha256'], 'source_best_epoch': epoch,
             'source_checkpoint': str(checkpoint), 'bundle_sha256': sha(output / 'components.pt'),
             'source_files': files, 'code_sha256': code_digest(), 'groups': list(GROUPS),
             'comparison': 'full slow-CNN update vs frozen base + new multiscale residual; original loss in both'}
    if study['combo_weight'] <= 0 or study['mean_weight'] != 0 or study['residual_width_native'] != 51:
        raise ValueError('Preserve the original SlowPCC objective')
    write(output / 'study.json', study)
    write(output / 'cache_view.json', {'protocol': PROTOCOL, 'completed': True, 'study_sha256': sha(output / 'study.json'),
          'train_root': str(cache_run / 'cache'), 'train': entries, 'validation_root': str(cache_run / 'cache'),
          'validation': manifest['validation'], 'validation_rows': manifest['validation_rows']})
    status = {'ready': True, 'training_started': False, 'optimizer_updates': 0, 'FM_optimizer_updates': 0,
              'source_checkpoint': str(checkpoint), 'source_best_epoch': epoch, 'reference_score': score,
              'initial_slow_state_sha256': bundle['slow_state_sha256'], 'zero_init_max_abs_difference': difference,
              'cpu_forward_shape': list(added.shape), 'train_records': len(entries), 'validation_audit': audit,
              'training': training, 'sampling': dict(SAMPLING), 'beta': .75, 'extra_scale_loss_weight': 0.,
              'trainable_parameters': {g: sum(p.numel() for p in m.parameters() if p.requires_grad)
                                       for g, m in [('control', control), ('residual', residual)]},
              'cache_regenerated': False, 'source_caches_modified': False}
    write(output / 'preflight.json', status)
    print(status, flush=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=str(RESULTS / 'B5-best-exp-Control100PCC/partner_multiscale_v1'))
    p.add_argument('--source-checkpoint', help='Defaults to source-run/adapt_multiscale/best.pt (trained ALL best)')
    p.add_argument('--control100-run', default=str(RESULTS / 'B5-best-exp-SlowPCC/control100_pcc20_cache_v1'))
    p.add_argument('--training-seed', type=int, default=43)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    prepare(args)


if __name__ == '__main__':
    main()
