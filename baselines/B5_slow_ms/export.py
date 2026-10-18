"""Portable FM + slow correction; prediction needs I/demographics, never the training cache."""
import argparse
import copy
from pathlib import Path
import torch
from baselines.B5.checkpoint import atomic_save
from baselines.B5_best.bundle import validate_bundle
from baselines.B5_best.common import state_sha, sha, write
from baselines.B5_best.slow import load_slow
from . import PROTOCOL, SLOW_PROTOCOL, CORRECTION
from .common import load_checkpoint


def export(study_path, checkpoint_path, output):
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new portable bundle path')
    cp, study, original, model = load_checkpoint(checkpoint_path, study_path, torch.device('cpu'))
    bundle = copy.deepcopy(original)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if cp['group'] == 'residual':
        bundle['slow'] = {'protocol': SLOW_PROTOCOL, 'config': {'context_seconds': 50, 'group': 'F',
                          'model_channels': 48, 'correction': copy.deepcopy(CORRECTION)}, 'model': state}
    else:
        bundle['slow']['model'] = state
    bundle['slow_state_sha256'] = state_sha(state)
    score = cp.get('validation_score')
    if score is None:
        raise ValueError('Export an evaluated checkpoint, such as best.pt or final last.pt')
    bundle['reference_qc_r'] = {k: score[k] for k in ('task1_r_missing11', 'task2_r_missing11')}
    bundle['reference_score_protocol'] = study['score_protocol']
    bundle['provenance'].update(stable_reference=False, experiment=PROTOCOL,
          experiment_group=cp['group'], source_updated_slow_epoch=cp['epoch'], slow_sha256=sha(checkpoint_path),
          study_sha256=sha(study_path), FM_frozen=True, final_beta=study['beta'],
          source_best_checkpoint_sha256=study['source_files'][study['source_checkpoint']],
          slow_base_frozen=cp['group'] == 'residual', extra_scale_loss_weight=0.,
          training_seed=study['training']['seed'],
          reference_FM_handling='historical_fixed_control100_predictions_per_pair')
    validate_bundle(bundle)
    load_slow(bundle['slow'], torch.device('cpu'), 'slow_f')
    if state_sha(bundle['fm']['ema']) != original['fm_state_sha256']:
        raise ValueError('Export unexpectedly modified FM')
    atomic_save(output, bundle)
    write(output.with_suffix('.json'), {'bundle_sha256': sha(output), 'protocol': PROTOCOL,
          'group': cp['group'], 'epoch': cp['epoch'], 'FM_unchanged': True, 'sampling': study['sampling'],
          'beta': study['beta'], 'reference_qc_r': bundle['reference_qc_r'],
          'reference_score_protocol': study['score_protocol'], 'target_required_at_inference': False,
          'training_cache_required_at_inference': False, 'inference_module': 'baselines.B5_best.predict'})
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'checkpoint', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    export(args.study, args.checkpoint, args.output)
    print({'portable_bundle': args.output, 'training_started': False}, flush=True)


if __name__ == '__main__':
    main()
