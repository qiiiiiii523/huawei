"""Reuse the exact final 120s QC scorer and fixed per-pair FM predictions."""
import argparse
from baselines.B5.runtime import device_from_name
from baselines.B5_best.common import sha
from baselines.B5_control_pcc.evaluate import evaluate as shared_evaluate
from .common import load_cache, load_checkpoint


def evaluate(model, bundle, study, cache_root, manifest, device, output, group, epoch, initial=False):
    return shared_evaluate(model, bundle, study, cache_root, manifest, device, output, group, epoch, initial,
                           model_info={'model': 'B5-SlowMSResidual-output-hybrid',
                                       'slow_CNN_trainable': group == 'control',
                                       'slow_base_frozen': group == 'residual',
                                       'correction_trainable': group == 'residual',
                                       'extra_scale_loss_weight': 0.})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('study', 'cache', 'checkpoint', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    device = device_from_name(args.device)
    cp, study, bundle, model = load_checkpoint(args.checkpoint, args.study, device)
    manifest = load_cache(args.cache, args.study)
    if cp['cache_sha256'] != sha(args.cache):
        raise ValueError('Checkpoint/cache mismatch')
    evaluate(model, bundle, study, args.cache, manifest, device, args.output, cp['group'], cp['epoch'])


if __name__ == '__main__':
    main()
