"""Matched continuation and residual training, only after explicit server authorization."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from baselines.B5.checkpoint import atomic_save, pack_rng, restore_rng
from baselines.B5.runtime import device_from_name, seed_all
from baselines.B5_best import SAMPLING
from baselines.B5_best.common import write, sha, state_sha
from baselines.B5_control_pcc.core import generate, device_batch, model_inputs
from baselines.B5_control_pcc.losses import objective
from baselines.B5_control_pcc.train import append_metrics
from . import PROTOCOL, GROUPS
from .common import load_study, load_cache, load_checkpoint, make_model, base_digest, Records
from .evaluate import evaluate


def run(study_path, cache_path, output, group, device_name, resume=None):
    study_path, cache_path, output = (Path(p).resolve() for p in (study_path, cache_path, output))
    if group not in GROUPS:
        raise ValueError('Unknown arm')
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Use empty arm output; put the tee log beside it')
    study, bundle = load_study(study_path)
    manifest = load_cache(cache_path, study_path)
    cache_sha, study_sha = sha(cache_path), sha(study_path)
    cfg, device = study['training'], device_from_name(device_name)
    seed_all(cfg['seed'], True)
    model = make_model(bundle, group, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(Records(manifest['train_root'], manifest['train']), batch_size=cfg['batch_records'],
                        shuffle=True, generator=generator, num_workers=0)
    scales = torch.tensor(bundle['fm']['scales']['d12'], device=device)
    start, updates, best_r, best_epoch = 1, 0, None, None
    if resume:
        cp, _, _, _ = load_checkpoint(resume, study_path, device)
        if cp['group'] != group or cp['output_dir'] != str(output) or cp['cache_sha256'] != cache_sha:
            raise ValueError('Resume must preserve arm, output and cache')
        if not (output / 'best.pt').is_file():
            raise FileNotFoundError('Resume requires the existing best.pt; do not lose the epoch-zero candidate')
        best_cp = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        if best_cp['study_sha256'] != study_sha or best_cp['group'] != group:
            raise ValueError('Existing best checkpoint belongs to another arm/study')
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        generator.set_state(cp['loader_rng'].cpu())
        start, updates = cp['epoch'] + 1, cp['updates']
        best_r, best_epoch = best_cp['best_r'], best_cp['best_epoch']
        restore_rng(cp['rng'])
    else:
        output.mkdir(parents=True, exist_ok=True)
        write(output / 'initialization.json', {'protocol': PROTOCOL, 'group': group,
              'study_sha256': study_sha, 'cache_sha256': cache_sha, 'training': cfg,
              'source_checkpoint': study['source_checkpoint'], 'source_best_epoch': study['source_best_epoch'],
              'initial_slow_state_sha256': study['initial_slow_state_sha256'],
              'fixed_FM_state_sha256': bundle['fm_state_sha256'], 'sampling': SAMPLING, 'beta': study['beta'],
              'slow_base_frozen': group == 'residual', 'extra_scale_loss_weight': 0.,
              'old_F_weight': study['residual_weight'], 'combo_PCC_weight': study['combo_weight'],
              'trainable_parameters': sum(p.numel() for p in parameters),
              'optimizer_initialization': 'fresh_AdamW', 'selection': 'final_hybrid_main_correlation',
              'epoch_zero_is_candidate': True, 'score_protocol': study['score_protocol'],
              'FM_optimizer_updates': 0})
        # New branch initialization must not change dropout/RNG sequence vs control.
        seed_all(cfg['seed'], True)

    def payload(epoch, score):
        return {'protocol': PROTOCOL, 'group': group, 'study_sha256': study_sha, 'cache_sha256': cache_sha,
                'output_dir': str(output), 'epoch': epoch, 'updates': updates, 'best_r': best_r,
                'best_epoch': best_epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'loader_rng': generator.get_state(), 'rng': pack_rng(), 'sampling_config': dict(SAMPLING),
                'validation_score': score, 'score_protocol': study['score_protocol'],
                'fixed_FM_state_sha256': bundle['fm_state_sha256'], 'extra_scale_loss_weight': 0.,
                'torch_version': str(torch.__version__)}

    def validate(epoch):
        rng = pack_rng()
        try:
            folder = output / 'validation' / f'epoch_{epoch:04d}'
            if folder.exists():
                import time
                folder = folder.with_name(folder.name + '_attempt_' + str(time.time_ns()))
            print({'phase': 'final_120s_hybrid_validation', 'group': group, 'epoch': epoch}, flush=True)
            return evaluate(model, bundle, study, cache_path, manifest, device, folder, group, epoch, epoch == 0)
        finally:
            restore_rng(rng)

    if not resume:
        score = validate(0)
        best_r, best_epoch = score['main_correlation'], 0
        append_metrics(output, 0, score)
        atomic_save(output / 'best.pt', payload(0, score))
        atomic_save(output / 'last.pt', payload(0, score))
    for epoch in range(start, cfg['epochs'] + 1):
        model.train()
        totals, count = {}, 0
        for batch in loader:
            batch = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = generate(model, model_inputs(batch))
            loss, parts = objective(prediction, batch, scales, study['residual_weight'], study['combo_weight'],
                                    study['residual_width_native'], beta=study['beta'], multiscale=None)
            loss.backward()
            if group == 'residual' and any(p.grad is not None or p.requires_grad for p in model.base.parameters()):
                raise ValueError('Frozen slow base unexpectedly received gradients')
            torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            n = len(prediction)
            for key, value in {'loss': loss, **parts}.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * n
            count += n
        record = {'epoch': epoch, 'updates': updates, 'train': {k: v / count for k, v in totals.items()}}
        score, improved = None, False
        if epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs']:
            score = validate(epoch)
            append_metrics(output, epoch, score)
            improved = score['main_correlation'] > best_r
            if improved:
                best_r, best_epoch = score['main_correlation'], epoch
            record['validation'] = score
        if state_sha(bundle['fm']['ema']) != bundle['fm_state_sha256']:
            raise ValueError('Frozen FM changed')
        if group == 'residual' and base_digest(model, group) != study['initial_slow_state_sha256']:
            raise ValueError('Frozen slow base changed')
        record.update(best_r=best_r, best_epoch=best_epoch, selected_best=improved,
                      FM_optimizer_updates=0, base_unchanged=(group == 'residual'))
        if improved:
            atomic_save(output / 'best.pt', payload(epoch, score))
        atomic_save(output / 'last.pt', payload(epoch, score))
        with (output / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)
    write(output / 'completed.json', {'completed': True, 'group': group, 'epochs': cfg['epochs'],
          'updates': updates, 'best_r': best_r, 'best_epoch': best_epoch, 'extra_scale_loss_weight': 0.,
          'FM_optimizer_updates': 0, 'slow_base_frozen': group == 'residual'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('study', 'cache', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume')
    p.add_argument('--execute-training', action='store_true')
    args = p.parse_args()
    if not args.execute_training:
        p.error('No training started. Use --execute-training explicitly on the server')
    run(args.study, args.cache, args.output, args.group, args.device, args.resume)


if __name__ == '__main__':
    main()
