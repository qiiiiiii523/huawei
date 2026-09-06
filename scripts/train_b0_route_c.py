"""Task1 C1: fine-tune the shared strict pretrain on quality-weighted pseudo pairs."""
from pathlib import Path
import argparse
import csv
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ecg12gen.dataset import ECGDataConfig
from ecg12gen.b0_route_data import B0RouteDataset, load_frozen_preprocessor
from ecg12gen.b0_route_c import B0PseudoDataset, B0PseudoLoss
from ecg12gen.models.b0_linear import B0Linear
from ecg12gen.evaluate import evaluate_predictions
from train_b0_route import make_loader, predict_validation, save_evaluation
from train_b0_route_b import read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-run', default='outputs/b0_routes/task1/B_pretrain')
    parser.add_argument('--pseudo-dir', default='../task1_rpeak_pseudo_output')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    init_run = (ROOT / args.init_run).resolve()
    output = (ROOT / args.output_dir).resolve()
    pseudo_dir = (ROOT / args.pseudo_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    parent = read_json(init_run / 'config_snapshot.json')
    summary = read_json(init_run / 'training_summary.json')
    if (parent.get('route') != 'B' or parent.get('stage') != 'pretrain'
        or parent.get('task_id') != 'task1' or parent.get('smoke_test') is not False
        or parent.get('epochs') != 100 or summary.get('completed') is not True
        or summary.get('epochs_completed') != 100):
        raise ValueError('C1 requires the completed 100-epoch Task1 strict pretrain')
    checkpoint = init_run / 'best_model.pt'
    scales_path = init_run / 'preprocessing_scales.npz'
    if sha256(checkpoint) != summary['best_model_sha256']:
        raise ValueError('Pretraining checkpoint hash mismatch')
    if sha256(scales_path) != summary['scales_sha256']:
        raise ValueError('Pretraining scale hash mismatch')

    seed, batch_size = 42, 16
    epochs = 1 if args.smoke_test else 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = ECGDataConfig.from_yaml(ROOT / 'configs/common.yaml')
    preprocessor = load_frozen_preprocessor(config, scales_path)
    scale = torch.tensor(preprocessor.scale_uV_by_source['d12'], dtype=torch.float32)
    train = B0PseudoDataset(config, preprocessor, pseudo_dir)
    validation = B0RouteDataset(config, preprocessor, 'task1', 'validation', 'weak')
    train_loader = make_loader(train, batch_size, True, seed)
    validation_loader = make_loader(validation, batch_size, False, seed + 2)
    model = B0Linear(1)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = B0PseudoLoss(ROOT / 'configs/losses.yaml')
    quality = np.array([float(r['alignment_quality_score']) for r in train.rows])
    print(f'C1 data checks passed: {len(train)} windows, '
          f'{len({r["subject_id"] for r in train.rows})} train subjects, '
          f'quality={quality.min():.6f}..{quality.max():.6f}', flush=True)
    print(f'Batches per epoch: {len(train_loader)}; validation: {len(validation)} raw weak windows', flush=True)

    output.mkdir(parents=True, exist_ok=True)
    # Byte-identical copy maintains the scale provenance hash.
    (output / 'preprocessing_scales.npz').write_bytes(scales_path.read_bytes())
    write_json(output / 'config_snapshot.json', {
        **vars(args), 'task_id': 'task1', 'route': 'C1', 'stage': 'finetune',
        'seed': seed, 'batch_size': batch_size, 'epochs': epochs,
        'optimizer': 'AdamW', 'learning_rate': 0.001, 'weight_decay': 0.0001,
        'optimizer_reset_at_finetune': True, 'scheduler': None, 'early_stopping': False,
        'device': 'cpu', 'baseline_uV': 0, 'physiology_space': 'mV',
        'loss_profile': 'rpeak_pseudo', 'loss_weights': loss_fn.weights,
        'quality_reduction': 'sum(q_i * component_i) / sum(q_i); all four components',
        'mixture': False, 'validation_view': 'original_raw_weak_uV',
        'checkpoint_selection': 'best_raw_original_validation_r',
        'checkpoint_tie_break': 'earliest_epoch_exact_equality',
        'pseudo_dir': str(pseudo_dir), 'init_run': str(init_run),
        'init_checkpoint_sha256': sha256(checkpoint),
        'init_checkpoint_epoch': summary['best_epoch'],
        'train_windows': len(train), 'validation_windows': len(validation),
        'train_subjects': len({r['subject_id'] for r in train.rows}),
        'batches_per_epoch': len(train_loader),
        'quality_min': float(quality.min()), 'quality_max': float(quality.max()),
    })
    snapshot = output / 'source_snapshot'
    for relative in (
        'configs/common.yaml', 'configs/losses.yaml', 'configs/preprocessing.yaml',
        'configs/training_protocol_v1.yaml', 'scripts/train_b0_route.py',
        'scripts/train_b0_route_b.py', 'scripts/train_b0_route_c.py',
        'ecg12gen/b0_route_c.py', 'ecg12gen/b0_route_data.py',
        'ecg12gen/models/b0_linear.py', 'ecg12gen/losses.py',
    ):
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / relative).read_bytes())
    (snapshot / 'pseudo_metadata.csv').write_bytes(train.metadata_path.read_bytes())
    (snapshot / 'subject_split.csv').write_bytes(config.path('subject_split_csv').read_bytes())

    best_r, best_epoch, updates = -float('inf'), None, 0
    with (output / 'training_log.csv').open('w', encoding='utf-8', newline='') as f:
        writer = None
        for epoch in range(1, epochs + 1):
            model.train()
            totals, quality_mass = {}, 0.0
            for step, batch in enumerate(train_loader):
                if args.smoke_test and step >= 2:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss, logs = loss_fn(model(batch['inputs']), batch, scale)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Non-finite gradient')
                optimizer.step()
                updates += 1
                mass = float(batch['quality'].sum())
                quality_mass += mass
                for key, value in logs.items():
                    totals[key] = totals.get(key, 0.0) + value * mass
            prediction, target, _, _ = predict_validation(model, validation_loader, scale)
            metrics, _ = evaluate_predictions(prediction, target, 'task1')
            r = float(metrics['twelve_lead_mean_pearson_r'])
            if not np.isfinite(r):
                raise FloatingPointError('Non-finite validation r')
            row = {'epoch': epoch, **{k: v / quality_mass for k, v in totals.items()},
                   'validation_r': r,
                   'validation_rmse_uV': float(metrics['twelve_lead_mean_rmse_uV'])}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            if r > best_r:
                best_r, best_epoch = r, epoch
                torch.save(model.state_dict(), output / 'best_model.pt')
            print(f'C1 finetune | Epoch {epoch}/{epochs} | loss={row["total"]:.6f} | '
                  f'val_r={r:.6f} | best_r={best_r:.6f}', flush=True)
    model.load_state_dict(torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True))
    prediction, target, inputs, metadata = predict_validation(model, validation_loader, scale)
    original = save_evaluation(output / 'evaluation/original', prediction, target, 'task1', metadata)
    copied = prediction.copy()
    copied[:, :1] = inputs
    if not np.array_equal(copied[:, 1:], prediction[:, 1:]):
        raise RuntimeError('Copy-at-eval modified missing leads')
    copy_metrics = save_evaluation(output / 'evaluation/copy_at_eval', copied, target, 'task1', metadata)
    write_json(output / 'training_summary.json', {
        'completed': True, 'route': 'C1', 'stage': 'finetune',
        'smoke_test': args.smoke_test, 'epochs_completed': epochs,
        'best_epoch': best_epoch, 'optimizer_updates': updates,
        'original': original, 'copy_at_eval': copy_metrics,
        'init_checkpoint_sha256': sha256(checkpoint),
        'best_model_sha256': sha256(output / 'best_model.pt'),
        'scales_sha256': sha256(output / 'preprocessing_scales.npz'),
    })
    print(f'Completed. Results: {output}')


if __name__ == '__main__':
    main()
