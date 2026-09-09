"""Task1 P0: train-only Ridge initialization followed by shared sync loss.

Put in scripts/; the companion b0_joint_anchor.py belongs in ecg12gen/models/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from ecg12gen.dataset import ECGDataConfig, JointAnchorDataset
from ecg12gen.d12_pretrain import StrictD12PretrainDataset
from ecg12gen.preprocessing import ECGPreprocessor, PreprocessingConfig
from ecg12gen.models.b0_joint_anchor import B0JointAnchor
from ecg12gen.losses import strict_anchor_pretrain_loss
from ecg12gen.evaluate import (
    evaluate_joint_anchor_predictions, evaluate_predictions,
    evaluate_centered_diagnostic, write_report,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, values):
    Path(path).write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')


def tensor(x):
    return torch.from_numpy(np.array(x, dtype=np.float32, copy=True))


class StrictTorchDataset(Dataset):
    def __init__(self, source, preprocessor, subject_split):
        self.source, self.preprocessor = source, preprocessor
        if any(subject_split.get(r['subject_id']) != 'train' for r in source.rows):
            raise ValueError('Strict index includes non-training subjects')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        if sample.split != 'train' or not np.array_equal(sample.X_ecg, sample.Y_12lead[:1]):
            raise ValueError('Strict training requires train-only same-window anchor')
        a = self.preprocessor.transform_window(sample.X_ecg, 'ecg_machine_i').model_signal
        y = self.preprocessor.transform_d12_target(sample.Y_12lead).model_signal
        return tensor(a), tensor(y)


def ridge_weights(dataset, alpha):
    """Minimize sum_{j,n,t}(w_j*a_nt-y_jnt)^2 + alpha*sum_j(w_j^2)."""
    xx, xy = 0.0, np.zeros(12, dtype=np.float64)
    for i in range(len(dataset)):
        a, y = dataset[i]
        x = a.numpy()[0].astype(np.float64)
        xx += np.dot(x, x)
        xy += y.numpy().astype(np.float64) @ x
    if xx <= 0 or not np.isfinite(xx) or not np.isfinite(xy).all():
        raise ValueError('Invalid train-only Ridge sufficient statistics')
    return (xy / (xx + alpha)).astype(np.float32)


def validation_arrays(dataset, preprocessor):
    anchors, targets, raw_anchors, metadata = [], [], [], []
    for sample in dataset:
        if sample.split != 'validation' or not np.array_equal(sample.anchor_i_ecg, sample.Y_12lead[:1]):
            raise ValueError('Invalid validation anchor or split')
        anchors.append(preprocessor.transform_window(sample.anchor_i_ecg, 'ecg_machine_i').model_signal)
        raw_anchors.append(sample.anchor_i_ecg.copy())
        targets.append(sample.Y_12lead.copy())
        metadata.append({'subject_id': sample.subject_id, 'window_id': sample.window_id,
                         'target_record_id': sample.target_record_id, 'input_type': sample.input_type})
    if not anchors:
        raise ValueError('No validation samples')
    return tensor(np.stack(anchors)), np.stack(targets), np.stack(raw_anchors), metadata


@torch.no_grad()
def predict(model, anchors, scale):
    model.eval()
    result = []
    for batch in anchors.split(16):
        # Context and target intentionally absent from inference.
        result.append((model(None, batch) * scale[None, :, None]).numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-dir', default='outputs/b0_joint_anchor/task1/preprocessing')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0:
        parser.error('epochs and ridge-alpha must be positive')
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    prep = (ROOT / args.preparation_dir).resolve()
    prepared = json.loads((prep / 'preparation_summary.json').read_text(encoding='utf-8'))
    scale_path = prep / 'preprocessing_scales.npz'
    if (prepared.get('completed') is not True or prepared.get('task_id') != 'task1'
        or prepared.get('fit_split') != 'train' or prepared.get('validation_used_to_fit_scales') is not False
        or sha(scale_path) != prepared['scale_sha256']):
        raise ValueError('Invalid or modified preprocessing preparation')
    cfg = ECGDataConfig.from_yaml(ROOT / 'configs/common.yaml')
    for name, path in {
        'common.yaml': ROOT / 'configs/common.yaml',
        'preprocessing.yaml': cfg.path('preprocessing_config'),
        'subject_split.csv': cfg.path('subject_split_csv'),
        'd12_strict_pretrain_index.csv': ROOT / 'metadata/d12_strict_pretrain_index.csv',
        'task1_pair_manifest.csv': cfg.path('task1_pair_manifest_csv'),
        'task1_window_metadata.csv': cfg.path('task1_output') / 'task1_window_metadata.csv',
    }.items():
        if sha(path) != prepared['source_sha256'][name]:
            raise ValueError(f'Data/preprocessing contract changed since preparation: {name}')
    with np.load(scale_path, allow_pickle=False) as f:
        scales = {k: f[k].copy() for k in f.files}
    for key, channels in [('d12', 12), ('ecg_machine_i', 1), ('watch_ecg', 1)]:
        s = scales[key]
        if s.shape != (channels,) or not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError(f'Invalid frozen scale: {key}')
    if not np.array_equal(scales['ecg_machine_i'], scales['d12'][:1]):
        raise ValueError('Anchor scale differs from D12 I')
    preprocessor = ECGPreprocessor(PreprocessingConfig.from_yaml(cfg.path('preprocessing_config')), scales)
    with cfg.path('subject_split_csv').open(encoding='utf-8-sig', newline='') as f:
        subject_split = {r['subject_id']: r['split'] for r in csv.DictReader(f)}
    strict_source = StrictD12PretrainDataset(cfg, 'd12_i_pretrain')
    train = StrictTorchDataset(strict_source, preprocessor, subject_split)
    validation = JointAnchorDataset(cfg, 'task1', 'validation')
    anchors, target, raw_anchor, metadata = validation_arrays(validation, preprocessor)
    train_subjects = {r['subject_id'] for r in strict_source.rows}
    if train_subjects & {r['subject_id'] for r in metadata}:
        raise ValueError('Training/validation subject overlap')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    protocol = yaml.safe_load((ROOT / 'configs/context_fusion_protocol.yaml').read_text(encoding='utf-8'))
    gate_bias = float(protocol['fusion_definitions']['initialization']['gate_logit_bias'])
    model = B0JointAnchor(context_channels=1, gate_logit_bias=gate_bias)
    print(f'Fitting Ridge initialization on {len(train)} strict training windows...', flush=True)
    weights = ridge_weights(train, args.ridge_alpha)
    with torch.no_grad():
        model.mapping.weight.copy_(torch.from_numpy(weights[:, None, None]))
    # All unused context/fusion modules remain frozen in P0.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001)
    loss_cfg = yaml.safe_load((ROOT / 'configs/losses.yaml').read_text(encoding='utf-8'))['strict_anchor_pretrain']
    loss_weights = dict(huber_weight=float(loss_cfg['huber_full_d12']),
                        pcc_weight=float(loss_cfg['pcc_full_d12']),
                        physiology_weight=float(loss_cfg['physiology']),
                        observed_weight=float(loss_cfg['observed_anchor_consistency']))
    scale = tensor(scales['d12'])
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    epochs = 1 if args.smoke_test else args.epochs
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preprocessing_scales.npz').write_bytes(scale_path.read_bytes())
    write_json(output / 'config_snapshot.json', {
        **vars(args), 'stage': 'P0_anchor_only', 'task_id': 'task1', 'seed': seed,
        'epochs': epochs, 'batch_size': 16, 'optimizer': 'AdamW', 'learning_rate': 0.001,
        'weight_decay': 0.0001, 'scheduler': None, 'early_stopping': False,
        'baseline_uV': 0, 'initialization': 'strict_train_closed_form_ridge',
        'ridge_objective': 'sum_squared_error_plus_alpha_times_squared_weights',
        'loss': 'strict_anchor_pretrain_loss', 'loss_weights': loss_weights,
        'model_description': 'Ridge-initialized linear backbone, gradient-trained shared loss',
        'architecture_id': model.architecture_id, 'architecture_config': model.architecture_config,
        'architecture_config_hash': model.architecture_config_hash,
        'checkpoint_metric': 'r_submit_12_raw_uV', 'tie_break': 'earliest_exact_equal',
        'train_windows': len(train), 'validation_windows': len(validation),
        'context_enabled': False, 'preprocessing_sha256': sha(scale_path),
    })
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for relative in ['scripts/train_b0_joint_p0.py', 'ecg12gen/models/b0_joint_anchor.py',
                     'ecg12gen/losses.py', 'ecg12gen/evaluate.py', 'ecg12gen/preprocessing.py',
                     'ecg12gen/dataset.py', 'ecg12gen/d12_pretrain.py', 'configs/losses.yaml',
                     'configs/context_fusion_protocol.yaml', 'configs/training_protocol_v1.yaml']:
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / relative).read_bytes())
    (output / 'preparation_summary.json').write_bytes((prep / 'preparation_summary.json').read_bytes())
    best_r, best_epoch = -float('inf'), None
    with (output / 'training_log.csv').open('w', encoding='utf-8', newline='') as f:
        writer = None
        for epoch in range(1, epochs + 1):
            model.train()
            total, count = 0.0, 0
            for step, (anchor, y) in enumerate(loader):
                if args.smoke_test and step >= 2:
                    break
                optimizer.zero_grad(set_to_none=True)
                prediction = model(None, anchor)
                loss = strict_anchor_pretrain_loss(prediction, y, anchor, d12_scale_uV=scale, **loss_weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Non-finite gradient')
                optimizer.step()
                total += float(loss.detach()) * len(anchor)
                count += len(anchor)
            raw = predict(model, anchors, scale)
            metrics, _, _, _ = evaluate_joint_anchor_predictions(raw, target, raw_anchor, 'task1')
            r = float(metrics['r_submit_12'])
            if not np.isfinite(r):
                raise FloatingPointError('Non-finite validation r')
            row = {'epoch': epoch, 'loss': total / count,
                   **{k: metrics[k] for k in ['r_raw_12', 'r_submit_12', 'r_missing11']},
                   'submit_rmse_uV': metrics['twelve_lead_mean_rmse_uV']}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); f.flush()
            if r > best_r:
                best_r, best_epoch = r, epoch
                torch.save({'state_dict': model.state_dict(), 'stage': 'P0_anchor_only',
                            'task_id': 'task1', 'architecture_id': model.architecture_id,
                            'architecture_config': model.architecture_config,
                            'architecture_config_hash': model.architecture_config_hash,
                            'preprocessing_sha256': sha(scale_path),
                            'epoch': epoch, 'smoke_test': args.smoke_test}, output / 'best_model.pt')
            print(f'P0 | Epoch {epoch}/{epochs} | loss={row["loss"]:.6f} | '
                  f'r_raw={row["r_raw_12"]:.6f} | r_submit={r:.6f} | '
                  f'r_missing11={row["r_missing11"]:.6f}', flush=True)
    saved = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'], strict=True)
    raw = predict(model, anchors, scale)
    metrics, raw_details, submit_details, submit = evaluate_joint_anchor_predictions(raw, target, raw_anchor, 'task1')
    write_report(output / 'evaluation', metrics, submit_details, title='B0 Task1 P0 submit-view validation')
    raw_metrics, _ = evaluate_predictions(raw, target, 'task1')
    write_report(output / 'evaluation/raw_prediction', raw_metrics, raw_details)
    centered, centered_details = evaluate_centered_diagnostic(submit, target, 'task1')
    write_report(output / 'evaluation/centered_diagnostic', centered, centered_details,
                 title='Centered diagnostic - not official')
    np.save(output / 'validation_prediction_raw.npy', raw)
    np.save(output / 'validation_prediction_submit.npy', submit)
    np.save(output / 'validation_anchor_i.npy', raw_anchor)
    with (output / 'validation_metadata.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0])); writer.writeheader(); writer.writerows(metadata)
    write_json(output / 'training_summary.json', {
        'completed': True, 'stage': 'P0_anchor_only', 'smoke_test': args.smoke_test,
        'epochs_completed': epochs, 'best_epoch': best_epoch, 'validation': metrics,
        'raw_prediction': raw_metrics, 'architecture_id': model.architecture_id,
        'architecture_config_hash': model.architecture_config_hash,
        'best_model_sha256': sha(output / 'best_model.pt'), 'preprocessing_sha256': sha(scale_path),
    })
    print(f'Completed. Results: {output}')


if __name__ == '__main__':
    main()
