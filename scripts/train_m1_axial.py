'''Train formal M1 P0/P1 with the frozen main data and loss contracts.'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# YAML is loaded only after training opt-in

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# lazy imports below the explicit training gate
# from ecg12gen.m1_axial import M1AxialLeadTimeModel
# from ecg12gen.m1_axial_train import fit_m1
# from ecg12gen.m1_data import build_m1_datasets, fit_m1_preprocessor
# from ecg12gen.training import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'common.yaml'))
    parser.add_argument('--m1-config', default=str(ROOT / 'configs' / 'm1.yaml'))
    parser.add_argument('--attention-axes', choices=('both', 'time_only', 'lead_only'), default=None)
    parser.add_argument('--task-id', choices=('task1', 'task2'), required=True)
    parser.add_argument('--stage', choices=('P0_anchor_only', 'P1_joint_anchor'), required=True)
    parser.add_argument('--fusion-mode', choices=('none', 'film', 'gated_residual', 'film_gated_residual'), default='none')
    parser.add_argument('--p0-checkpoint')
    parser.add_argument('--context-source-type', choices=('watch_ecg', 'body_scale_d6', 'ecg_machine_d6'))
    parser.add_argument('--body-scale-variant', choices=('A_raw_window', 'B_detrend_0p2Hz_then_window'), default='A_raw_window')
    parser.add_argument('--context-dropout', type=float, default=0.0)
    parser.add_argument('--source-dropout', type=float, default=0.0)
    parser.add_argument('--freeze-anchor-epochs', type=int, default=0)
    parser.add_argument('--backbone-lr', type=float, default=1e-3)
    parser.add_argument('--fusion-lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-validation-batches', type=int, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--scales', help='Frozen main-v3 NPZ scales; use B5 shared scales for comparison')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if not args.execute_training:
        parser.error('Training was not started; explicitly pass --execute-training on the training server')
    import yaml
    from ecg12gen.m1_axial import M1AxialLeadTimeModel
    from ecg12gen.m1_axial_train import fit_m1
    from ecg12gen.m1_data import build_m1_datasets, fit_m1_preprocessor
    from ecg12gen.m1_protocol import protocol_metadata
    from ecg12gen.training import seed_everything
    if args.max_validation_batches is not None:
        parser.error('Partial validation cannot be used for record scoring')
    if args.epochs<1 or args.batch_size<1 or args.validate_every<1:
        parser.error('epochs, batch size and validation interval must be positive')
    if args.backbone_lr<=0 or args.fusion_lr<=0:
        parser.error('learning rates must be positive')
    if args.max_train_batches is not None and args.max_train_batches<1:
        parser.error('max-train-batches must be positive')
    if any((Path(args.output_dir)/name).exists() for name in ('m1_best.pt','m1_last.pt')):
        parser.error('Use a new run output directory')

    if args.stage == 'P0_anchor_only' and args.fusion_mode != 'none':
        raise SystemExit('P0_anchor_only requires --fusion-mode none')
    if args.stage == 'P0_anchor_only' and (args.p0_checkpoint or args.context_source_type):
        raise SystemExit('P0_anchor_only cannot receive a P0 checkpoint or context source')
    if args.stage == 'P1_joint_anchor' and args.fusion_mode == 'none':
        raise SystemExit('P1_joint_anchor requires a non-none fusion mode')
    if args.stage == 'P1_joint_anchor' and not args.p0_checkpoint:
        raise SystemExit('P1_joint_anchor requires --p0-checkpoint')
    if args.task_id == 'task1' and args.stage == 'P1_joint_anchor' and args.context_source_type not in (None, 'watch_ecg'):
        raise SystemExit('task1 P1 context-source-type must be watch_ecg')
    if args.task_id == 'task2' and args.stage == 'P1_joint_anchor' and args.context_source_type not in ('body_scale_d6', 'ecg_machine_d6'):
        raise SystemExit('task2 P1 requires exactly one d6 context source')

    seed_everything(42, deterministic=True)
    with Path(args.m1_config).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)['architecture']
    if args.attention_axes is not None:
        config = dict(config)
        if args.attention_axes == 'both': config.pop('attention_axes', None)
        else: config['attention_axes'] = args.attention_axes
    source_type = 'watch_ecg' if args.task_id == 'task1' and args.stage == 'P1_joint_anchor' else args.context_source_type
    preprocessor = fit_m1_preprocessor(args.config, args.task_id, args.stage, source_type, args.body_scale_variant, scales_path=args.scales)
    train, validation = build_m1_datasets(args.config, args.task_id, args.stage, source_type, preprocessor, args.body_scale_variant)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preprocessing_scales.json').write_text(
        json.dumps({key: value.tolist() for key, value in preprocessor.scale_uV_by_source.items()}, indent=2), encoding='utf-8'
    )
    preprocessor.save(output / 'preprocessing_scales.npz')
    model = M1AxialLeadTimeModel(
        fusion_mode=args.fusion_mode, task_id=args.task_id, config=config,
        context_dropout=args.context_dropout, source_dropout=args.source_dropout,
    )
    run = {
        **model.architecture_metadata,
        **protocol_metadata(),
        'scales_source': str(Path(args.scales).resolve()) if args.scales else 'strict Huawei train-only fit',
        'stage': args.stage,
        'context_source_type': source_type,
        'body_scale_variant': args.body_scale_variant,
        'p0_checkpoint': str(Path(args.p0_checkpoint).resolve()) if args.p0_checkpoint else None,
        'target_d12_scale_uV': preprocessor.scale_uV_by_source['d12'].tolist(),
        'seed': 42,
        'deterministic': True,
        'checkpoint_selection_metric': 'r_missing11',
        'evaluation_aggregation': 'record_macro_after_chronological_window_stitch',
        'target_quality_mask_source': 'metadata/device_interpretation_qc.csv',
        'experiments': ['M1-P0'] if args.stage == 'P0_anchor_only' else [
            {'film': 'M1-P1-C1', 'gated_residual': 'M1-P1-C2', 'film_gated_residual': 'M1-P1-C3'}[args.fusion_mode]
        ],
    }
    (output / 'm1_run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(f'M1 data: train={len(train)} validation={len(validation)} parameters={model.parameter_count:,}')
    checkpoint = fit_m1(
        model, train, validation, preprocessor.scale_uV_by_source['d12'], output,
        stage=args.stage, epochs=args.epochs, device=args.device,
        p0_checkpoint=args.p0_checkpoint, backbone_lr=args.backbone_lr,
        fusion_lr=args.fusion_lr, freeze_anchor_epochs=args.freeze_anchor_epochs,
        max_train_batches=args.max_train_batches, validate_every=args.validate_every, batch_size=args.batch_size, max_validation_batches=args.max_validation_batches, body_scale_variant=args.body_scale_variant,
    )
    print(f'M1 checkpoint: {checkpoint}')


if __name__ == '__main__':
    main()
