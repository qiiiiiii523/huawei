"""CPU compatibility/zero-branch check; no optimizer update or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_initialization(config, checkpoint):
    import torch
    from .checkpoint import checkpoint_preprocessor, initialize_model, model_from_checkpoint, scales_digest, scales_payload
    from .config import ModelConfig
    from .data import load_preprocessor, preprocessing_config
    from .model import B5UNet
    preprocessor = load_preprocessor(config)
    if checkpoint['scales_sha256'] != scales_digest(scales_payload(preprocessor)):
        raise ValueError('Configured frozen scales differ from the initialization checkpoint')
    checkpoint_preprocessor(checkpoint, preprocessing_config(config))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = B5UNet(ModelConfig.from_dict(config['model'])).eval()
    audit = initialize_model(model, checkpoint)
    reference = model_from_checkpoint(checkpoint, torch.device('cpu'))
    length = 5000
    generator = torch.Generator().manual_seed(42)
    state = torch.randn(1, 11, length, generator=generator)
    inputs = {'anchor': torch.linspace(-1., 1., length).reshape(1, 1, length),
              'numeric': torch.zeros(1, 3), 'sex': torch.full((1,), 2, dtype=torch.long),
              'field_mask': torch.zeros(1, 4, dtype=torch.bool), 'age_topcoded': torch.zeros(1, 1)}
    with torch.no_grad():
        prediction, anchor = model(state, torch.tensor([.3]), **inputs)
        previous, previous_anchor = reference(state, torch.tensor([.3]), **inputs)
    delta = float((prediction-previous).abs().max())
    anchor_delta = float((anchor-previous_anchor).abs().max())
    # A matching trained F checkpoint is not an identity-to-B5 check. Its own
    # reference is the same trained architecture, loaded strictly in both paths.
    if delta != 0 or anchor_delta != 0:
        raise ValueError(f'Initialization changed source behavior: velocity={delta}, anchor={anchor_delta}')
    return {'training_started': False, 'optimizer_updates': 0, 'cpu_forward_checked': True,
            'velocity_max_abs_difference': delta, 'anchor_max_abs_difference': anchor_delta,
            'parameters': sum(p.numel() for p in model.parameters()),
            'new_branch_parameters': sum(p.numel() for p in model.fusion_branch.parameters()) if model.fusion_branch else 0,
            'architecture_id': model.config.architecture_id, 'initialization': audit,
            'ready': True, 'note': 'Synthetic full-window forward check, not an ECG performance score.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/b5_f_finetune_meta.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    from .config import load_config
    from .checkpoint import load_checkpoint
    from .objective import file_sha256
    result = inspect_initialization(load_config(args.config), load_checkpoint(args.checkpoint))
    result['checkpoint'] = str(Path(args.checkpoint).resolve())
    result['checkpoint_sha256'] = file_sha256(args.checkpoint)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
