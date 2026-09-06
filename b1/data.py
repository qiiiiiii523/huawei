"""Local path resolution without modifying public configuration or data."""
from pathlib import Path
from ecg12gen.dataset import ECGDataConfig

ROOT = Path(__file__).resolve().parents[1]


def local_config():
    original = ECGDataConfig.from_yaml(ROOT / 'configs/common.yaml')
    raw = {**original.raw, 'paths': dict(original.raw['paths'])}
    for key in ('task1_output', 'task2_output', 'task2_body_scale_ablation'):
        directory = original.path(key)
        if (directory / key).is_dir():
            directory = directory / key
        raw['paths'][key] = str(directory)
    body = Path(raw['paths']['task2_body_scale_ablation'])
    for key in ('task2_body_scale_ablation_manifest', 'task2_body_scale_b_train_input',
                'task2_body_scale_b_validation_input', 'task2_body_scale_b_metadata'):
        raw['paths'][key] = str(body / original.path(key).name)
    return ECGDataConfig(raw=raw, repository_root=ROOT)
