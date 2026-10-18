import hashlib
from pathlib import Path
import torch
from baselines.B5_best import SAMPLING
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.slow import load_slow
from baselines.B5_best.common import read, sha, state_sha
from baselines.B5_control_pcc.common import Records
from . import PROTOCOL, GROUPS, CORRECTION
from .model import SlowMSResidual


def code_digest():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for directory in ('baselines/B5_slow_ms', 'baselines/B5_control_pcc', 'baselines/B5_best', 'baselines/B5', 'ecg12gen'):
        paths.extend((root / directory).glob('*.py'))
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def load_study(path):
    path = Path(path).resolve()
    study = read(path)
    if (study.get('protocol') != PROTOCOL or study.get('score_protocol') != SCORE_PROTOCOL or
            study.get('sampling') != SAMPLING or study.get('beta') != .75 or
            study.get('correction') != CORRECTION or study.get('extra_scale_loss_weight') != 0.):
        raise ValueError('Wrong residual study/QC/sampling/architecture/loss')
    if study.get('code_sha256') != code_digest():
        raise ValueError('Frozen study code changed; use its original worktree')
    for name, expected in study['source_files'].items():
        if sha(name) != expected:
            raise ValueError('Read-only source changed: ' + name)
    bundle_path = path.parent / 'components.pt'
    if sha(bundle_path) != study['bundle_sha256']:
        raise ValueError('Shared best initializer changed')
    bundle = load_bundle(bundle_path)
    if bundle['slow_state_sha256'] != study['initial_slow_state_sha256']:
        raise ValueError('Wrong shared slow weights')
    return study, bundle


def load_cache(path, study_path):
    view = read(path)
    if (view.get('completed') is not True or view.get('protocol') != PROTOCOL or
            view.get('study_sha256') != sha(study_path) or not view.get('train')):
        raise ValueError('Cache view belongs to another study or is incomplete')
    for entries, root in ((view['train'], view['train_root']), (view['validation'], view['validation_root'])):
        for item in entries:
            file = (Path(root) / item['file']).resolve()
            if not file.is_relative_to(Path(root).resolve()) or sha(file) != item['sha256']:
                raise ValueError('Source cache missing or modified: ' + str(file))
    return view


def make_model(bundle, group, device):
    if group not in GROUPS:
        raise ValueError('Unknown arm')
    base = load_slow(bundle['slow'], torch.device('cpu'), 'slow_f')
    if group == 'control':
        return base.to(device).requires_grad_(True)
    return SlowMSResidual(base).to(device)


def base_digest(model, group):
    return state_sha(model.base.state_dict()) if group == 'residual' else None


def load_checkpoint(path, study_path, device):
    study, bundle = load_study(study_path)
    cp = torch.load(path, map_location='cpu', weights_only=True)
    if (cp.get('protocol') != PROTOCOL or cp.get('group') not in GROUPS or
            cp.get('study_sha256') != sha(study_path) or cp.get('sampling_config') != SAMPLING or
            cp.get('score_protocol') != SCORE_PROTOCOL or cp.get('extra_scale_loss_weight') != 0. or
            cp.get('fixed_FM_state_sha256') != bundle['fm_state_sha256']):
        raise ValueError('Checkpoint belongs to another paired study')
    model = make_model(bundle, cp['group'], device)
    model.load_state_dict(cp['model'], strict=True)
    if cp['group'] == 'residual' and base_digest(model, cp['group']) != study['initial_slow_state_sha256']:
        raise ValueError('Frozen slow base was modified')
    return cp, study, bundle, model
