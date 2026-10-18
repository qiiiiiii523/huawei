from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from baselines.B5_best.bundle import load_bundle
from baselines.B5_best.common import read, sha
from baselines.B5_best.slow import load_slow
from baselines.B5_best import SAMPLING
from baselines.B5.scoring import SCORE_PROTOCOL
from . import PROTOCOL, GROUPS


def multiscale_for_group(study, group):
    if group not in GROUPS:
        raise ValueError('Unknown experiment group')
    result = dict(study['multiscale'])
    result['weight'] = study['multiscale']['weight'] if group == GROUPS[2] else 0.
    return result


def code_digest():
    root = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob('*.py'))
    for name in ('B5', 'B5_best'):
        paths += list((root / 'baselines' / name).glob('*.py'))
    paths += list((root / 'ecg12gen').glob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def load_study(path):
    path = Path(path).resolve()
    study = read(path)
    if (study.get('protocol') != PROTOCOL or study.get('score_protocol') != SCORE_PROTOCOL
            or study.get('sampling') != SAMPLING or study.get('beta') != .75):
        raise ValueError('Wrong paired experiment / reliable-I scoring / sampling')
    if study.get('code_sha256') != code_digest():
        raise ValueError('Study code changed: use its frozen worktree or prepare a new study')
    cfg = study.get('multiscale', {})
    floors = np.asarray(cfg.get('std_floor', []))
    if (cfg.get('protocol') != 'native50-full120-multiscale-shape-v1' or cfg.get('widths') != [251, 751]
            or cfg.get('floor_source') != 'TRAIN-target-QC-only' or floors.shape != (3, 11)
            or not np.isfinite(floors).all() or np.any(floors <= 0)
            or not np.isfinite(cfg.get('weight', np.nan)) or cfg.get('weight', 0) <= 0
            or cfg.get('shape_huber_weight') != .1):
        raise ValueError('Missing or invalid frozen TRAIN-only multiscale settings')
    for name, expected in study['source_files'].items():
        if sha(name) != expected:
            raise ValueError('Read-only source changed: ' + name)
    bundle_path = path.parent / 'components.pt'
    if sha(bundle_path) != study['bundle_sha256']:
        raise ValueError('Shared initializer changed')
    return study, load_bundle(bundle_path)


def load_cache(path, study_path):
    """Read-only view of two existing caches. Never executes their old loaders."""
    path = Path(path).resolve()
    manifest = read(path)
    if manifest.get('study_sha256') != sha(study_path) or not manifest.get('completed'):
        raise ValueError('Cache view belongs to another study')
    verified = set()
    for group in GROUPS:
        source = manifest['train_sources'][group]
        for item in source['entries']:
            file = (Path(source['root']) / item['file']).resolve()
            identity = (str(file), item['sha256'])
            if identity in verified:
                continue
            if sha(file) != item['sha256']:
                raise ValueError('Training cache changed: ' + item['file'])
            verified.add(identity)
    if manifest['train_sources'][GROUPS[1]] != manifest['train_sources'][GROUPS[2]]:
        raise ValueError('B/C must use the identical control100 training cache')
    for item in manifest['validation']:
        if sha(Path(manifest['validation_root']) / item['file']) != item['sha256']:
            raise ValueError('Validation cache changed: ' + item['file'])
    return manifest


def validation_key(task, pair_id):
    return task + ':' + pair_id


class Records(Dataset):
    def __init__(self, root, entries):
        self.root, self.entries = Path(root), entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        with np.load(self.root / self.entries[index]['file'], allow_pickle=False) as data:
            return {k: torch.from_numpy(np.asarray(data[k]).copy()) for k in data.files}


def slow_model(bundle, device, trainable=True):
    return load_slow(bundle['slow'], device, 'slow_f').requires_grad_(trainable)


def load_checkpoint(path, study_path, device):
    study, bundle = load_study(study_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('protocol') != PROTOCOL or checkpoint.get('group') not in GROUPS
            or checkpoint.get('study_sha256') != sha(study_path)
            or checkpoint.get('sampling_config') != SAMPLING
            or checkpoint.get('multiscale') != multiscale_for_group(study, checkpoint.get('group'))):
        raise ValueError('Checkpoint belongs to another experiment')
    model = slow_model(bundle, device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return checkpoint, study, bundle, model
