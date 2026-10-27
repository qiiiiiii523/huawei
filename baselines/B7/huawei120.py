"""Huawei model selection is always full real120s, never mean short-window r."""
from contextlib import contextmanager
import copy
from pathlib import Path
import torch
from .checkpoint import file_sha256
from .data import GROUPS, read_manifest

HUAWEI_SELECTION_PROTOCOL = 'B7-Huawei-full120-stitched-missing11-v1'


def full120_validation_manifest(views):
    from .huawei10 import VIEW_PROTOCOL
    if views.get('huawei_window_protocol') != VIEW_PROTOCOL:
        raise ValueError('Huawei10 validation requires the immutable views and original full120 source')
    source = views.get('view_source_manifest')
    if not source or file_sha256(source) != views.get('view_source_sha256'):
        raise ValueError('Huawei full-record source identity changed or is missing')
    original = read_manifest(source)
    parents = [r for r in original['records'] if r['split'] == 'validation']
    children = [r for r in views['records'] if r['split'] == 'validation']
    if not parents or {r['group'] for r in parents} != set(GROUPS):
        raise ValueError('Full120 validation requires the original three Huawei groups')
    known = {r['record_id']: r for r in parents}
    grouped = {k: [] for k in known}
    for row in children:
        key = row.get('parent_record_id')
        if key not in known:
            raise ValueError('Huawei validation view has an unknown parent')
        parent = known[key]
        if any(row[k] != parent[k] for k in ('subject_id', 'split', 'source', 'group')):
            raise ValueError('Huawei view changed patient/split/group ownership')
        if row.get('parent_target_record_id') != parent['target_record_id']:
            raise ValueError('Huawei view changed physical target identity')
        if Path(row['resolved_path']).resolve() != Path(parent['resolved_path']).resolve():
            raise ValueError('Huawei view and full record use different waveform files')
        start, stop = row.get('window_start'), row.get('window_stop')
        if type(start) is not int or stop != start + 5000 or start not in range(0, 60000, 5000):
            raise ValueError('Full120 selection requires twelve real consecutive10s views')
        grouped[key].append(start)
    if any(sorted(starts) != list(range(0, 60000, 5000)) for starts in grouped.values()):
        raise ValueError('Missing or duplicate Huawei validation windows')
    full = copy.deepcopy(original)
    full['records'] = copy.deepcopy(parents)
    full['scales_uV'] = list(views['scales_uV'])
    full['selection_protocol'] = HUAWEI_SELECTION_PROTOCOL
    # Parent QC/input reliability is the full-record scoring authority.
    # Do not drop short constant windows or invent a per-window score here.
    return full


@contextmanager
def full120_precision():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
