"""Original 120-second recording coordinates, independent of FM flow time."""
import torch


def absolute_position(indices, offset=0):
    return (indices.to(torch.float32) + offset) / 30000.0 - 1.0


def view_metadata(metadata, row):
    result = dict(metadata) if metadata else {
        'numeric': [0., 0., 0.], 'sex': 2, 'field_mask': [False]*4, 'age_topcoded': [0.]}
    result['record_offset'] = row.get('window_start', 0)
    return result


def inference_offsets(metadatas, length, device):
    if length not in (5000, 60000):
        raise ValueError('C5 requires a real 10s view or full120s input')
    offsets = []
    for metadata in metadatas:
        if length == 5000 and (not metadata or 'record_offset' not in metadata):
            raise ValueError('10s C5 inference requires original record_offset; do not reset each view')
        value = (metadata or {}).get('record_offset', 0)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError('record_offset must be one integer sample index')
            value = value.item()
        if type(value) is not int or value < 0 or value + length > 60000 or value % 5000:
            raise ValueError('Invalid original record offset for a nonoverlap 10s view')
        offsets.append(value)
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def validate_position_views(manifest, stage):
    from .huawei10 import VIEW_PROTOCOL
    if stage != 'huawei10' or manifest.get('huawei_window_protocol') != VIEW_PROTOCOL:
        raise ValueError('C5 requires the same fixed Huawei10 views as C0')
    for row in manifest['records']:
        start, stop = row.get('window_start'), row.get('window_stop')
        if (type(start) is not int or type(stop) is not int or start < 0 or start % 5000
                or stop != start + 5000 or stop > 60000 or not row.get('parent_record_id')):
            raise ValueError('C5 view lost its original 120s position')
