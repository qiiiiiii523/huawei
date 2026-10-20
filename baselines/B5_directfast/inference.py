"""Target-free full-record fast prediction. P and the original I head stay fixed."""
import numpy as np
import torch

from baselines.B5.flow import sample
from baselines.B5_best.combine import upsample_native
from baselines.B5_best.slow import predict_native
from baselines.B5_final.profile import SAMPLING
from . import FIELDS
from .model import visible_windows, assemble


@torch.no_grad()
def predict_base(model, group, anchor_uV, metadata, record_key, scale, slow_uV, original_i_uV, device, batch_size=4):
    if group not in ("D1", "F1") or batch_size < 1 or not isinstance(record_key, str) or not record_key:
        raise ValueError("Need F1/D1, positive batch size and stable record key")
    model.eval()
    fields = visible_windows(anchor_uV, metadata, scale[0], device)
    q = []
    for offset in range(0, 12, batch_size):
        size = min(batch_size, 12 - offset)
        inputs = {k: v[offset:offset + size] for k, v in fields.items()}
        if group == "D1":
            prediction = model(**inputs)
        else:
            keys = [record_key + ":" + str(5000 * i) for i in range(offset, offset + size)]
            prediction = sample(model, inputs, keys, **SAMPLING)[:, 1:]
        q.append(prediction.float().cpu().numpy() * scale[None, 1:, None])
    q = np.concatenate(q, axis=0).transpose(1, 0, 2).reshape(11, 60000)
    return assemble(q, slow_uV, original_i_uV)


@torch.no_grad()
def predict_frozen(frozen_fm, frozen_p, scale, anchor_uV, metadata, device, batch_size=4):
    fields = visible_windows(anchor_uV, metadata, scale[0], device)
    heads = []
    for offset in range(0, 12, batch_size):
        inputs = {k: v[offset:offset + batch_size] for k, v in fields.items()}
        heads.append(frozen_fm.encode_conditions(**inputs).anchor_prediction[:, 0].cpu().numpy())
    original_i = np.concatenate(heads).reshape(1, 60000) * scale[0]
    raw50 = np.asarray(anchor_uV, np.float32).reshape(1, 6000, 10).mean(-1) / scale[0]
    meta = {k: torch.from_numpy(np.asarray(metadata[k])[None].copy()).to(device) for k in FIELDS}
    native = predict_native(frozen_p, {"anchor": torch.from_numpy(raw50[None]).to(device), **meta})
    slow = upsample_native(native[0].cpu().numpy().astype(float) * scale[1:, None].astype(float))
    return slow.astype(np.float32), original_i.astype(np.float32)
