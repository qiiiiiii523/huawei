"""One centre window plus separately encoded real neighbours for H's 250-sample halo."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from . import FIELDS


class TrainingWindows(Dataset):
    def __init__(self, run, entries, scales):
        self.root, self.entries = Path(run), entries
        self.scale = np.asarray(scales, np.float32)
        self.cache = OrderedDict()

    def __len__(self):
        return 12 * len(self.entries)

    def __getitem__(self, index):
        record, centre = divmod(index, 12)
        if record not in self.cache:
            with np.load(self.root / self.entries[record]["file"], allow_pickle=False) as data:
                self.cache[record] = {k: data[k].copy() for k in data.files}
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(record)
        data = self.cache[record]
        first, last = max(0, centre - 1), min(12, centre + 2)
        starts = slice(first * 5000, last * 5000)
        center = slice(centre * 5000, (centre + 1) * 5000)
        count = last - first
        inputs = {k: torch.from_numpy(np.repeat(np.asarray(data[k])[None], count, axis=0).copy()) for k in FIELDS}
        inputs["anchor"] = torch.from_numpy((data["anchor_uV"][:, starts].reshape(1, count, 5000)
                                            .transpose(1, 0, 2) / self.scale[0]).copy())
        return {"inputs": inputs, "centre": centre - first,
            "target": torch.from_numpy((data["target_uV"][:, center] / self.scale[:, None])[None].copy()),
            "fast_target": torch.from_numpy((data["fast_target"][:, center] / self.scale[1:, None])[None].copy()),
            "slow": torch.from_numpy((data["slow_uV"][:, center] / self.scale[1:, None])[None].copy()),
            "mask": torch.from_numpy(data["mask"][None].copy()),
            "key": self.entries[record]["target_record_id"] + ":" + str(centre * 5000)}
