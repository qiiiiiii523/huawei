"""Task-2 body-scale A/B selectors for the joint-anchor public dataset."""
from __future__ import annotations
from pathlib import Path
from .contracts import ContractError, JointAnchorSample
from .dataset import ECGDataConfig, JointAnchorDataset

class BodyScaleVariantDataset(JointAnchorDataset):
    """Reusable body-scale-only view; target-time observed input remains I only."""
    def __init__(self, config: ECGDataConfig | str | Path, split: str, variant: str = "A_raw_window",
                 context_channel_indices: tuple[int, ...] | list[int] | None = None) -> None:
        super().__init__(config, "task2", split, variant, context_channel_indices)
        self._indices = [i for i in self._indices if self._rows[i].get("input_type") == "body_scale_d6"]
        if not self._indices:
            raise ContractError("No quality-gated body-scale joint-anchor rows")
