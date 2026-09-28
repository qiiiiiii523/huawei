from __future__ import annotations

import torch

from .contracts import ContractError
from .m1_axial import ARCHITECTURE_ID
from .m1_axial import ARCHITECTURE_VERSION
from .m1_axial import AxialLeadTimeBlock
from .m1_axial import M1AxialLeadTimeModel
from .m1_axial import MultiScaleCNNEncoder

class M1MaskedCNNLeadTimeTransformer(M1AxialLeadTimeModel):
    def __init__(self) -> None:
        super().__init__(fusion_mode='none', task_id='task1')

    def forward(self, ecg: torch.Tensor, lead_mask: torch.Tensor | None = None, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        if ecg.ndim != 3 or ecg.shape[1] != 1:
            raise ContractError()
        if missing_mask is not None and lead_mask is not None:
            expected = ~lead_mask.to(dtype=torch.bool, device=ecg.device)
            if not torch.equal(missing_mask.to(dtype=torch.bool, device=ecg.device), expected):
                raise ContractError('missing_mask must complement lead_mask')
        return super().forward(ecg, lead_mask=lead_mask)
