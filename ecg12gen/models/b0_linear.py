from __future__ import annotations

import torch
from torch import nn


class B0Linear(nn.Module):
    """
    B0可训练线性基线。

    对每个采样点执行输入导联到12导联的线性映射，
    不包含时序卷积、非线性层、adapter或baseline head。

    输入：
        [batch, input_channels, time]

    输出：
        [batch, 12, time]
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 12,
    ) -> None:
        super().__init__()

        if input_channels not in {1, 6}:
            raise ValueError(
                "B0Linear supports only 1 or 6 input channels"
            )

        if output_channels != 12:
            raise ValueError(
                "B0Linear output_channels must be 12"
            )

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)

        self.mapping = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch, leads, time]"
            )

        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input leads, "
                f"got {inputs.shape[1]}"
            )

        predictions = self.mapping(inputs)

        if predictions.shape[1] != 12:
            raise RuntimeError(
                "B0Linear must output 12 leads"
            )

        return predictions