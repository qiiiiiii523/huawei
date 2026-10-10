"""Linear 1-D resizing using deterministic-capable indexed tensor operations.

Same half-pixel coordinates and edge extension as interpolate(align_corners=False),
without CUDA's unsupported upsample_linear1d backward. No trainable parameters.
"""
from __future__ import annotations

import torch


def resize_linear_1d(signal: torch.Tensor, length: int) -> torch.Tensor:
    if signal.ndim != 3 or signal.shape[-1] < 1 or not isinstance(length, int) or length < 1:
        raise ValueError('Linear resizing expects [B,C,T] and a positive integer output length')
    if not signal.is_floating_point():
        raise ValueError('Linear resizing requires floating-point signals')
    source_length = signal.shape[-1]
    if source_length == length:
        return signal
    # Coordinate precision does not depend on AMP. Float/half outputs retain dtype.
    position = (torch.arange(length, device=signal.device, dtype=torch.float64) + .5) * (source_length / length) - .5
    left_unclamped = position.floor().to(torch.long)
    right_weight = position - left_unclamped
    left = left_unclamped.clamp(0, source_length - 1)
    right = (left_unclamped + 1).clamp(0, source_length - 1)
    values = signal.float() if signal.dtype in (torch.float16, torch.bfloat16) else signal
    weight = right_weight.to(values.dtype).reshape(1, 1, length)
    # With strict determinism enabled, PyTorch chooses deterministic CUDA
    # index_select backward, including repeated edge/source indices.
    result = values.index_select(-1, left) * (1 - weight) + values.index_select(-1, right) * weight
    return result.to(signal.dtype)
