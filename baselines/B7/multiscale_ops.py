"""Fixed-weight temporal operators with deterministic matrix-product backward.

CUDA linear-interpolation backward (and adaptive pooling in some runtimes) can
conflict with torch.use_deterministic_algorithms(True). These fixed matrices
express the same operations without those backward kernels. CUBLAS workspace
configuration is established by the training entry point before CUDA work.
Matrices are cached, not learned or included in checkpoint state.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def pool_weights(length, count, device, dtype):
    if length < 1 or count < 1:
        raise ValueError("Pool dimensions must be positive")
    matrix = torch.zeros(count, length, dtype=torch.float64)
    # Exactly the start=floor(i*L/K), stop=ceil((i+1)*L/K) adaptive-average bins.
    for index in range(count):
        start = index * length // count
        stop = ((index + 1) * length + count - 1) // count
        matrix[index, start:stop] = 1 / (stop - start)
    return matrix.to(device=device, dtype=dtype)


@lru_cache(maxsize=8)
def interpolation_weights(count, length, device, dtype):
    if count < 2 or length < 2:
        raise ValueError("Linear interpolation needs at least two input/output points")
    positions = torch.arange(length, dtype=torch.float64) * ((count - 1) / (length - 1))
    positions = positions.clamp(0, count - 1)
    left = positions.floor().long().clamp_max(count - 2)
    fraction = positions - left
    matrix = torch.zeros(count, length, dtype=torch.float64)
    columns = torch.arange(length)
    matrix[left, columns] = 1 - fraction
    matrix[left + 1, columns] = fraction
    return matrix.to(device=device, dtype=dtype)


def pool_control_features(features, count):
    weights = pool_weights(features.shape[-1], count, str(features.device), features.dtype)
    return features @ weights.T


def interpolate_control_points(coefficients, length):
    """Same endpoint-aligned piecewise-linear interpolation, retaining all points."""
    weights = interpolation_weights(coefficients.shape[-1], length,
                                    str(coefficients.device), coefficients.dtype)
    return coefficients @ weights
