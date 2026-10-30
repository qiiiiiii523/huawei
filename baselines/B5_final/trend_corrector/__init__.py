"""Target-free, zero-mean 120-second trend corrector for B5-final."""

from .model import TrendCorrector, compose_d12, make_features

__all__ = ["TrendCorrector", "compose_d12", "make_features"]
