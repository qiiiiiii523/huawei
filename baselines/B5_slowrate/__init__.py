"""Matched full-120s attention branches at three sampling rates."""
PROTOCOL = "B5-final-slowrate-residual-v1"
RATES = {"R2p5": 2.5, "R5": 5., "R10": 10.}
GROUPS = tuple(RATES)
