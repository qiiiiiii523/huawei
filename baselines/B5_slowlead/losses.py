"""Supervise total P+G120 with the established slow and hybrid-PCC losses."""
from baselines.B5_slow120.losses import objective as source_objective


def objective(prediction, batch, scales):
    # base_uV already contains original P: replace it, never add P a second time.
    return source_objective(prediction - batch["base_slow"], batch, scales)
