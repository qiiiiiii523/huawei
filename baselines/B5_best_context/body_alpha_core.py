"""Fixed-strength body-II diagnostics. No training or target-conditioned inference."""
from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np

from .adapters import _visible_digest

ALPHAS = (0., .25, .5, .75, 1.)


def group_name(alpha):
    if alpha not in ALPHAS:
        raise ValueError("Use the five prespecified alpha values")
    return "alpha_" + str(int(round(alpha * 100))).zfill(3)


def mix_ii(base, corrected, alpha):
    """Mix raw-uV II without centering, alignment or changes to another lead."""
    if alpha not in ALPHAS or not np.isfinite(alpha):
        raise ValueError("Unsupported fixed alpha")
    base, corrected = np.asarray(base), np.asarray(corrected)
    if base.shape != corrected.shape or base.dtype.kind != "f":
        raise ValueError("II waveforms require matching floating arrays")
    if not np.isfinite(base).all() or not np.isfinite(corrected).all():
        raise ValueError("II waveforms must be finite")
    if alpha == 0:
        return base.copy()
    if alpha == 1:
        return corrected.astype(base.dtype, copy=True)
    return (base.astype(float) + alpha * (corrected.astype(float) - base)).astype(base.dtype)


class SharedCorrection:
    """Memoize one body forward per unique visible input, shared by all alphas."""
    def __init__(self, engine):
        self.engine, self.cache, self.forward_calls = engine, {}, 0

    def get(self, prediction, observed_i, history):
        digest = hashlib.sha256()
        for value in (prediction[1], observed_i):
            value = np.ascontiguousarray(value)
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        digest.update(("none" if history is None else
            history.source + ":" + history.record_id + ":" + _visible_digest(history)).encode())
        key = digest.hexdigest()
        if key not in self.cache:
            self.engine.mode = "correct"
            result, status = self.engine(prediction.copy(), observed_i, history)
            other = [0, *range(2, 12)]
            if not np.array_equal(result[other], prediction[other]):
                raise ValueError("Body correction changed a lead other than II")
            if result.shape != prediction.shape or not np.isfinite(result).all():
                raise ValueError("Invalid body correction")
            self.cache[key] = result[1].astype(prediction.dtype, copy=True), dict(status)
            self.forward_calls += 1
        corrected, status = self.cache[key]
        return corrected.copy(), dict(status)


class FixedAlphaAdapter:
    def __init__(self, shared, alpha):
        group_name(alpha)
        self.shared, self.alpha, self.mode = shared, alpha, "correct"

    def __call__(self, prediction, observed_i, history):
        corrected, status = self.shared.get(prediction, observed_i, history)
        result = prediction.copy()
        result[1] = mix_ii(prediction[1], corrected, self.alpha)
        residual = result[1].astype(float) - prediction[1]
        status.update(adapter="body_fixed_alpha", fixed_alpha=self.alpha,
            max_II_correction_uV=float(np.max(np.abs(residual))),
            rms_II_correction_uV=float(np.sqrt(np.mean(residual ** 2))),
            target_used_for_alpha=False, alpha_shared_across_12_windows=True)
        return result, status


def cases():
    return [("baseline", "none", None, None), ("d6_only", "fir25", None, None),
            *[(group_name(a), "fir25", group_name(a), None) for a in ALPHAS]]


def target_mean(rows):
    targets = defaultdict(list)
    for row in rows:
        value = row["r_II"]
        if value is not None and np.isfinite(value):
            targets[row["target_record_id"]].append(float(value))
    return float(np.mean([np.mean(v) for v in targets.values()])) if targets else None


def oracle_diagnostic(records):
    """Target-level oracle on a common finite cohort, NEVER an inference adapter.

    Repeated pairs are averaged within each target before selecting alpha. Ties
    prefer the larger alpha. Incomplete/undefined targets are explicitly omitted
    from both the oracle and its fixed-alpha comparators.
    """
    targets = defaultdict(lambda: defaultdict(list))
    for row in records:
        targets[row["target_record_id"]][row["alpha"]].append(row["r_II"])
    means, excluded = {}, []
    for target, by_alpha in targets.items():
        if (set(by_alpha) != set(ALPHAS) or
                any(v is None or not np.isfinite(v) for values in by_alpha.values() for v in values)):
            excluded.append(target)
            continue
        if len({len(v) for v in by_alpha.values()}) != 1:
            raise ValueError("Alpha populations differ within a target")
        means[target] = {a: float(np.mean(v)) for a, v in by_alpha.items()}
    if not means:
        raise ValueError("No common finite targets for alpha diagnosis")
    fixed = {a: float(np.mean([v[a] for v in means.values()])) for a in ALPHAS}
    best_fixed = max(ALPHAS, key=lambda a: (fixed[a], a))
    chosen = {target: max(ALPHAS, key=lambda a: (values[a], a)) for target, values in means.items()}
    upper = float(np.mean([means[t][a] for t, a in chosen.items()]))
    return {"diagnostic_only": True, "deployable": False, "target_used_for_oracle_selection": True,
        "target_weighting": "mean pairs within target, then equal target weights",
        "comparable_targets": len(means), "excluded_targets": sorted(excluded),
        "fixed_alpha_II_r_common_cohort": {str(a): fixed[a] for a in ALPHAS},
        "best_fixed_alpha": best_fixed, "best_fixed_II_r": fixed[best_fixed],
        "oracle_II_r": upper, "oracle_gain_over_best_fixed": upper - fixed[best_fixed],
        "oracle_alpha_counts": {str(a): sum(v == a for v in chosen.values()) for a in ALPHAS},
        "note": "Validation upper bound; never use target-selected alpha at test time."}, chosen
