"""Physical-uV targets; smooth complete records before extracting windows."""
import numpy as np

GENERATED = (1, 6, 7, 8, 9, 10, 11)


def slow_record(wave):
    x = np.asarray(wave, np.float64)
    if x.ndim != 2 or x.shape[-1] < 501 or not np.isfinite(x).all():
        raise ValueError('Need finite channel/time complete record at 500Hz')
    padded = np.pad(x, ((0, 0), (250, 250)), mode='edge')
    sums = np.concatenate((np.zeros((len(x), 1)), np.cumsum(padded, axis=-1)), axis=-1)
    return (sums[:, 501:] - sums[:, :-501]) / 501


def missing11(slow7_uV, observed_i_uV):
    """Full-record linear lead restoration in physical units, using slow I."""
    y = np.asarray(slow7_uV, np.float64)
    i = np.asarray(observed_i_uV, np.float64).reshape(1, -1)
    if y.shape != (7, i.shape[-1]) or not np.isfinite(y).all():
        raise ValueError('Need seven physical slow curves aligned with observed I')
    si = slow_record(i)[0]; ii = y[0]
    return np.stack((ii, ii-si, -(si+ii)/2, si-ii/2, ii-si/2, *y[1:]))
