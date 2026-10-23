"""PTB physical calibration and conservative technical QC, reused from public-slow."""
import numpy as np
from baselines.B5.metadata import encode_demographics
from baselines.B5_best import LEADS

def longest_run(flags):
    padded = np.r_[False, np.asarray(flags, bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def calibrated_record(path):
    import wfdb
    from scipy.signal import resample_poly
    head = wfdb.rdheader(str(path))
    lookup = {name.lower(): i for i, name in enumerate(head.sig_name)}
    indices = [lookup[name.lower()] for name in LEADS]
    if float(head.fs) != 1000:
        raise ValueError("PTB adapter expects documented 1000Hz; investigate other sources")
    record = wfdb.rdrecord(str(path), channels=indices, physical=False)
    digital = np.asarray(record.d_signal, np.float64).T
    gains = np.asarray(record.adc_gain, float)
    baselines = np.asarray(record.baseline, float)
    factors = {"mv": 1000., "uv": 1., "µv": 1., "μv": 1., "v": 1e6}
    units = [str(u or "mV").lower() for u in record.units]
    if any(u not in factors for u in units) or np.any(gains <= 0):
        raise ValueError("Unknown voltage calibration")
    wave = (digital - baselines[:, None]) / gains[:, None]
    wave *= np.asarray([factors[u] for u in units])[:, None]
    mask, reasons = np.ones(12, bool), []
    for i, name in enumerate(LEADS):
        r = []
        if not np.isfinite(wave[i]).all():
            r.append("nonfinite")
        if np.isfinite(wave[i]).all():
            if np.ptp(wave[i]) < 1.:
                r.append("flat_under_1uV")
            if np.max(np.abs(wave[i])) > 50000:
                r.append("absolute_over_50mV")
            if longest_run(np.diff(digital[i]) == 0) >= 5000:
                r.append("constant_5s")
        resolution = int(record.adc_res[i] or 16)
        if not 8 <= resolution <= 32:
            raise ValueError("Unknown ADC resolution")
        center = int(record.adc_zero[i] or 0)
        rails = (digital[i] <= center - 2 ** (resolution - 1)) | (digital[i] >= center + 2 ** (resolution - 1) - 1)
        if longest_run(rails) >= 20:
            r.append("ADC_rail_20ms")
        mask[i] = not r
        reasons.append(";".join(r))
    # Invalid targets become finite placeholders ONLY under a false supervision mask.
    # Invalid I is never used. Healthy target voltages are never centered or calibrated to Y.
    clean = wave.copy()
    clean[~mask] = 0
    converted = resample_poly(clean, up=1, down=2, axis=-1).astype(np.float32)
    metadata = {}
    for comment in head.comments:
        if ":" in comment:
            key, value = comment.split(":", 1)
            if key.strip().lower() in ("age", "sex", "height", "weight"):
                metadata[key.strip().lower()] = value.strip()
    return converted, mask, reasons, encode_demographics(metadata), head
