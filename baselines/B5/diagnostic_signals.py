"""Read-only signal diagnostics. These transforms never change official scores."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fresh_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Use a new/empty diagnostic output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator > 0 else None


def boxcar_width(seconds: float, fs: int = 500) -> int:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('slow-seconds must be positive and finite')
    width = max(3, int(round(seconds * fs)))
    return width if width % 2 else width + 1


def slow_component(x: np.ndarray, width: int) -> np.ndarray:
    """O(T) centered odd-width boxcar, with edge padding at RECORD boundaries."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite 1-D record with >=2 points')
    if width < 3 or width % 2 != 1 or width > len(x):
        raise ValueError('Boxcar width must be odd, >=3 and <= record length')
    padded = np.pad(x, (width // 2, width // 2), mode='edge')
    sums = np.concatenate(([0.], np.cumsum(padded, dtype=np.float64)))
    return (sums[width:] - sums[:-width]) / width


def ratio(a: float, b: float) -> float | None:
    return float(a / b) if b > 0 else None


def record_metrics(prediction: np.ndarray, target: np.ndarray, width: int) -> dict[str, Any]:
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Prediction and target must be matching finite 1-D signals')
    error = p - y
    raw_mse = float(np.mean(error ** 2))
    mean_error = float(error.mean())
    centered_mse = float(np.mean((error - mean_error) ** 2))
    ps, ys = slow_component(p, width), slow_component(y, width)
    pf, yf = p - ps, y - ys
    slow_error, fast_error = ps - ys, pf - yf
    slow_mse, fast_mse = float(np.mean(slow_error ** 2)), float(np.mean(fast_error ** 2))
    cross = float(2 * np.mean(slow_error * fast_error))
    if not np.isclose(raw_mse, mean_error ** 2 + centered_mse, rtol=1e-9, atol=1e-6):
        raise ArithmeticError('Offset decomposition failed')
    if not np.isclose(raw_mse, slow_mse + fast_mse + cross, rtol=1e-9, atol=1e-6):
        raise ArithmeticError('Slow/fast decomposition failed')
    return {
        'points': len(y), 'raw_r': pearson(p, y), 'raw_mse': raw_mse,
        'raw_rmse_uV': float(np.sqrt(raw_mse)),
        'prediction_mean_uV': float(p.mean()), 'target_mean_uV': float(y.mean()),
        'mean_error_uV': mean_error, 'offset_mse': mean_error ** 2,
        'offset_mse_fraction': ratio(mean_error ** 2, raw_mse),
        'centered_mse': centered_mse, 'centered_rmse_uV': float(np.sqrt(centered_mse)),
        'prediction_std_uV': float(p.std()), 'target_std_uV': float(y.std()),
        'std_ratio': ratio(float(p.std()), float(y.std())),
        'slow_r': pearson(ps, ys), 'fast_r': pearson(pf, yf),
        'slow_mse': slow_mse, 'fast_mse': fast_mse, 'slow_fast_error_cross_term': cross,
        'slow_rmse_uV': float(np.sqrt(slow_mse)), 'fast_rmse_uV': float(np.sqrt(fast_mse)),
        'slow_std_ratio': ratio(float(ps.std()), float(ys.std())),
        'fast_std_ratio': ratio(float(pf.std()), float(yf.std())),
        **boundary_metrics(p, y),
    }


def boundary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compare natural target transitions with predicted 10s-window transitions."""
    p,y=np.asarray(prediction,dtype=np.float64),np.asarray(target,dtype=np.float64)
    if p.shape!=y.shape or p.ndim!=1 or len(p)%5000:
        raise ValueError('Boundary diagnostics require complete 5000-point windows')
    boundaries=np.arange(5000,len(p),5000)
    if not len(boundaries):
        return {'boundary_count':0,'point_jump_error_mae_uV':None,'point_jump_error_max_uV':None,
                'prediction_point_jump_mae_uV':None,'target_point_jump_mae_uV':None,
                'window_mean_change_error_mse':None,'window_mean_change_error_rmse_uV':None,
                'boundary_100ms_each_side_error_jump_mae_uV':None}
    pj=p[boundaries]-p[boundaries-1]
    yj=y[boundaries]-y[boundaries-1]
    mean_error=p.reshape(-1,5000).mean(axis=1)-y.reshape(-1,5000).mean(axis=1)
    change=np.diff(mean_error)
    error=p-y
    neighborhood=[error[b:b+50].mean()-error[b-50:b].mean() for b in boundaries]
    return {'boundary_count':len(boundaries),
            'point_jump_error_mae_uV':float(np.abs(pj-yj).mean()),
            'point_jump_error_max_uV':float(np.abs(pj-yj).max()),
            'prediction_point_jump_mae_uV':float(np.abs(pj).mean()),
            'target_point_jump_mae_uV':float(np.abs(yj).mean()),
            'window_mean_change_error_mse':float(np.mean(change**2)),
            'window_mean_change_error_rmse_uV':float(np.sqrt(np.mean(change**2))),
            'boundary_100ms_each_side_error_jump_mae_uV':float(np.mean(np.abs(neighborhood)))}


def record_groups(metadata: list[dict[str, str]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(metadata):
        key = row.get('pair_id') or row.get('target_record_id')
        if not key:
            raise ValueError('Metadata needs pair_id or target_record_id')
        groups.setdefault(key, []).append(i)
    for key, indices in groups.items():
        indices.sort(key=lambda i: int(metadata[i]['start_sample_500hz']))
        starts = [int(metadata[i]['start_sample_500hz']) for i in indices]
        if starts != list(range(0, len(indices) * 5000, 5000)):
            raise ValueError(f'Incomplete, overlapping or duplicated windows: {key}')
        if len({metadata[i]['target_record_id'] for i in indices}) != 1:
            raise ValueError(f'Mixed target records: {key}')
        expected = {int(metadata[i]['expected_window_count']) for i in indices
                    if metadata[i].get('expected_window_count')}
        if len(expected) > 1 or (expected and expected != {len(indices)}):
            raise ValueError(f'Incomplete expected record: {key}')
    return groups


def aggregate_leads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for lead in LEADS[1:]:
        selected = [r for r in rows if r['lead'] == lead]
        points = sum(r['points'] for r in selected)
        if not points:
            continue
        weighted = lambda field: sum(r[field] * r['points'] for r in selected) / points
        macro = lambda field: [r[field] for r in selected if r[field] is not None]
        row = {'lead': lead, 'records': len(selected), 'points': points}
        for component in ('raw', 'slow', 'fast'):
            values = macro(component + '_r')
            row[component + '_r'] = float(np.mean(values)) if values else None
            row[component + '_undefined_records'] = len(selected) - len(values)
            row[component + '_rmse_uV'] = float(np.sqrt(weighted(component + '_mse')))
        row['centered_rmse_uV'] = float(np.sqrt(weighted('centered_mse')))
        row['offset_mse_fraction'] = ratio(weighted('offset_mse'), weighted('raw_mse'))
        row['slow_fast_error_cross_term'] = weighted('slow_fast_error_cross_term')
        for field in ('std_ratio', 'slow_std_ratio', 'fast_std_ratio'):
            values = macro(field)
            row['median_' + field] = float(np.median(values)) if values else None
        boundaries=sum(r['boundary_count'] for r in selected)
        row['boundary_count']=boundaries
        for field in ('point_jump_error_mae_uV','boundary_100ms_each_side_error_jump_mae_uV'):
            row[field]=sum(r[field]*r['boundary_count'] for r in selected if r['boundary_count'])/boundaries if boundaries else None
        mse=sum(r['window_mean_change_error_mse']*r['boundary_count'] for r in selected if r['boundary_count'])/boundaries if boundaries else None
        row['window_mean_change_error_rmse_uV']=float(np.sqrt(mse)) if mse is not None else None
        result.append(row)
    return result
