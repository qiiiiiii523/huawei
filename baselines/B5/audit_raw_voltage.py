"""Audit Huawei XML calibration against cached raw targets; never rewrites data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np

from .diagnostic_signals import LEADS, fresh_directory, read_rows, write_rows


def unit_factor(unit: str) -> float:
    name = unit.strip().replace('µ', 'μ').lower()
    factors = {'uv': 1., 'μv': 1., 'mv': 1000., 'v': 1_000_000.}
    if name not in factors:
        raise ValueError(f'Unsupported XML voltage unit: {unit}')
    return factors[name]


def read_xml_independent(path: Path):
    """Independent digits*scale+origin reconstruction and calibration inventory."""
    root = ET.parse(path).getroot()
    local = lambda tag: tag.rsplit('}', 1)[-1]
    direct = lambda node, name: next((child for child in node if local(child.tag) == name), None)
    descendant = lambda node, name: next((child for child in node.iter() if local(child.tag) == name), None)
    codes = {'MDC_ECG_LEAD_' + lead.upper(): lead for lead in LEADS}
    candidates = []
    for sequence_set in (node for node in root.iter() if local(node.tag) == 'sequenceSet'):
        sequences = []
        for component in (child for child in sequence_set if local(child.tag) == 'component'):
            sequences.extend(child for child in component if local(child.tag) == 'sequence')
        if not sequences:
            sequences = [node for node in sequence_set.iter() if local(node.tag) == 'sequence']
        leads, fs = {}, None
        for sequence in sequences:
            code_node = direct(sequence, 'code')
            code = code_node.get('code', '') if code_node is not None else ''
            if code == 'TIME_ABSOLUTE':
                increment = descendant(sequence, 'increment')
                if increment is not None:
                    unit = increment.get('unit', 's').lower()
                    seconds = float(increment.get('value')) * ({'s': 1., 'ms': .001}.get(unit, float('nan')))
                    if not np.isfinite(seconds) or seconds <= 0:
                        raise ValueError('Unsupported or invalid XML time increment')
                    fs = 1. / seconds
            if code in codes:
                leads[codes[code]] = sequence
        if fs and set(LEADS).issubset(leads):
            digits = [descendant(leads[lead], 'digits') for lead in LEADS]
            length = min(len(node.text.split()) if node is not None and node.text else 0 for node in digits)
            if length:
                candidates.append((length, leads, fs))
    if not candidates:
        raise ValueError('No complete 12-lead XML sequence set')
    _, selected, fs = max(candidates, key=lambda item: item[0])
    signals, headers = [], []
    for lead in LEADS:
        value = direct(selected[lead], 'value')
        if value is None:
            raise ValueError('XML lead has no value')
        digits, scale, origin = (descendant(value, name) for name in ('digits', 'scale', 'origin'))
        if digits is None or not digits.text or scale is None:
            raise ValueError('XML lead has no digits/scale')
        values = np.fromstring(digits.text, sep=' ', dtype=np.float64)
        scale_value, scale_unit = float(scale.get('value')), scale.get('unit', 'uV')
        origin_value = float(origin.get('value', '0')) if origin is not None else 0.
        origin_unit = origin.get('unit', scale_unit) if origin is not None else scale_unit
        signal = values * scale_value * unit_factor(scale_unit) + origin_value * unit_factor(origin_unit)
        if not np.isfinite(signal).all():
            raise ValueError('Nonfinite XML voltage')
        signals.append(signal)
        headers.append({'lead': lead, 'digits_count': len(values), 'digits_min': float(values.min()),
                        'digits_max': float(values.max()), 'scale_value': scale_value, 'scale_unit': scale_unit,
                        'origin_value': origin_value, 'origin_unit': origin_unit,
                        'independent_mean_uV': float(signal.mean()), 'independent_std_uV': float(signal.std())})
    if len({len(x) for x in signals}) != 1:
        raise ValueError('Inconsistent XML lead lengths')
    return np.stack(signals), float(fs), headers


def resample_like_cache(signal: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import resample_poly
    signal = np.asarray(signal, dtype=np.float32)
    if abs(fs - 500) < .5:
        return signal
    rounded = int(round(fs))
    if rounded < 1 or abs(fs - rounded) > .5:
        raise ValueError('Unsupported source sampling rate')
    fraction = Fraction(500, rounded)
    return resample_poly(signal, fraction.numerator, fraction.denominator, axis=1).astype(np.float32)


def resolve_xml(raw_root: Path, stored: str) -> Path:
    # Handle Data/... manifests on Linux even when raw_root ends in lowercase data.
    parts = list(PurePosixPath(stored.replace('\\', '/')).parts)
    if any(part == '..' for part in parts):
        raise ValueError('Unsafe relative raw source path')
    markers = [i for i, part in enumerate(parts) if part.lower() == 'data']
    if markers:
        suffix = parts[markers[-1]+1:]
        candidates = [raw_root.joinpath(*suffix), raw_root.joinpath(*parts[markers[-1]:])]
    elif stored.startswith(('/', '\\')) or (parts and ':' in parts[0]):
        raise ValueError('Absolute source path requires a Data component or corrected manifest')
    else:
        candidates = [raw_root.joinpath(*parts)]
    candidates = [p.resolve() for p in candidates]
    if any(not p.is_relative_to(raw_root) for p in candidates):
        raise ValueError('Raw path escapes explicit root')
    found = [p for p in candidates if p.is_file()]
    if len(set(found)) > 1:
        raise ValueError('Ambiguous raw source; specify the actual Data folder as --raw-root')
    return found[0] if found else candidates[0]


def cached_records(config):
    from ecg12gen.contracts import SupervisionMode
    from ecg12gen.d12_pretrain import StrictD12PretrainDataset
    from ecg12gen.dataset import JointAnchorDataset
    from .data import common_config
    common = common_config(config)
    records = defaultdict(list)
    strict = StrictD12PretrainDataset(common, SupervisionMode.D12_I_PRETRAIN.value)
    for row in strict.rows:
        records[(row['target_record_id'], 'train')].append({
            'task': row['source_task_id'], 'split': 'train', 'array_index': int(row['source_array_index']),
            'start': int(row['start_sample_500hz']), 'array': strict.targets[row['source_task_id']],
            'subject_id': row['subject_id']})
    for task in ('task1', 'task2'):
        dataset = JointAnchorDataset(common, task, 'validation')
        for i in dataset._indices:
            row = dataset._rows[i]
            records[(row['target_record_id'], 'validation')].append({
                'task': task, 'split': 'validation', 'array_index': i,
                'start': int(row['start_sample_500hz']), 'array': dataset._targets,
                'subject_id': row['subject_id']})
    if any(len({split for record, split in records if record == rid}) > 1 for rid,_ in records):
        raise ValueError('Target record crosses train/validation split')
    return records


def cache_distributions(records):
    result = []
    for (record_id, split), refs in sorted(records.items()):
        unique = {}
        for ref in refs:
            value = np.asarray(ref['array'][ref['array_index']])
            if value.shape != (12, 5000) or not np.isfinite(value).all():
                raise ValueError('Invalid cached raw target')
            if ref['start'] in unique and not np.array_equal(value, unique[ref['start']]):
                raise ValueError('Different caches disagree for the same physical target window')
            unique[ref['start']] = value
        signal = np.concatenate([unique[start] for start in sorted(unique)], axis=1).astype(np.float64)
        for lead, name in enumerate(LEADS):
            result.append({'record_id': record_id, 'split': split, 'subject_id': refs[0]['subject_id'],
                           'lead': name, 'unique_windows': len(unique), 'points': signal.shape[1],
                           'mean_uV': float(signal[lead].mean()), 'std_uV': float(signal[lead].std()),
                           'min_uV': float(signal[lead].min()), 'max_uV': float(signal[lead].max()),
                           'coverage': 'available strict-train or validation cached windows; not necessarily full native record'})
    return result


def compare_record(record_id, refs, xml_path, atol, rtol):
    from ecg12gen.raw_task1 import parse_d12_xml
    independent, fs, headers = read_xml_independent(xml_path)
    parsed = parse_d12_xml(xml_path, LEADS)
    if parsed.signal_uV.shape != independent.shape or not np.isclose(fs, parsed.sampling_rate_hz):
        raise ValueError('Production and independent XML sequence selection differ')
    parser_ok = bool(np.allclose(parsed.signal_uV, independent, atol=atol, rtol=rtol))
    expected = resample_like_cache(parsed.signal_uV, parsed.sampling_rate_hz)
    comparisons = []
    for ref in refs:
        start = ref['start']
        wanted = expected[:, start:start+5000]
        actual = np.asarray(ref['array'][ref['array_index']], dtype=np.float64)
        if wanted.shape != actual.shape:
            raise ValueError('XML too short for cached window start/length')
        difference = actual - wanted.astype(np.float64)
        for lead, name in enumerate(LEADS):
            comparisons.append({'record_id': record_id, 'task': ref['task'], 'split': ref['split'],
                'array_index': ref['array_index'], 'start_sample_500hz': start, 'lead': name,
                'matches_reparsed_xml': bool(np.allclose(actual[lead], wanted[lead], atol=atol, rtol=rtol)),
                'max_abs_difference_uV': float(np.max(np.abs(difference[lead]))),
                'mean_difference_uV': float(difference[lead].mean()),
                'cached_mean_uV': float(actual[lead].mean()), 'reparsed_mean_uV': float(wanted[lead].mean())})
    for row in headers:
        row.update({'record_id': record_id, 'xml_path': str(xml_path), 'native_fs_hz': fs})
    return {'record_id': record_id, 'status': 'checked', 'xml_path': str(xml_path), 'native_fs_hz': fs,
            'independent_parser_matches_production': parser_ok,
            'max_parser_difference_uV': float(np.max(np.abs(parsed.signal_uV.astype(np.float64)-independent))),
            'cache_matches_reparsed_xml': all(r['matches_reparsed_xml'] for r in comparisons)}, headers, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--raw-root', type=Path, help='Actual folder containing 心电图机d12; defaults to config Data root')
    parser.add_argument('--record-id', action='append', help='Explicit d12 record IDs; otherwise hardest validation V3 mean offsets')
    parser.add_argument('--max-records', type=int, default=5)
    parser.add_argument('--atol-uV', type=float, default=.01)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.max_records < 1 or not np.isfinite([args.atol_uV, args.rtol]).all() or min(args.atol_uV,args.rtol)<0:
        parser.error('Invalid record count or tolerances')
    from .config import load_config, resolve_path
    config = load_config(args.config)
    raw_root = (args.raw_root or resolve_path(config, 'data_root')).resolve()
    output = fresh_directory(args.output_dir)
    records = cached_records(config)
    distributions = cache_distributions(records)
    write_rows(output / 'cached_target_record_statistics.csv', distributions)
    distribution_summary = []
    for split in ('train','validation'):
        for lead in LEADS:
            values = [r['mean_uV'] for r in distributions if r['split']==split and r['lead']==lead]
            if values:
                distribution_summary.append({'split':split,'lead':lead,'physical_records':len(values),
                    'mean_voltage_p05_uV':float(np.percentile(values,5)),
                    'mean_voltage_median_uV':float(np.median(values)),
                    'mean_voltage_p95_uV':float(np.percentile(values,95)),
                    'note':'descriptive only; never fit preprocessing or corrections from validation'})
    write_rows(output / 'cached_target_distribution_summary.csv', distribution_summary)
    if args.record_id:
        selected = list(dict.fromkeys(args.record_id))
    else:
        ranked = sorted((r for r in distributions if r['split']=='validation' and r['lead']=='V3'), key=lambda r:abs(r['mean_uV']), reverse=True)
        selected = [r['record_id'] for r in ranked[:args.max_records]]
    manifest_rows = read_rows(Path(config['repository_root']) / 'metadata/raw_record_manifest.csv')
    raw_lookup = {r['record_id']: r for r in manifest_rows}
    checks, headers, comparisons, required = [], [], [], []
    for record_id in selected:
        refs = [ref for (rid,_), values in records.items() if rid==record_id for ref in values]
        if not refs or record_id not in raw_lookup:
            checks.append({'record_id':record_id,'status':'error','reason':'Record is absent from eligible cache or raw manifest'})
            continue
        source = raw_lookup[record_id]['source_path']
        try:
            xml_path = resolve_xml(raw_root, source)
            required.append({'record_id':record_id,'source_path':source,'expected_xml_path':str(xml_path),'exists':xml_path.is_file()})
            if not xml_path.is_file():
                checks.append({'record_id':record_id,'status':'missing_raw','reason':'Upload raw XML; cache-only statistics do not verify XML calibration'})
                continue
            print('Checking XML/cache:', record_id, flush=True)
            result, h, c = compare_record(record_id,refs,xml_path,args.atol_uV,args.rtol)
            checks.append(result); headers.extend(h); comparisons.extend(c)
        except (OSError,ValueError,KeyError,ET.ParseError) as exc:
            checks.append({'record_id':record_id,'status':'error','reason':str(exc)})
    write_rows(output / 'required_raw_files.csv', required)
    write_rows(output / 'xml_calibration.csv', headers)
    write_rows(output / 'cache_comparison.csv', comparisons)
    complete = bool(checks) and all(r['status']=='checked' and r['independent_parser_matches_production'] and r['cache_matches_reparsed_xml'] for r in checks)
    report = {'training_started':False,'raw_data_modified':False,'checks_complete_and_matching':complete,
              'raw_root':str(raw_root),'atol_uV':args.atol_uV,'rtol':args.rtol,'records':checks,
              'scope':'Consistency with XML-declared scale/origin and current cache resampling. Does not certify device calibration or physiological validity.'}
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# B5 原始电压与缓存核对','','未训练、未修改数据。',f'全部选定记录核对完成且匹配：{complete}', '',
           '缺失原始 XML 不能算通过；待上传文件见 required_raw_files.csv。',
           'cached_target_distribution_summary.csv 只描述训练/验证分布，不用验证统计拟合scale。',
           'XML核对验证的是声明的单位/scale/origin与缓存一致，不证明设备校准本身正确。']
    for row in checks:
        lines.append(f"- {row['record_id']}: {row['status']}; {row.get('reason', '')}")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Done: {output / "audit.json"}; complete_and_matching={complete}',flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
