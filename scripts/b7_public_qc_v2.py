"""Audit public ECG or prepare fresh QC-v2 caches without starting training."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from baselines.B7.data import load_record, read_manifest
from baselines.B7.prepare import prepare, public_index, read_raw
from baselines.B7.public_qc import PUBLIC_QC_VERSION, deduplicate, inspect_public


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def rebase_long_manifest(source, public, destination):
    """Raw-uV arrays do not depend on fitted scales: reuse files, replace manifest."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    current = read_manifest(source)
    short = read_manifest(public)
    if short.get('public_qc_version') != PUBLIC_QC_VERSION:
        raise ValueError('Prepare corrected public10 first')
    if any(r['source'] not in {'ptbdb', 'huawei'} and r['group'] not in {'task1', 'machine', 'body'}
           for r in current['records']):
        raise ValueError('Only PTBDB/Huawei raw-uV caches may be rebased; public10 must be rebuilt')
    manifest = copy.deepcopy(current)
    for row in manifest['records']:
        load_record(row)  # Verify existing raw-uV arrays and masks; never modify them.
        row['path'] = row.pop('resolved_path')
    manifest.update(scales_uV=short['scales_uV'], scale_source=str(Path(public).resolve()),
                    public_qc_version=PUBLIC_QC_VERSION,
                    reused_raw_manifest=str(source),
                    reused_raw_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    long_waveform_qc_changed=False)
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != manifest:
            raise ValueError(f'Existing rebased manifest differs: {destination}; choose a new output')
    else:
        write_json(destination, manifest)
    read_manifest(destination)
    print(f'[rebase] {source} -> {destination}; raw-uV files unchanged', flush=True)


def audit(data_root, output):
    if output.exists():
        raise FileExistsError('Use a new audit output')
    rows = public_index('public10', data_root)
    def inspect(row):
        anchor, target, fs, _ = read_raw(Path(row['path']))
        mask, ok, quality = inspect_public(anchor, target, np.isfinite(target).all(1), fs)
        return {**row, 'input_reliable': ok, 'lead_mask': mask.tolist(), 'public_qc': quality}
    checked = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(inspect, rows), 1):
            checked.append(row)
            if i % 1000 == 0:
                print(f'[audit] {i}/{len(rows)}', flush=True)
    retained, duplicates = deduplicate(checked)
    excluded = [r['record_id'] for r in retained if not r['input_reliable'] or
                not any(r['lead_mask'][j] for j in (1,6,7,8,9,10,11))]
    excluded_set = set(excluded)
    retained = [r for r in retained if r['record_id'] not in excluded_set]
    counts = Counter((r['source'], r['split']) for r in retained)
    summary = {'version': PUBLIC_QC_VERSION, 'input_records': len(rows),
               'retained_records': len(retained), 'duplicate_groups': len(duplicates),
               'duplicate_copies_removed': sum(len(g['members']) - 1 for g in duplicates),
               'excluded_unusable': excluded,
               'retained_by_source_split': {f'{k[0]}:{k[1]}': v for k,v in sorted(counts.items())},
               'invalid_I_records': [r['record_id'] for r in checked if not r['input_reliable']],
               'flag_counts': dict(Counter(flag for r in checked for flag in r['public_qc']['flags']))}
    write_json(output, {'summary': summary, 'duplicate_groups': duplicates, 'records': checked})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true', help='Write JSON only; no training/cache arrays')
    parser.add_argument('--reuse-long-from', type=Path, help='Old pipeline root containing PTBDB/Huawei raw-uV caches')
    parser.add_argument('--resume', action='store_true', help='Verify/reuse only a matching QC-v2 partial preparation')
    args = parser.parse_args()
    if args.audit_only:
        return audit(args.data_root, args.output)
    public = args.output.resolve() / 'data/public10/manifest.json'
    if public.exists():
        if not args.resume or read_manifest(public).get('public_qc_version') != PUBLIC_QC_VERSION:
            raise FileExistsError('Use a new data output, or --resume for an existing QC-v2 output')
    else:
        prepare(Namespace(output=str(public.parent), index=None, public_source='public10',
                          root=str(args.data_root), scale_manifest=None,
                          resume=args.resume and public.parent.exists()))
    if args.reuse_long_from:
        from baselines.B7.pipeline import reused_manifests
        for name, source in reused_manifests(args.reuse_long_from).items():
            if name != 'public10':
                rebase_long_manifest(source, public, args.output.resolve()/f'data/{name}/manifest.json')
    print(f'[ready] corrected data root: {args.output.resolve()}; no training started', flush=True)


if __name__ == '__main__':
    main()
