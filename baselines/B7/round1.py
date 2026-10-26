"""C0-C4: random public10 P -> Huawei10 A. No PTBDB, Q, B or 40s stages."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from .checkpoint import file_sha256, file_hashes, load_checkpoint
from .config import read_json
from .data import GROUPS, checked_records, read_manifest
from .huawei10 import VIEW_PROTOCOL
from .presets import m7_config
from .public_qc import PUBLIC_QC_VERSION, PUBLIC_SOURCES
from .train import code_sha, validate_settings

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("C0", "C1", "C2", "C3", "C4")
LOSS_CHANGES = {"C0": (.1, 0., 0.), "C1": (.3, 0., 0.), "C2": (.1, .1, 0.),
                "C3": (.3, .1, 0.), "C4": (.1, 0., .1)}


def configurations(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown round-one variant")
    return {stage: read_json(ROOT / f"configs/round1/{variant}/{stage}.json") for stage in ("P", "A")}


def check_configs(variant, configs):
    expected = LOSS_CHANGES[variant]
    for name, cfg in configs.items():
        mc, lc, sc = validate_settings(cfg)
        if mc != m7_config("linear") or sc.window_seconds != 10 or sc.samples != 16 or sc.steps != 16 or sc.solver != "heun":
            raise ValueError("Round1 freezes M7-linear,500Hz/10s,Heun16/K16")
        if (lc.pcc_weight, lc.limb_wave_weight, lc.projection_weight) != expected:
            raise ValueError("Loss differs from the named C experiment")
        if lc.pcc_leads != 11 or lc.ii_weight != 5 or lc.mae_weight != 1 or lc.low_frequency_weight:
            raise ValueError("Round1 keeps FM7/MAE7-II5, PCC11 and no LF term")
        tr = cfg["training"]
        if name == "P":
            if (cfg["stage"], tr["updates"], tr["validate_every"], tr["batch_size"], tr["initial_validation"]) != ("public10", 20000, 10000, 64, False):
                raise ValueError("P must run20000, batch64, validation10000/20000")
        elif (cfg["stage"], tr["updates"], tr["validate_every"], tr["batch_size"], tr["initial_validation"]) != ("huawei10", 4000, 1000, 32, True):
            raise ValueError("A requires huawei10, max4000, batch32, validation0/every1000")


def preflight(public_path, huawei_path):
    public, huawei = read_manifest(public_path), read_manifest(huawei_path)
    if public.get("public_qc_version") != PUBLIC_QC_VERSION or {r["source"] for r in public["records"]} != PUBLIC_SOURCES:
        raise ValueError("Use corrected QC-v2 PTB-XL+Chapman only")
    if huawei.get("huawei_window_protocol") != VIEW_PROTOCOL:
        raise ValueError("Build the fixed Huawei10 index with baselines.B7.huawei10 first")
    if public["scales_uV"] != huawei["scales_uV"]:
        raise ValueError("Public/Huawei must share frozen scales")
    if huawei.get("public_scale_manifest_sha256") != file_sha256(public_path):
        raise ValueError("Huawei10 view was created against a different public manifest")
    if file_sha256(huawei["view_source_manifest"]) != huawei["view_source_sha256"]:
        raise ValueError("Huawei source index changed after view creation")
    for name, manifest in (("P", public), ("A", huawei)):
        target_identity = {}
        for split in ("train", "validation"):
            reliable = [r for r in manifest["records"] if r["split"] == split and r["input_reliable"]]
            if not reliable:
                raise ValueError(f"{name} missing reliable {split} data")
            if name == "A" and {r["group"] for r in reliable} != set(GROUPS):
                raise ValueError("Huawei10 requires all three paired groups in train and validation")
        for index, (row, (anchor, target, mask, _)) in enumerate(checked_records(manifest['records']), 1):
            if anchor.shape != (1, 5000) or target.shape != (12, 5000):
                raise ValueError("Both stages require actual full10s windows")
            if name == "A" and (row["source"] in PUBLIC_SOURCES or row["group"] not in GROUPS):
                raise ValueError("Non-Huawei records in A manifest")
            if name == "P" and (not row["input_reliable"] or not mask[0]):
                raise ValueError("QC-v2 public data contain invalid I")
            if name == "A":
                key = row['target_record_id']
                identity = (row['subject_id'], mask.tobytes(), hashlib.sha256(
                    np.where(mask[:, None], target, 0.).tobytes()).hexdigest())
                if key in target_identity and target_identity[key] != identity:
                    raise ValueError("Repeated Huawei target window has inconsistent waveform/QC")
                target_identity[key] = identity
            if index % 5000 == 0:
                print(f"[preflight] {name} {index}/{len(manifest['records'])}", flush=True)
    return public, huawei


@contextmanager
def lock_run(path):
    # OS-owned advisory lock: released on crash; persistent file avoids unlink races.
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0); handle.write(b"0"); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def verify_stage(folder, cfg, manifest_path, init=None):
    done = read_json(folder / "completed.json")
    best, last = load_checkpoint(folder / "best_r.pt"), load_checkpoint(folder / "last.pt")
    for ck in (best, last):
        if ck['identity']['config'] != cfg or ck['identity']['code_sha256'] != code_sha() or ck['identity']['manifest_sha256'] != file_sha256(manifest_path):
            raise ValueError("Completed stage identity changed")
        if init is not None and ck['initialization'].get('sha256') != file_sha256(init):
            raise ValueError("A no longer descends from the selected P EMA")
    if not done.get("completed") or done['update'] != last['update'] or last.get('pending_validation'):
        raise ValueError("Stage has no verified successful completion")
    if best['best_r'] != done['best_r'] or best['update'] != done['best_update']:
        raise ValueError("Best checkpoint and completion disagree")
    return {"best_r": done['best_r'], "best_update": best['update'], "updates": last['update'],
            "best_checkpoint": str(folder / 'best_r.pt'), "stopped_early": done.get('stopped_early', False)}


def run(args):
    configs = configurations(args.variant)
    check_configs(args.variant, configs)
    public_path, huawei_path = map(lambda p: Path(p).resolve(), (args.public_manifest, args.huawei_manifest))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with lock_run(output / ".round1.lock"):
        public, huawei = preflight(public_path, huawei_path)
        # All raw files are hashed, so a changed cache cannot silently reuse a done stage.
        raw = sorted({r['resolved_path'] for m in (public, huawei) for r in m['records']})
        specification = {"variant": args.variant, "configs": configs, "code_sha256": code_sha(),
                         "public_manifest": str(public_path), "huawei_manifest": str(huawei_path),
                         "manifest_sha256": [file_sha256(public_path), file_sha256(huawei_path)],
                         "raw_sha256": file_hashes(raw)}
        spec_path = output / 'specification.json'
        if spec_path.exists() and read_json(spec_path) != specification:
            raise ValueError("Data/code/config changed; use a new round1 output")
        write(spec_path, specification)
        if args.check_only:
            print('[CHECKED] No training launched', flush=True)
            return
        status = {"jobs": {"P": "pending", "A": "pending"}, "training_results": {}, "success": False}
        write(output / 'status.json', status)
        (output / 'logs').mkdir(exist_ok=True)
        for stage, manifest_path, gpu in (("P", public_path, args.gpu_p), ("A", huawei_path, args.gpu_a)):
            folder = output / stage
            init = output / 'P/best_r.pt' if stage == 'A' else None
            if not (folder / 'completed.json').exists():
                cfg_path = output / f'{stage}_config.json'
                write(cfg_path, configs[stage])
                command = [sys.executable, '-u', '-m', 'baselines.B7.train', '--manifest', str(manifest_path),
                           '--config', str(cfg_path), '--output', str(folder), '--device', args.device]
                if (folder / 'last.pt').exists():
                    command.append('--resume')
                elif folder.exists():
                    raise RuntimeError(f"Partial stage {folder} has no recoverable last.pt; inspect or use new output")
                elif init is not None:
                    command += ['--init-checkpoint', str(init)]
                environment = dict(os.environ)
                if gpu is not None:
                    environment['CUDA_VISIBLE_DEVICES'] = gpu
                status['jobs'][stage] = 'running'; write(output / 'status.json', status)
                print(f'[START] {args.variant}_{stage}; log={output / "logs" / (stage + ".log")}', flush=True)
                with (output / 'logs' / (stage + '.log')).open('ab') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        returncode = child.wait()
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.kill(); child.wait()
                        status['jobs'][stage] = 'interrupted'
                        if stage == 'P': status['jobs']['A'] = 'blocked'
                        write(output / 'status.json', status)
                        raise
                if returncode:
                    status['jobs'][stage] = 'failed'
                    if stage == 'P': status['jobs']['A'] = 'blocked'
                    write(output / 'status.json', status)
                    raise RuntimeError(f"{stage} exited {returncode}; inspect independent log before restarting")
            try:
                summary = verify_stage(folder, configs[stage], manifest_path, init)
            except Exception as error:
                status['jobs'][stage] = 'failed'
                status['error'] = str(error)
                if stage == 'P': status['jobs']['A'] = 'blocked'
                write(output / 'status.json', status)
                raise
            status['jobs'][stage] = 'done'; status['training_results'][stage] = summary
            write(output / 'status.json', status)
            print(f'[DONE] {args.variant}_{stage} best={summary["best_r"]:.6f}', flush=True)
        status['success'] = True; write(output / 'status.json', status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--public-manifest', required=True)
    p.add_argument('--huawei-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu-p', help='Physical GPU index or UUID; GPU UUID preferred')
    p.add_argument('--gpu-a', help='May differ from P; changing device does not change the experiment')
    p.add_argument('--device', default='cuda')
    p.add_argument('--check-only', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
