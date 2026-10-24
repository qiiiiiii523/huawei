"""Move only compact long-window jobs to another GPU without changing model code.

The switch command drains P children of a STOPPED compact controller before
retiring that idle controller. P checkpoints/configs/caches are never rewritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.B7 import compact_suite, pipeline

AUDIT = 'long_gpu_migration.json'
LIMIT = 'long_gpu_concurrency.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def acquire_lock(path):
    # Same lock protocol as pipeline, with explicit cleanup on a busy lock.
    handle = Path(path).open('a+b')
    try:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if Path(path).stat().st_size == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BaseException:
        handle.close()
        raise


def setup(args):
    args.output = args.output.resolve()
    args.config_dir = ROOT / 'configs/b7_t_compact'
    args.experiment_suite = True
    args.retry_preparation = False
    args.gpu = args.gpu_p
    args.probe_first = False
    args.memory_budget_gib = 35.
    args.dry_run = False
    return args


def jobs_for(args, *, changed):
    jobs = compact_suite.suite_jobs(args)
    if changed:
        for job in jobs:
            config = read(job.arguments[job.arguments.index('--config') + 1])
            if config['sampling']['window_seconds'] == 40:
                job.gpu = args.gpu_long
    return jobs


def specification(args, jobs):
    return {'fingerprint': pipeline.fingerprint(args), 'raw_files_must_remain_unchanged': True,
            'experiments': [{'name': j.name, 'after': list(j.dependencies), 'arguments': j.arguments,
                'gpu': j.gpu, 'config': read(j.arguments[j.arguments.index('--config') + 1])} for j in jobs]}


def validate_change(before, after):
    if len(before['experiments']) != 8 or len(after['experiments']) != 8:
        raise ValueError('Expected the complete eight-stage compact suite')
    stripped = copy.deepcopy(after)
    for old, new, restored in zip(before['experiments'], after['experiments'], stripped['experiments']):
        if new['config']['sampling']['window_seconds'] == 10 and new['gpu'] != old['gpu']:
            raise ValueError('P GPU must not change')
        restored['gpu'] = old['gpu']
    if stripped != before:
        raise ValueError('Only long-window GPU assignment may change')


def verify_completed_P(args):
    for job in jobs_for(args, changed=False):
        if job.name.endswith(':P'):
            print(f'[verify] {job.name}: checking completion, checkpoint and data identity', flush=True)
            pipeline.validate_artifact(job)


def migrate(args):
    before = specification(args, jobs_for(args, changed=False))
    after = specification(args, jobs_for(args, changed=True))
    validate_change(before, after)
    locks = []
    try:
        for name in ('.compact_suite.lock', '.pipeline.lock'):
            locks.append(acquire_lock(args.output / name))
        current = read(args.output / 'pipeline_spec.json')
        if current == after and (args.output / AUDIT).is_file():
            verify_audit(args, after)
            if not (args.output / LIMIT).is_file():
                atomic_json(args.output / LIMIT, {'max_train': 1})
            return
        if current != before:
            raise ValueError('Original suite code/config/data paths/plan changed; migration refused')
        verify_completed_P(args)
        backup = args.output / f'pipeline_spec_before_long_gpu_{time.time_ns()}.json'
        atomic_json(backup, before)
        audit = {'before_sha256': digest(before), 'after_sha256': digest(after),
                 'backup': str(backup), 'gpu_p': args.gpu_p, 'gpu_long': args.gpu_long,
                 'training_code_changed': False, 'P_weights_retrained': False}
        # Write audit first: a crash before the final replacement leaves the old plan usable.
        atomic_json(args.output / AUDIT, audit)
        atomic_json(args.output / LIMIT, {'max_train': 1})
        atomic_json(args.output / 'pipeline_spec.json', after)
        print(f'[migrated] P GPU{args.gpu_p} unchanged; long jobs GPU{args.gpu_long}; concurrency=1', flush=True)
    finally:
        for lock in reversed(locks):
            lock.close()


def verify_audit(args, expected):
    audit = read(args.output / AUDIT)
    if (audit['after_sha256'] != digest(expected) or audit['gpu_p'] != args.gpu_p
            or audit['gpu_long'] != args.gpu_long or read(args.output / 'pipeline_spec.json') != expected):
        raise ValueError('GPU migration identity changed')
    before = read(audit['backup'])
    if digest(before) != audit['before_sha256']:
        raise ValueError('Original plan backup changed')
    validate_change(before, expected)


class LiveArgs:
    def __init__(self, args):
        self.__dict__.update(vars(args))

    @property
    def max_train(self):
        value = read(self.output / LIMIT)['max_train']
        if type(value) is not int or value not in (1, 2):
            raise ValueError('Concurrency must be integer 1 or 2')
        return value


def run(args):
    expected = specification(args, jobs_for(args, changed=True))
    verify_audit(args, expected)
    # Only this standalone process selects a different dispatch plan. No package file changes.
    original = compact_suite.suite_jobs
    compact_suite.suite_jobs = lambda value: jobs_for_original(value, original)
    try:
        return compact_suite.run(LiveArgs(args))
    finally:
        compact_suite.suite_jobs = original


def jobs_for_original(args, original):
    jobs = original(args)
    for job in jobs:
        cfg = read(job.arguments[job.arguments.index('--config') + 1])
        if cfg['sampling']['window_seconds'] == 40:
            job.gpu = args.gpu_long
    return jobs


def process(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'state': fields[0], 'start': fields[19], 'uid': root.stat().st_uid,
            'cwd': (root / 'cwd').resolve(),
            'command': [s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s]}


def active_children(pid):
    children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    active = []
    for child in children:
        try:
            item = process(int(child))
        except FileNotFoundError:
            continue
        if item['state'] not in ('Z', 'X'):
            active.append(int(child))
    return active


def check_controller(snapshot, args):
    command = snapshot['command']
    if (snapshot['uid'] != os.getuid() or snapshot['cwd'] != ROOT
            or snapshot['state'] not in ('T', 't')
            or '-m' not in command
            or command[command.index('-m') + 1] != 'baselines.B7.compact_suite'):
        raise ValueError('Expected your STOPPED original compact_suite controller in this worktree')
    output = Path(command[command.index('--output') + 1])
    if not output.is_absolute():
        output = snapshot['cwd'] / output
    if output.resolve() != args.output:
        raise ValueError('Controller output does not match requested suite')


def require_live_switch():
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Live switch requires the Linux server')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('pidfd support required for safe controller replacement')


def switch(args):
    require_live_switch()
    initial = process(args.controller_pid)
    check_controller(initial, args)
    # Prevent using this command to adopt already-launched long-window stages.
    for job in jobs_for(args, changed=False):
        if not job.name.endswith(':P') and job.artifact.parent.exists():
            raise ValueError(f'{job.name} already has a directory; do not use this P-only switch')
    expected = specification(args, jobs_for(args, changed=False))
    if read(args.output / 'pipeline_spec.json') != expected:
        raise ValueError('Original suite identity changed; controller left untouched')
    # pidfd pins the process identity across a potentially long wait.
    pidfd = os.pidfd_open(args.controller_pid)
    try:
        current = process(args.controller_pid)
        if current['start'] != initial['start']:
            raise ValueError('Controller PID was reused')
        while True:
            check_controller(process(args.controller_pid), args)
            done = all(job.artifact.is_file() for job in jobs_for(args, changed=False)
                       if job.name.endswith(':P'))
            children = active_children(args.controller_pid)
            if done and not children:
                break
            if not children and not done:
                raise RuntimeError('P children exited without completion; inspect logs, controller left paused')
            print(f'[waiting] P validation/training still active: {children}; controller remains paused', flush=True)
            time.sleep(15)
        verify_completed_P(args)
        check_controller(process(args.controller_pid), args)
        if active_children(args.controller_pid):
            raise RuntimeError('Controller still has active children; no signal sent')
        # Only the stopped idle dispatcher is retired. All P children have exited successfully.
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        for _ in range(100):
            try:
                if process(args.controller_pid)['state'] in ('Z', 'X'):
                    break
            except FileNotFoundError:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Old controller did not exit; migration not attempted')
    finally:
        os.close(pidfd)
    migrate(args)
    return run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('switch', 'migrate', 'run', 'limit'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/B7-T-compact-suite')
    parser.add_argument('--controller-pid', type=int)
    parser.add_argument('--gpu-p', default='1')
    parser.add_argument('--gpu-long', default='0')
    parser.add_argument('--value', type=int, choices=(1, 2))
    parser.add_argument('--data-root', type=Path, default=Path('/home/qht/huawei/data'))
    parser.add_argument('--metadata', type=Path, default=Path('/home/qht/huawei/metadata'))
    parser.add_argument('--ptb-root', type=Path, default=Path('/home/qht/huawei/data/PTB-Diagnostic'))
    parser.add_argument('--reuse-data-from', type=Path,
                        default=Path('/home/qht/huawei_B7_T/results/B7-T/auto_bs32_seed42'))
    args = setup(parser.parse_args())
    if args.action == 'limit':
        if args.value is None:
            parser.error('limit requires --value 1 or 2')
        verify_audit(args, specification(args, jobs_for(args, changed=True)))
        atomic_json(args.output / LIMIT, {'max_train': args.value})
        print(f'Concurrency set to {args.value}; running jobs are never killed', flush=True)
        return
    if args.action == 'switch' and (args.controller_pid is None or args.controller_pid <= 1):
        parser.error('switch requires the stopped original --controller-pid')
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    if args.action == 'switch':
        raise SystemExit(switch(args))
    if args.action == 'migrate':
        migrate(args)
        return
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
