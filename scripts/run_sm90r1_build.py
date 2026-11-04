#!/usr/bin/env python3
"""Run one CPU-only build with a shared lock and explicit resource limits."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def acquire_lock(paths):
    if len(paths)==1:
        handle=paths[0].open('a')
        try:fcntl.flock(handle,fcntl.LOCK_EX)
        except BaseException:
            handle.close();raise
        return handle
    while True:
        for path in paths:
            handle=path.open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close()
            except BaseException:
                handle.close();raise
            else:return handle
        time.sleep(.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--max-gib', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--lock', type=Path, default=Path('/tmp/tilemega_sm90r1_build.lock'))
    parser.add_argument('--lock-pools',type=Path,
        default=Path(__file__).resolve().parents[1]/'configs/bench/sm90r6_build_lock_pools.json')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.max_gib < 1 or args.timeout < 1:
        parser.error('a command and positive resource limits are required')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record = dict(command=command, max_address_space_gib=args.max_gib,
                  timeout_s=args.timeout, lock=str(args.lock), queued=time.time(),
                  status='queued', classification='compilation; not a numerical test')
    locks=[args.lock]
    if args.lock_pools.exists():
        pools=json.loads(args.lock_pools.read_text())
        if str(args.lock) in pools:
            locks=[Path(value) for value in pools[str(args.lock)]]
            if (not locks or args.lock not in locks or len(set(locks))!=len(locks)
                    or any(not value.is_absolute() for value in locks)):
                parser.error('a build lock pool requires distinct absolute paths including the requested lock')
            record.update(requested_lock=str(args.lock),lock_pool=list(map(str,locks)),
                lock_pool_config_sha256=hashlib.sha256(args.lock_pools.read_bytes()).hexdigest())
    record['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    process = None

    def save():
        temporary = args.receipt.with_suffix('.pending')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(args.receipt)

    save()

    def stop(signum, frame):
        del frame
        record.update(status='cancelled' if process else 'cancelled_before_start',
                      signal=signum)
        if process is None:
            record['ended'] = time.time()
            save()
            raise SystemExit(1)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with acquire_lock(locks) as lock:
        record['lock']=str(lock.name)
        record['started'] = time.time()

        def limits():
            bound = args.max_gib * 1024 ** 3
            resource.setrlimit(resource.RLIMIT_AS, (bound, bound))

        try:
            process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                       start_new_session=True, preexec_fn=limits)
        except OSError as error:
            record.update(status='build_spawn_failed', error=str(error), ended=time.time())
            save()
            return 1
        record.update(pid=process.pid, status='running')
        save()

        try:
            result = process.wait(timeout=args.timeout)
            if record['status'] != 'cancelled':
                record['status'] = 'completed' if result == 0 else 'build_failed'
        except subprocess.TimeoutExpired:
            record['status'] = 'build_timeout'
            os.killpg(process.pid, signal.SIGTERM)
            try:
                result = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                result = process.wait()
        record.update(exit_code=result, ended=time.time())
        save()
        return 0 if record['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
