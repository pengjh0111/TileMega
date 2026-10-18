#!/usr/bin/env python3
"""Run one CPU-only build with a shared lock and explicit resource limits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--max-gib', type=int, default=64)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--lock', type=Path, default=Path('/tmp/tilemega_sm90r1_build.lock'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.max_gib < 1 or args.timeout < 1:
        parser.error('a command and positive resource limits are required')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record = dict(command=command, max_address_space_gib=args.max_gib,
                  timeout_s=args.timeout, lock=str(args.lock), queued=time.time(),
                  status='queued', classification='compilation; not a numerical test')
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
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
