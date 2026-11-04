#!/usr/bin/env python3
"""Bound preflight lock waits while retaining the standard GPU guard."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


class LockTimeout(TimeoutError):
    pass


class BoundedLocks:
    def __init__(self, seconds):
        self.seconds = seconds

    def __getattr__(self, name):
        return getattr(fcntl, name)

    def flock(self, handle, operation):
        if operation != fcntl.LOCK_EX:
            return fcntl.flock(handle, operation)
        deadline = time.monotonic() + self.seconds
        while True:
            try:
                return fcntl.flock(handle, operation | fcntl.LOCK_NB)
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockTimeout(str(getattr(handle, 'name', handle)))
                time.sleep(min(.1, remaining))


def main(argv=None):
    from tilemega.bench import gpu_guard
    args = list(sys.argv[1:] if argv is None else argv)
    metadata = argparse.ArgumentParser(add_help=False)
    metadata.add_argument('--out', type=Path, required=True)
    parsed, _ = metadata.parse_known_args(args[:args.index('--')] if '--' in args else args)
    seconds = float(os.environ.get('TILEMEGA_PREFLIGHT_LOCK_TIMEOUT_S', '120'))
    if not 0 < seconds <= 600:
        raise ValueError('preflight lock timeout must be in (0, 600] seconds')
    parsed.out.mkdir(parents=True, exist_ok=True)
    if any((parsed.out / p).exists() for p in ('guard.jsonl', 'guard_result.json', 'pgid')):
        raise ValueError('guard attempt directory already contains execution evidence')
    (parsed.out / 'lock_policy.json').write_text(json.dumps(dict(
        timeout_s=seconds, launcher=str(Path(__file__).resolve()),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        guard_sha256=hashlib.sha256(Path(gpu_guard.__file__).read_bytes()).hexdigest()))+'\n')
    original = gpu_guard.fcntl
    original_argv = sys.argv
    sys.argv = [str(Path(__file__)), *args]
    gpu_guard.fcntl = BoundedLocks(seconds)
    try:
        return gpu_guard.main()
    except LockTimeout as error:
        # Locks precede telemetry and child creation; ExitStack releases any
        # earlier NUMA/group locks before this rejected attempt is recorded.
        if (parsed.out / 'guard.jsonl').exists() or (parsed.out / 'pgid').exists():
            raise RuntimeError('lock timeout occurred after preflight began') from error
        reason = 'preflight lock timeout: ' + str(error)
        (parsed.out / 'guard_result.json').write_text(json.dumps(dict(
            code=75, reason=reason, acceptance='rejected', ownership_verified=False,
            numa_concurrency_trial=os.environ.get('TILEMEGA_NUMA_CONCURRENCY_TRIAL')=='1'))+'\n')
        print(reason, flush=True)
        return 75
    finally:
        gpu_guard.fcntl = original
        sys.argv = original_argv


if __name__ == '__main__':
    raise SystemExit(main())
