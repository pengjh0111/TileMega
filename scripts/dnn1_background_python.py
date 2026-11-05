#!/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python
"""Versioned background admission for every scheduler retry and guard attempt."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import runpy
import sys

import dnn1_idle_floor_python as floor
from dnn1_background_cards import main_reservations
import dnn1_baselines as base


def admitted(row, original, policy, needs=0, nonce=None, *, scheduler=False):
    index = row['gpu_index']
    return (original(row, policy, needs, nonce) and not main_reservations(index) and
            (not scheduler or not base.held(f'/tmp/tilemega_gpu{index}.lock')))


def install(scheduler):
    floor.install_floor()
    import tilemega.bench.gpu_guard as guard
    original = guard.idle
    guard.idle = lambda row, policy, needs=0, nonce=None: admitted(
        row, original, policy, needs, nonce, scheduler=scheduler)


def nonblocking_guard(guard_main, out):
    original = fcntl.flock
    def acquire(fd, operation):
        return original(fd, operation | fcntl.LOCK_NB if operation == fcntl.LOCK_EX else operation)
    fcntl.flock = acquire
    try:
        return guard_main()
    except BlockingIOError:
        out.mkdir(parents=True, exist_ok=True)
        base.write(out / 'guard_result.json', dict(code=75,
            reason='preflight physical lock busy; background yields', acceptance='rejected',
            ownership_verified=False, power_monitoring=False))
        return 75
    finally:
        fcntl.flock = original


def main():
    args = sys.argv[1:]
    if args[:2] == ['-m', 'tilemega.bench.scheduler']:
        install(scheduler=True)
        sys.executable = str(Path(__file__).resolve())
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name='__main__', alter_sys=True)
    elif args[:2] == ['-m', 'tilemega.bench.gpu_guard']:
        from dnn1_guard_entry import run
        out = Path(args[args.index('--out') + 1])
        def guarded():
            install(scheduler=False)
            from tilemega.bench.gpu_guard import main as guard_main
            return nonblocking_guard(guard_main, out)
        sys.argv = [sys.argv[0], *args[2:]]
        result = run(guard_main=guarded)
        path = out / 'guard_result.json'
        if path.exists():
            receipt = base.read(path)
            receipt.update(idle_memory_floor_mib=1,
                background_provider_sha256=base.sha(__file__),
                idle_floor_provider_sha256=base.sha(floor.__file__))
            base.write(path, receipt)
        return result
    else:
        entry = str(Path(__file__).with_name('dnn1_resource_entry.py'))
        os.execv(floor.REAL_PYTHON, [floor.REAL_PYTHON, entry, *args])


if __name__ == '__main__':
    raise SystemExit(main())
