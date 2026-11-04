#!/usr/bin/env python3
"""Run the unchanged GPU guard with DNN-owned signal cleanup.

The guard's separately sessioned payload otherwise survives a SIGTERM sent to
the guard itself. Acceptance, device locks and telemetry remain the original
guard's responsibility; an interrupted session is always rejected.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time


def run(guard_main=None):
    from tilemega.bench.gpu_guard import descendants, process_start, session_members
    if guard_main is None:
        from tilemega.bench.gpu_guard import main as guard_main
    args = sys.argv[1:]
    out = Path(args[args.index('--out') + 1])
    active = False
    original = {}

    def interrupted(number, _frame):
        nonlocal active
        if active:
            return
        active = True
        for signum in original:
            signal.signal(signum, signal.SIG_IGN)
        own = descendants(os.getpid()) | session_members(os.getpid())
        own.discard(os.getpid())
        starts = {pid: process_start(pid) for pid in own}
        record = dict(evidence='verified signal receipt; sender unknown',
            signal=number, guard_pid=os.getpid(), time=time.time(),
            captured_owned_pids=starts, acceptance='rejected',
            reason='DNN1 guard interrupted; discard complete cell session')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'dnn1_guard_interruption.json').write_text(json.dumps(record, indent=2) + '\n')
        # Keep the original guard's flock contexts alive until its owned
        # descendants have been signalled; never include an unrelated PID.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            for pid, start in starts.items():
                if start is not None and process_start(pid) == start:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
            if signum == signal.SIGTERM:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and any(
                        start is not None and process_start(pid) == start
                        for pid, start in starts.items()):
                    time.sleep(.05)
        (out / 'guard_result.json').write_text(json.dumps(dict(
            code=128 + number, reason=record['reason'], acceptance='rejected',
            ownership_verified=False), indent=2) + '\n')
        raise SystemExit(128 + number)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        original[number] = signal.signal(number, interrupted)
    os.environ['TILEMEGA_DNN1_GUARD_PID'] = str(os.getpid())
    os.environ['TILEMEGA_DNN1_GUARD_START'] = process_start(os.getpid())
    try:
        return guard_main()
    finally:
        for number, handler in original.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    raise SystemExit(run())
