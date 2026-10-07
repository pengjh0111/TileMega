#!/usr/bin/env python3
"""Await scheduler completion markers via inotify, without status polling."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def await_completion(path):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(path.parent),
                                      0x00000008 | 0x00000080) < 0:
        raise OSError(ctypes.get_errno(), 'completion watch failed')
    try:
        while not path.exists():
            select.select([fd], [], [])
            os.read(fd, 65536)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--after-queue', type=Path,
                        help='wait for this completion event before starting')
    args = parser.parse_args()
    if args.after_queue:
        await_completion(args.after_queue)
    steps = {s['name'] for path in args.queue_dir.glob('queue_*.json')
             for s in json.loads(path.read_text())}
    if not steps:
        raise ValueError('empty queue')
    args.out.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(args.out), 0x00000008 | 0x00000080) < 0:
        raise OSError(ctypes.get_errno(), 'inotify setup failed')
    command = [sys.executable, str(Path(__file__).with_name('scheduler.py')),
               '--queue-dir', str(args.queue_dir), '--out', str(args.out),
               '--policy', str(args.policy)]
    seen = set()
    with (args.out / 'scheduler.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        pidfd = os.pidfd_open(process.pid)
        try:
            while True:
                readable, _, _ = select.select([fd, pidfd], [], [])
                if pidfd in readable:
                    raise RuntimeError('scheduler exited before queue completion')
                os.read(fd, 65536)
                path = args.out / 'state.json'
                state = json.loads(path.read_text()) if path.exists() else {}
                terminal = {n for n in steps if state.get(n, {}).get('status') in
                            ('done', 'failed', 'skipped', 'not_run')}
                for name in sorted(terminal - seen):
                    print(json.dumps(dict(event='queue_step_complete', name=name,
                                          status=state[name]['status'])), flush=True)
                seen.update(terminal)
                if terminal == steps:
                    result = dict(event='queue_complete', steps=len(steps),
                                  passed=all(state[n]['status'] == 'done' for n in steps))
                    (args.out / 'completion.json').write_text(json.dumps(result) + '\n')
                    print(json.dumps(result), flush=True)
                    return 0 if result['passed'] else 1
        finally:
            process.terminate()
            process.wait(timeout=10)
            os.close(pidfd)
            os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
