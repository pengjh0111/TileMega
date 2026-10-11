#!/usr/bin/env python3
"""Reap orphan compilers from explicitly named, completed local checkpoints."""
import argparse
import json
from pathlib import Path

from tilemega.bench.gpu_guard import process_start, session_members, stop


def reap(state, names):
    receipts = []
    for name in names:
        row = state[name]
        if not name.startswith('UT_') or row['status'] not in {'done', 'failed'}:
            raise ValueError('only completed UT checkpoints may be reaped: ' + name)
        pid = row['pid']
        if process_start(pid) is not None:
            raise ValueError('checkpoint PID still exists or has been reused: ' + name)
        members = session_members(pid)
        starts = {member: process_start(member) for member in members}
        starts[pid] = None
        # A session ID remains reserved while its orphan members survive.
        # The parent has exited; stop still verifies each member's birth time.
        class Completed:
            def __init__(self):
                self.pid = pid

            def wait(self, timeout=None):
                return row['exit_code']
        stop(Completed(), starts)
        receipts.append(dict(name=name, session=pid, members=starts))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--names', nargs='+', required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reap(json.loads(args.state.read_text()), args.names),
                                   indent=2) + '\n')
    print('Reaped only completed checkpoint sessions: ' + ', '.join(args.names))


if __name__ == '__main__':
    main()
