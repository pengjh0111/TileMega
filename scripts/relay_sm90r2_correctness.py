#!/usr/bin/env python3
"""Publish CPU receipt checks when correctness on another GPU finishes."""
import argparse
import ctypes
import os
from pathlib import Path
import select

from sm90d1_common import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    manifest = read(a.manifest)
    state_path = Path(manifest['source_state'])
    lib = ctypes.CDLL(None, use_errno=True)
    fd = lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    try:
        if lib.inotify_add_watch(fd, os.fsencode(state_path.parent), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        pending = list(manifest['checks'])
        while pending:
            state = read(state_path)
            for item in pending[:]:
                status = state.get(item['source_job'], {}).get('status')
                if status not in ('done', 'failed', 'skipped', 'cancelled', 'not_run'):
                    continue
                # A failed source still runs the verifier, which rejects it and
                # prevents dependent timing. A completion event is not a pass.
                destination = Path(item['destination'])
                payload = [item['verifier']]
                if destination.exists() and read(destination) != payload:
                    raise ValueError('refusing to overwrite another queued job')
                write(destination, payload)
                print(item['source_job'], status, 'receipt check published', flush=True)
                pending.remove(item)
            if pending:
                select.select([fd], [], [])
                os.read(fd, 1 << 20)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
