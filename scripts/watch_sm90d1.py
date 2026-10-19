#!/usr/bin/env python3
"""Observe terminal scheduler events, without polling queued GPU jobs."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import struct

from sm90d1_common import RUN, read, write


def status():
    result = {}
    for path in (RUN / 'scheduler').glob('gpu*/state.json'):
        records = read(path)
        result[path.parent.name] = {state:sum(r['status']==state for r in records.values())
            for state in ('done','failed','skipped','pending','running','cancelled')}
    if (RUN / 'gpu-budget.json').exists():
        result['gpu_hours_used'] = read(RUN / 'gpu-budget.json')['used_s']/3600
    write(RUN / 'progress.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    lib = ctypes.CDLL(None, use_errno=True)
    descriptor = lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    for folder in (RUN / 'scheduler').glob('gpu*'):
        if lib.inotify_add_watch(descriptor, os.fsencode(folder), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')
    print(json.dumps(status()), flush=True)
    while True:
        ready, _, _ = select.select([descriptor], [], [], 55 if a.once else None)
        if not ready:
            return
        events = os.read(descriptor, 1 << 20)
        offset, completed = 0, []
        while offset < len(events):
            _, _, _, length = struct.unpack_from('iIII', events, offset)
            name = events[offset+16:offset+16+length].split(b'\0',1)[0].decode()
            offset += 16+length
            if name.endswith(('.done','.failed','.skipped','.cancelled')):
                completed.append(name)
        if completed:
            print(json.dumps(dict(events=completed, status=status())), flush=True)
            if a.once:
                return


if __name__ == '__main__':
    main()
