"""Observe queue events and process health without controlling test execution."""
import argparse
from collections import Counter
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import time

from .scheduler import TERMINAL


ALERTS = {'failure', 'timing_paused', 'process_missing', 'timeout_overrun',
          'unreadable_state', 'completed'}


class Changes:
    """Directory watches survive the scheduler's atomic state-file replacement."""
    def __init__(self, folders):
        self.fd = -1
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.fd < 0:
                return
            for folder in set(map(Path, folders)):
                mask = 0x8 | 0x80 | 0x200 | 0x400 | 0x800
                if libc.inotify_add_watch(self.fd, os.fsencode(folder), mask) < 0:
                    self.close()
                    break
        except (AttributeError, OSError):
            self.close()

    def wait(self, seconds):
        ready, _, _ = select.select([self.fd] if self.fd >= 0 else [], [], [], seconds)
        if ready:
            try:
                while os.read(self.fd, 65536):
                    pass
            except BlockingIOError:
                pass

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def process_identity(pid):
    if not pid or int(pid) <= 0:
        return None
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def snapshot(out, queue_files):
    steps = {}
    for path in queue_files:
        for step in json.loads(Path(path).read_text()):
            if step['name'] in steps:
                raise ValueError('duplicate monitored step ' + step['name'])
            steps[step['name']] = step
    if not steps:
        raise ValueError('no monitored steps')
    state = json.loads((out / 'state.json').read_text())
    state = {name: state.get(name, dict(status='queued', attempts=0)) for name in steps}
    control_path = out / 'control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    pid = int((out / 'scheduler.pid').read_text())
    return dict(state=state, steps=steps, control=control, scheduler_pid=pid,
                counts=dict(Counter(row['status'] for row in state.values())))


class Observer:
    def __init__(self, previous=None, grace_s=120):
        self.previous = previous
        self.grace_s = grace_s
        self.absent = {}
        self.reported = set()
        self.identities = {}

    def inspect(self, value, now, alive=process_identity):
        state, previous, events = value['state'], self.previous, []
        if previous is None:
            events.append(dict(event='baseline', counts=value['counts'],
                running=[name for name, row in state.items() if row['status'] == 'running']))
        for name, row in state.items():
            old = previous.get('state', {}).get(name, {}) if previous else row
            signature = (row['status'], row.get('attempts', 0))
            if row['status'] == 'failed' and signature != (old.get('status'), old.get('attempts', 0)):
                events.append(dict(event='failure', step=name, exit_code=row.get('exit_code'),
                    attempts=row.get('attempts', 0), log=str(Path(value['out']) / (name + '.log')),
                    attempt_dir=row.get('out')))
        if (value['control'].get('timing_paused')
                and not (previous or {}).get('control', {}).get('timing_paused')):
            events.append(dict(event='timing_paused', reason=value['control'].get('reason')))
        complete = all(row['status'] in TERMINAL for row in state.values())
        if complete:
            if not previous or not all(row['status'] in TERMINAL for row in previous['state'].values()):
                events.append(dict(event='completed', counts=value['counts'],
                                   all_done=all(row['status'] == 'done' for row in state.values())))
        else:
            expected = [('scheduler', value['scheduler_pid'])]
            for name, row in state.items():
                if row['status'] != 'running':
                    continue
                expected.append((name, row.get('pid')))
                # Quiet logs or busy ptxas alone are no hang. Allow the GPU
                # guard's deadline plus the scheduler's additional grace.
                limit = value['steps'][name].get('timeout_s')
                grace = 600 if value['steps'][name].get('gpu') else self.grace_s
                if limit and row.get('started') and now > row['started'] + limit + grace:
                    key = ('timeout_overrun', name, row.get('attempts', 0))
                    if key not in self.reported:
                        self.reported.add(key)
                        events.append(dict(event=key[0], step=name, elapsed_s=now-row['started'],
                                           timeout_s=limit, pid=row.get('pid')))
            current = set(expected)
            self.absent = {key: since for key, since in self.absent.items() if key in current}
            for name, pid in expected:
                if not pid:
                    continue  # State may be persisted before Popen returns.
                key = (name, pid)
                identity = alive(pid)
                original = self.identities.setdefault(key, identity)
                if identity is not None and (original is None or identity == original):
                    self.identities[key] = identity
                    self.absent.pop(key, None)
                    continue
                since = self.absent.setdefault(key, now)
                if now - since >= self.grace_s and key not in self.reported:
                    self.reported.add(key)
                    events.append(dict(event='process_missing', step=name, pid=pid,
                                       missing_s=now-since))
        self.previous = value
        return events


def emit(path, event):
    event = dict(event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    line = json.dumps(event)
    with path.open('a') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def watch(args):
    args.events.parent.mkdir(parents=True, exist_ok=True)
    cursor = args.events.with_suffix('.cursor.json')
    singleton = args.events.with_suffix('.pid').open('a+')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    singleton.seek(0)
    singleton.truncate()
    singleton.write(str(os.getpid()) + '\n')
    singleton.flush()
    previous = json.loads(cursor.read_text()) if cursor.exists() else None
    observer = Observer(previous)
    changes = Changes([args.out, *(path.parent for path in args.queue_file)])
    error_since, error_reported = None, False
    try:
        while True:
            now = time.time()
            try:
                value = dict(snapshot(args.out, args.queue_file), out=str(args.out))
                events = observer.inspect(value, now)
                error_since, error_reported = None, False
                for event in events:
                    emit(args.events, event)
                temp = cursor.with_suffix('.tmp')
                temp.write_text(json.dumps(value) + '\n')
                temp.replace(cursor)
                if all(row['status'] in TERMINAL for row in value['state'].values()):
                    return
            except (OSError, ValueError) as error:
                error_since = now if error_since is None else error_since
                if now - error_since >= 120 and not error_reported:
                    emit(args.events, dict(event='unreadable_state', error=str(error)))
                    error_reported = True
            changes.wait(60)
    finally:
        changes.close()
        singleton.close()


def next_event(args):
    changes = Changes([args.events.parent])
    deadline, offset = time.monotonic() + args.timeout_s, args.offset
    try:
        while time.monotonic() < deadline:
            if args.events.exists():
                with args.events.open('rb') as stream:
                    stream.seek(offset)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line.endswith(b'\n'):
                            offset = start
                            break
                        offset = stream.tell()
                        event = json.loads(line)
                        if event['event'] in ALERTS:
                            print(json.dumps(dict(event, next_offset=offset)), flush=True)
                            return
            changes.wait(min(60, max(0, deadline-time.monotonic())))
    finally:
        changes.close()
    print(json.dumps(dict(event='wait_timeout', next_offset=offset)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    follow = sub.add_parser('watch')
    follow.add_argument('--out', type=Path, required=True)
    follow.add_argument('--queue-file', type=Path, action='append', required=True)
    follow.add_argument('--events', type=Path, required=True)
    once = sub.add_parser('next')
    once.add_argument('--events', type=Path, required=True)
    once.add_argument('--offset', type=int, default=0)
    once.add_argument('--timeout-s', type=float, default=21600)
    args = parser.parse_args()
    (watch if args.command == 'watch' else next_event)(args)


if __name__ == '__main__':
    main()
