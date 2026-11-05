"""Nonce-bound child cleanup for baseline controllers, including orphaned sessions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time
import uuid

NONCE_KEY = 'TILEMEGA_DNN1_CONTROLLER_NONCE'


class OwnedController:
    def __init__(self, out, grace_s=3):
        self.out, self.grace_s = Path(out), grace_s
        self.nonce = str(uuid.uuid4())
        self.processes, self.handlers, self.cleaning = [], {}, False

    def __enter__(self):
        self.previous_nonce = os.environ.get(NONCE_KEY)
        os.environ[NONCE_KEY] = self.nonce
        def interrupted(number, _frame):
            if not self.cleaning:
                raise SystemExit(128 + number)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            self.handlers[number] = signal.signal(number, interrupted)
        return self

    def record(self, pid):
        if pid == os.getpid():
            return None
        proc = Path('/proc') / str(pid)
        try:
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z' or (f'{NONCE_KEY}={self.nonce}'.encode() not in
                    (proc / 'environ').read_bytes().split(b'\0')):
                return None
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            role = ('guard' if b'tilemega.bench.gpu_guard' in argv else
                    'scheduler' if b'tilemega.bench.scheduler' in argv else 'child')
            return dict(pid=pid, start=fields[19], ppid=int(fields[1]), role=role,
                        command=b' '.join(argv).decode(errors='replace'))
        except (OSError, IndexError):
            return None

    def scan(self):
        return [row for path in Path('/proc').iterdir() if path.name.isdigit()
                if (row := self.record(int(path.name))) is not None]

    def register(self, process):
        self.processes.append(process)

    def cancel_cpu_waiter(self, process, folder):
        """Retire one untouched worker; refuse any live payload or unknown child."""
        folder = Path(folder)
        root = self.record(process.pid)
        if root is None:
            return False
        signals, frozen = [], []
        def subtree():
            rows = self.scan()
            selected = {process.pid}
            while True:
                expanded = selected | {r['pid'] for r in rows if r['ppid'] in selected}
                if expanded == selected:
                    return [r for r in rows if r['pid'] in selected]
                selected = expanded
        def untouched():
            if list(folder.glob('raw/**/measurement*/command.json')):
                return False
            for path in folder.glob('raw/**/pgid'):
                try:
                    pgid = int(path.read_text())
                except (OSError, ValueError):
                    return False
                # Payloads start new sessions, so they need not remain in the tree.
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                        if fields[0] != 'Z' and int(fields[2]) == pgid:
                            return False
                    except (OSError, IndexError, ValueError):
                        continue
            return True
        initial = subtree()
        try:
            for row in initial:
                if row['role'] in ('scheduler', 'guard'):
                    self.send(row, signal.SIGSTOP, signals)
                    frozen.append(row)
            rows = subtree()
            safe = lambda row: (row['role'] in ('scheduler', 'guard') or
                Path(row['command'].split()[0]).name in ('bash', 'flock', 'nvidia-smi'))
            if not untouched() or not rows or not all(safe(row) for row in rows):
                return False
            for number in (signal.SIGTERM, signal.SIGKILL):
                for row in rows:
                    self.send(row, number, signals)
                    self.send(row, signal.SIGCONT, signals)
                deadline = time.monotonic() + self.grace_s
                while subtree() and time.monotonic() < deadline:
                    time.sleep(.05)
                rows = subtree()
                if not rows:
                    break
            if rows:
                raise RuntimeError('CPU waiter scope failed to drain')
            process.wait(timeout=1)
            result = dict(time=time.time(), root=root, captured=initial, signals=signals,
                remaining=[], all_owned_worker_drained=True,
                policy='no measurement command, live payload PGID, or unknown child; raw retained')
            (folder / 'cpu_waiter_retired.json').write_text(json.dumps(result, indent=2) + '\n')
            return True
        finally:
            for row in frozen:
                self.send(row, signal.SIGCONT, signals)

    def send(self, row, number, signals):
        current = self.record(row['pid'])
        if current and current['start'] == row['start']:
            try:
                os.kill(row['pid'], number)
                signals.append(dict(pid=row['pid'], start=row['start'], signal=number,
                                    role=row['role']))
            except ProcessLookupError:
                pass

    def wait_without(self, excluded):
        deadline = time.monotonic() + self.grace_s
        while True:
            rows = [row for row in self.scan() if row['role'] not in excluded]
            if not rows or time.monotonic() >= deadline:
                return rows
            time.sleep(.05)

    def cleanup(self):
        self.cleaning = True
        signals, initial = [], self.scan()
        # Freeze only verified schedulers and guards. Guards retain physical
        # locks until their separately sessioned payloads have exited.
        for row in initial:
            if row['role'] in ('scheduler', 'guard'):
                self.send(row, signal.SIGSTOP, signals)
        for number in (signal.SIGTERM, signal.SIGKILL):
            for row in self.scan():
                if row['role'] not in ('scheduler', 'guard'):
                    self.send(row, number, signals)
            if not self.wait_without({'scheduler', 'guard'}):
                break
        payloads = [row for row in self.scan() if row['role'] not in ('scheduler', 'guard')]
        if not payloads:
            for number in (signal.SIGTERM, signal.SIGKILL):
                for row in self.scan():
                    self.send(row, number, signals)
                    self.send(row, signal.SIGCONT, signals)
                if not self.wait_without(set()):
                    break
        remaining = self.scan()
        for process in self.processes:
            try:
                process.wait(timeout=.1)
            except Exception:
                pass
        result = dict(time=time.time(), controller_pid=os.getpid(), nonce=self.nonce,
            captured_owned_processes=initial, signals=signals, remaining=remaining,
            all_owned_nonce_drained=not remaining,
            guards_retained_stopped_for_undrained_payloads=bool(payloads),
            acceptance='unfinished cells invalidated; completed accepted evidence retained')
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'controller_cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
        if remaining:
            raise RuntimeError('owned controller scope did not drain; retained guards keep locks')

    def __exit__(self, kind, value, traceback):
        try:
            self.cleanup()
        finally:
            for number, handler in self.handlers.items():
                signal.signal(number, handler)
            if self.previous_nonce is None:
                os.environ.pop(NONCE_KEY, None)
            else:
                os.environ[NONCE_KEY] = self.previous_nonce
