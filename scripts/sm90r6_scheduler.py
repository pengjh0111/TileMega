#!/usr/bin/env python3
"""Use bounded preflight lock waits with the unchanged queue scheduler."""
from pathlib import Path
import subprocess

from tilemega.bench import scheduler


class GuardLauncher:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, command, *args, **kwargs):
        if list(command[1:3]) == ['-m', 'tilemega.bench.gpu_guard']:
            command = [command[0], str(Path(__file__).with_name('sm90r6_guard.py')),
                       *command[3:]]
        return subprocess.Popen(command, *args, **kwargs)


if __name__ == '__main__':
    scheduler.subprocess = GuardLauncher()
    raise SystemExit(scheduler.main())
