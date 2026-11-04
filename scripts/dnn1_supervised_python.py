#!/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python
"""Preserve the baseline venv and route scheduler guards through DNN cleanup."""
from pathlib import Path
import os
import runpy
import sys


REAL_PYTHON = '/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'


def main():
    args = sys.argv[1:]
    if args[:2] == ['-m', 'tilemega.bench.scheduler']:
        # The unchanged scheduler constructs its guard argv from sys.executable.
        sys.executable = str(Path(__file__).resolve())
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name='__main__', alter_sys=True)
    elif args[:2] == ['-m', 'tilemega.bench.gpu_guard']:
        entry = str(Path(__file__).with_name('dnn1_guard_entry.py'))
        os.execv(REAL_PYTHON, [REAL_PYTHON, entry, *args[2:]])
    else:
        os.execv(REAL_PYTHON, [REAL_PYTHON, *args])


if __name__ == '__main__':
    main()
