#!/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python
"""Baseline-only blocking physical-lock admission with unchanged strict gates."""
from pathlib import Path
import sys

import dnn1_background_python as background
import dnn1_settling_background_python as settling


def admitted(row, original, policy, needs=0, nonce=None, *, scheduler=False):
    return (original(row, policy, needs, nonce) and
            not background.main_reservations(row['gpu_index']))


def main():
    background.admitted = admitted
    background.nonblocking_guard = lambda guard_main, out: guard_main()
    # The scheduler must propagate this adapter, including after a retry.
    settling.__file__ = str(Path(__file__).resolve())
    result = settling.main()
    args = sys.argv
    if '--out' in args:
        path = Path(args[args.index('--out') + 1]) / 'guard_result.json'
        if path.exists():
            value = background.base.read(path)
            value.update(physical_lock_protocol='blocking-flock-baseline-FIFO-v1',
                         fifo_provider_sha256=settling.sha(__file__))
            background.base.write(path, value)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
