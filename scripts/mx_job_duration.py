#!/usr/bin/env python3
"""Record GPU guard duration independently of host wall-clock adjustments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def run(command, output):
    wall_start = time.time_ns()
    start = time.monotonic_ns()
    process = subprocess.run(command)
    end = time.monotonic_ns()
    wall_end = time.time_ns()
    record = dict(argv=command, returncode=process.returncode,
                  monotonic_start_ns=start, monotonic_end_ns=end,
                  elapsed_seconds=(end-start)/1e9,
                  wall_start_ns=wall_start, wall_end_ns=wall_end,
                  wall_clock_discontinuity=abs((wall_end-wall_start)-(end-start)) > 1_000_000_000,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with Path(output).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a guard command is required')
    return run(command, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
