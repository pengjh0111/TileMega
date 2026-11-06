"""Wait for an idle C550 group without holding its GPU locks."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import time

from . import gpu_guard


def sample(indices, policy, needs):
    rows, errors, locked = {}, {}, []
    with ExitStack() as handles:
        for index in sorted(indices, key=int):
            handle = handles.enter_context(Path(f'/tmp/tilemega_mx_gpu{index}.lock').open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked.append(str(index))
    for index in indices:
        try:
            rows[str(index)] = gpu_guard.gpu(policy, index)
        except Exception as error:
            errors[str(index)] = f'{type(error).__name__}: {error}'
    ready = (not locked and not errors and bool(rows) and
             all(gpu_guard.idle(row, policy, needs, index) for index, row in rows.items()))
    return dict(time=time.time(), ready=ready, locked=locked, rows=rows, errors=errors)


def wait(indices, policy, needs, log, timeout, interval):
    if not indices or len(set(indices)) != len(indices) or not set(map(int, indices)) <= set(range(8)):
        raise ValueError('unique physical GPUs 0 through 7 are required')
    if timeout < -1 or interval <= 0 or needs < 0:
        raise ValueError('invalid idle-wait limits')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started, last_notice = time.monotonic(), float('-inf')
    while True:
        row = sample(indices, policy, needs)
        with log.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if row['ready']:
            return row
        if time.monotonic() - last_notice >= 30:
            occupied = [index for index, device in row['rows'].items()
                        if not gpu_guard.idle(device, policy, needs, index)]
            print(f'Waiting for GPUs: occupied={occupied}, locked={row["locked"]}, '
                  f'telemetry_errors={list(row["errors"])}', flush=True)
            last_notice = time.monotonic()
        elapsed = time.monotonic() - started
        if timeout >= 0 and elapsed >= timeout:
            raise TimeoutError('GPU idle window not reached; no child was launched')
        time.sleep(interval if timeout < 0 else min(interval, timeout - elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--needs-free-mib', type=int, default=12288)
    parser.add_argument('--timeout-s', type=float, default=21600, help='-1 waits without a deadline')
    parser.add_argument('--interval-s', type=float, default=10)
    args = parser.parse_args()
    try:
        wait(args.group.split(','), json.loads(args.policy.read_text()), args.needs_free_mib,
             args.out, args.timeout_s, args.interval_s)
    except TimeoutError as error:
        print(str(error), flush=True)
        return 75
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
