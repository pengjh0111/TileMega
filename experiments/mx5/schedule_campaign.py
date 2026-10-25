#!/usr/bin/env python3
"""Run immutable campaign batches on GPUs 7 and 6, draining on first failure.

Each invocation needs a new --out directory. --start/--stop use Python's
inclusive/exclusive job-index bounds; --jobs and --kinds further filter them.
--cards accepts 7, 6, or 7 6 (the default), with one active job per card.
Completed batches require all three matching rounds and a successful guard.
Partial batches are preserved and require a newly prepared campaign to retry.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CARDS = (7, 6)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def batch_state(job):
    """Only the original keys, successful processes and guard admit a skip."""
    keys_path = Path(job['keys'])
    if digest(keys_path) != job['keys_sha256']:
        raise ValueError('campaign keys changed: ' + str(keys_path))
    keys = json.loads(keys_path.read_text())
    if not keys or len(keys) != job['rows']:
        raise ValueError('campaign key count changed: ' + str(keys_path))
    directory = Path(job['directory'])
    artifacts = (list(directory.glob('round*')) + list(directory.glob('process*'))
                 + list(directory.glob('scheduler_job.json')))
    guard = directory / 'guard'
    if not artifacts and not guard.exists():
        return 'pending'
    try:
        if json.loads((guard / 'guard_result.json').read_text())['code'] != 0:
            raise ValueError('guard failed')
        for round_number in (1, 2, 3):
            output = directory / f'round{round_number}.jsonl'
            output.with_suffix('.log').read_bytes()
            process = json.loads((directory / f'process{round_number}.json').read_text())
            if process['returncode'] != 0 or process['round'] != round_number:
                raise ValueError('process failed')
            command = process['command']
            for flag, value in (('--keys', str(keys_path)), ('--out', str(output)),
                                ('--round', str(round_number))):
                if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
                    raise ValueError('process identity mismatch')
            rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
            if [row['key'] for row in rows] != keys or any(row['round'] != round_number for row in rows):
                raise ValueError('round key identity mismatch')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ValueError('partial or unsuccessful batch; preserve and prepare a new campaign: '
                         + str(directory)) from error
    return 'completed'


def select_jobs(document, args):
    jobs = document['jobs']
    stop = len(jobs) if args.stop is None else args.stop
    if args.start < 0 or stop < args.start or stop > len(jobs):
        raise ValueError('invalid start/stop job bounds')
    if args.jobs is not None and any(index < 0 or index >= len(jobs) for index in args.jobs):
        raise ValueError('job index outside campaign')
    selected = [(index, job) for index, job in enumerate(jobs)
                if args.start <= index < stop
                and (args.jobs is None or index in args.jobs)
                and (args.kinds is None or job['kind'] in args.kinds)]
    if not selected:
        raise ValueError('no selected campaign jobs')
    directories = [str(Path(job['directory']).resolve()) for _, job in selected]
    if len(set(directories)) != len(directories):
        raise ValueError('duplicate campaign job directories')
    return selected


def command_for(args, index, job, card):
    return [str(ROOT / 'scripts/mx_gpu_job.sh'), str(card), '900',
            str(Path(job['directory']) / 'guard'), sys.executable,
            str(ROOT / 'experiments/mx5/campaign.py'), 'run',
            '--manifest', str(args.manifest), '--job', str(index),
            '--iterations', str(args.iterations)]


def selected_cards(args):
    cards = tuple(getattr(args, 'cards', CARDS))
    if cards not in ((7,), (6,), CARDS):
        raise ValueError('--cards must be 7, 6, or 7 6, without duplicates')
    return cards


def execute(args, selected, manifest_hash, record):
    cards = selected_cards(args)
    pending = []
    for index, job in selected:
        state = batch_state(job)
        record('preflight', job=index, status=state)
        if state == 'pending':
            pending.append((index, job))
    active = {}
    failed = False
    interrupted = []

    def stop_launching(signum, frame):
        interrupted.append(signum)

    handlers = {sig: signal.signal(sig, stop_launching) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = dict(os.environ, NEEDS_MIB='20480',
               TILEMEGA_GPU_JOB_LOG=str(ROOT / 'runs/mx5/gpu_jobs.log'))
    try:
        while pending or active:
            # Observe every finished guard before assigning any newly free card.
            for card, (index, job, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[card]
                error = None
                try:
                    if code != 0:
                        raise ValueError(f'guard exited {code}')
                    if digest(args.manifest) != manifest_hash:
                        raise ValueError('campaign manifest changed')
                    if batch_state(job) != 'completed':
                        raise ValueError('guard returned without completed rounds')
                except (OSError, ValueError) as exception:
                    failed = True
                    error = str(exception)
                record('finish', job=index, card=card, returncode=code,
                       status='failed' if error else 'completed', error=error)
            if interrupted or failed:
                if not active:
                    break
            else:
                for card in cards:
                    if interrupted:
                        break
                    if card in active or not pending:
                        continue
                    index, job = pending.pop(0)
                    command = command_for(args, index, job, card)
                    log_path = args.out / f'job{index:04d}.log'
                    log = None
                    try:
                        if digest(args.manifest) != manifest_hash:
                            raise ValueError('campaign manifest changed')
                        if batch_state(job) != 'pending':
                            raise ValueError('batch changed since preflight')
                        launch = dict(job=index, card=card, command=command,
                                      manifest_sha256=manifest_hash,
                                      keys_sha256=job['keys_sha256'], log=str(log_path),
                                      environment={key: env[key] for key in
                                                   ('NEEDS_MIB', 'TILEMEGA_GPU_JOB_LOG')})
                        write_new(Path(job['directory']) / 'scheduler_job.json', launch)
                        log = log_path.open('x')
                        record('launch', **launch)
                        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        active[card] = (index, job, process, log)
                    except (OSError, ValueError) as error:
                        if log is not None:
                            log.close()
                        record('finish', job=index, card=card, status='failed',
                               returncode=None, error=str(error), command=command)
                        failed = True
                        break
            if active:
                time.sleep(0.2)
        for index, _ in pending:
            record('not_launched', job=index, status='pending')
        return 128 + interrupted[0] if interrupted else int(failed)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def schedule(args):
    cards = selected_cards(args)
    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()
    document = json.loads(args.manifest.read_text())
    manifest_hash = digest(args.manifest)
    selected = select_jobs(document, args)
    if document.get('processes_per_key') != 3:
        raise ValueError('campaign must specify three processes per key')
    with (args.manifest.parent / '.scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another scheduler holds this campaign') from error
        args.out.mkdir(parents=True, exist_ok=False)
        write_new(args.out / 'scheduler.json', dict(
            manifest=str(args.manifest), manifest_sha256=manifest_hash,
            selected=[index for index, _ in selected], cards=cards,
            iterations=args.iterations, timeout_seconds=900, needs_mib=20480,
            pid=os.getpid(), argv=sys.argv))
        with (args.out / 'events.jsonl').open('x') as events:
            def record(event, **values):
                row = dict(event=event, time_ns=time.time_ns(), **values)
                events.write(json.dumps(row, sort_keys=True) + '\n')
                events.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
            try:
                code = execute(args, selected, manifest_hash, record)
            except (OSError, ValueError) as error:
                record('error', error=str(error))
                code = 1
            write_new(args.out / 'result.json', dict(returncode=code))
            return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, nargs='+')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--kinds', nargs='+', choices=('gemm', 'head', 'attention'))
    parser.add_argument('--cards', type=int, nargs='+', choices=CARDS, default=list(CARDS))
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error('iterations must be at least three')
    try:
        return schedule(args)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
