#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare an isolated replay of accepted LLM jobs with their exact frozen APIs."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from collect_r16_candidates import load_driver, read_states
from r16_measurement_validation import completed_guard, file_sha, validate_llm
from summarize_r16 import record_directory


def prepare(root, queue_path, names, scheduler_roots, out):
    definitions = {job['name']:job for job in json.loads(queue_path.read_text())}
    state, cards = read_states(root, scheduler_roots)
    namespace = hashlib.sha256(str(out).encode()).hexdigest()[:12]
    jobs, provenance, files = [], [], {str(queue_path):file_sha(queue_path)}
    for index, name in enumerate(names):
        job = definitions[name]
        if not job.get('timing') or not job.get('gpu'):
            raise ValueError('replay requires an accepted timing job')
        directory = record_directory(root, name, state[name])
        guard = completed_guard(directory, state[name])
        paths = list(directory.glob('measurements/**/measurements.json'))
        if len(paths) != 1:
            raise ValueError('choose one static-batch accepted LLM job')
        measured, _, _ = validate_llm(paths[0], guard, root)
        if len(set(job.get('after', []))-set(state)):
            raise ValueError('original prerequisite state is absent')
        if any(state[dependency]['status'] != 'done' for dependency in job.get('after', [])):
            raise ValueError('original required correctness prerequisite did not pass')
        command = job['command']
        if '--expected-uuid' not in command or '--expected-pci' not in command:
            raise ValueError('frozen replay requires the original physical-device wrapper')
        if '--candidate' in command:
            candidate_path = Path(command[command.index('--candidate')+1])
            candidate = json.loads(candidate_path.read_text())
            load_driver(candidate).check_pair(candidate)
            files[str(candidate_path)] = file_sha(candidate_path)
        for argument in command:
            path = Path(argument)
            if path.is_absolute() and path.suffix in ('.py', '.json') and path.is_file():
                files[str(path)] = file_sha(path)
        if (int(job['env']['CUDA_VISIBLE_DEVICES']) != cards[name] or
                int(job['env']['TILEMEGA_DEVICE_INDEX']) != cards[name] or
                measured['gpu_index'] != cards[name]):
            raise ValueError('frozen command and accepted source card differ')
        clone = copy.deepcopy(job)
        clone.update(name=f'r16-replay-{namespace}-{index}', after=[], priority=index)
        clone.pop('after_any', None)
        jobs.append(clone)
        provenance.append(dict(original_job=name, replay_job=clone['name'],
            measurement=str(paths[0]), measurement_sha256=file_sha(paths[0]),
            previous_prerequisites=job.get('after', []), batch=measured['batch'],
            system=measured['system'], versions=measured['versions']))
    if len({job['env']['TILEMEGA_DEVICE_INDEX'] for job in jobs}) != 1:
        raise ValueError('one isolated replay queue uses one original physical card')
    if len(set(names)) != len(names):
        raise ValueError('duplicate requested source job')
    return jobs, provenance, files


def register_lease(root, jobs, out):
    # The persistent lease supervisor fails closed for this separate scheduler.
    # Register before launching so automatic cleanup cannot drop a live replay.
    uses_lease = False
    for job in jobs:
        command = job['command']
        uses_lease |= any('checkpoint-leased-v1' in argument for argument in command)
        if '--candidate' in command:
            candidate = json.loads(Path(command[command.index('--candidate')+1]).read_text())
            uses_lease |= bool(candidate.get('shared_checkpoint'))
    if not uses_lease:
        return
    directory = root/'runs/r16/moe-paired-external-v2/checkpoint-leased-v1'
    registry = root/'runs/r16/native-moe-shared-lease-v1/declared-jobs.json'
    with (directory/'declarations.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(registry.read_text())
        value['jobs'] = list(dict.fromkeys(value['jobs']+[job['name'] for job in jobs]))
        value.setdefault('isolated_replay_schedulers', []).append(str(out/'scheduler'))
        temporary = registry.with_name(registry.name+f'.{os.getpid()}.pending')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--job', action='append', required=True)
    parser.add_argument('--scheduler-root', type=Path, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    roots = args.scheduler_root or [root/'runs/r16/scheduler', root/'runs/r16/scheduler-rf2-final-v1']
    out = args.out.resolve()
    jobs, proof, files = prepare(root, args.queue.resolve(), args.job, roots, out)
    queue = out/'queue'
    queue.mkdir(parents=True, exist_ok=False)
    (queue/'queue_replay.json').write_text(json.dumps(jobs, indent=2)+'\n')
    command = [sys.executable, '-m', 'tilemega.bench.scheduler', '--queue-dir', str(queue),
        '--out', str(out/'scheduler'/('gpu'+jobs[0]['env']['TILEMEGA_DEVICE_INDEX'])),
        '--policy', str(root/'configs/bench/guard_policy.json')]
    receipt = dict(evidence='verified source receipts; replay prepared, not executed',
        command=command, sources=proof, frozen_files=files, remapped_devices=False,
        scope='same frozen binary/checkpoint/API and original UUID; independent replay, no new certification claim')
    (out/'reproduction.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    if args.run:
        if any(file_sha(path) != digest for path, digest in files.items()):
            raise ValueError('frozen replay input changed before launch')
        register_lease(root, jobs, out)
        env = dict(os.environ, **jobs[0]['env'])
        env['PYTHONPATH'] = str(root/'python')
        return subprocess.run(command, cwd=root, env=env).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
