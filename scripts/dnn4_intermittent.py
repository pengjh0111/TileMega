#!/usr/bin/env python3
"""Dispatch one guarded GPU0 process per reservation with idle gaps."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(os.environ.get('TILEMEGA_DNN4_REPO', '/root/TileMega-dnn3'))
ENTRY = Path('/root/python')
RUN = subprocess.run
GAP = 30
SESSION = Path(os.environ.get('TILEMEGA_DNN4_INTERMITTENT_SESSION',
    ROOT/'runs/dnn4/bert_priority/resume_gpu0_intermittent_20261111_v1'))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def translate(command):
    command = [str(x) for x in command]
    original = str(ROOT/'scripts/dnn4_guard_python.py')
    return [str(ENTRY/'guard.py') if value == original else value for value in command]


def dispatch(command, **kwargs):
    SESSION.mkdir(parents=True, exist_ok=True)
    # Serialize controllers as well as guarded payloads, so the gap spans protocols.
    with (SESSION/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        authorization = read(SESSION/'authorization.json')
        if authorization['status'] != 'active' or authorization['allowed_gpu_indices'] != [0]:
            raise ValueError('intermittent continuation is not active for GPU0')
        return dispatch_locked(command, **kwargs)


def dispatch_locked(command, **kwargs):
    """Keep original protocol order; combine only real per-job receipts."""
    command = [str(x) for x in command]
    assert command[1:3] == ['-m', 'tilemega.bench.scheduler']
    queue = Path(command[command.index('--queue-dir')+1])
    out = Path(command[command.index('--out')+1])
    steps = [s for path in sorted(queue.glob('queue_*.json')) for s in read(path)]
    if not steps or len({s['name'] for s in steps}) != len(steps):
        raise ValueError('empty or duplicate intermittent plan')
    out.mkdir(parents=True, exist_ok=True)
    states = read(out/'state.json') if (out/'state.json').exists() else {}
    if any(s.get('status') == 'running' for s in states.values()):
        raise ValueError('inspect interrupted descendants before recovering this protocol')
    inputs = {str(path): sha(path) for path in sorted(queue.glob('queue_*.json'))}
    frozen = out/'intermittent_inputs.json'
    if frozen.exists() and read(frozen) != inputs:
        raise ValueError('intermittent queue changed after its first dispatch')
    save(frozen, inputs)
    SESSION.mkdir(parents=True, exist_ok=True)
    sources = []
    for index, step in enumerate(steps):
        if read(SESSION/'authorization.json')['status'] != 'active':
            raise ValueError('user paused intermittent continuation')
        env = step.get('env', {})
        if (not step.get('gpu') or env.get('CUDA_VISIBLE_DEVICES') != '0'
                or env.get('TILEMEGA_DEVICE_INDEX') != '0'):
            raise ValueError('new hardware work must use GPU0')
        if any(states.get(dep, {}).get('status') != 'done' for dep in step.get('after', [])):
            raise ValueError('intermittent protocol dependency is not complete')
        payload = translate(step['command'])
        python_files = [x for x in payload if x.endswith('.py')]
        if any(not Path(x).is_relative_to(ENTRY) for x in python_files):
            raise ValueError('Python worker entry must be relocated under /root/python')
        prior = states.get(step['name'], {})
        if prior:
            source = Path(prior.get('source_state',
                str(Path(prior['out']).parents[2]/'scheduler/state.json')))
            actual = read(source)[step['name']]
            old_step = read(source.parent.parent/'queue/queue_single.json')
            if (actual['out'] != prior['out'] or actual['status'] != prior['status']
                    or len(old_step) != 1 or old_step[0]['command'] != payload
                    or old_step[0]['env'] != env):
                raise ValueError('partial receipt differs from its original individual job')
            old_guard = read(Path(prior['out'])/'guard_result.json')
            if prior['status'] == 'done' and old_guard['code'] == 0:
                sources.append(str(source))
                continue
            if prior.get('exit_code') != 75 or old_guard['code'] != 75:
                raise ValueError('only guard-deferred work may obtain a fresh reservation')
        while True:
            previous = SESSION/'last_release.json'
            release = read(previous) if previous.exists() else {}
            not_before = max(release.get('released_at', 0)+GAP,
                             release.get('not_before', 0), prior.get('retry_at', 0))
            while time.time() < not_before:
                if read(SESSION/'authorization.json')['status'] != 'active':
                    raise ValueError('user paused intermittent continuation')
                remaining = min(30, not_before-time.time())
                print(json.dumps(dict(status='idle_gap', seconds=remaining,
                                      reservation_held=False)), flush=True)
                time.sleep(remaining)
            job = out/'jobs'/f'{index+1:02d}'
            retry = 0
            while job.exists():
                retry += 1
                job = out/'jobs'/f'{index+1:02d}-reservation{retry+1}'
            one = dict(step, command=payload, after=[])
            save(job/'queue/queue_single.json', [one])
            launch = translate(command)
            launch[launch.index('--queue-dir')+1] = str(job/'queue')
            launch[launch.index('--out')+1] = str(job/'scheduler')
            launch[launch.index('--deadline-hours')+1] = '.1'
            print(json.dumps(dict(status='dispatch_one', name=step['name'], gpu=0,
                                  command=payload)), flush=True)
            done = RUN(launch, **kwargs)
            source = job/'scheduler/state.json'
            state = read(source).get(step['name'], {})
            state['source_state'] = str(source)
            save(previous, dict(released_at=time.time(), job=str(job),
                 status=state.get('status'), not_before=state.get('retry_at', 0)))
            states[step['name']] = state
            save(out/'state.json', states)
            receipt = dict(time=time.time(), name=step['name'], gpu=0, job=str(job),
                           state=state, source_state=str(source))
            with (SESSION/'jobs.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt)+'\n')
            guard = read(Path(state['out'])/'guard_result.json')
            if not done.returncode and state.get('status') == 'done' and guard['code'] == 0:
                sources.append(str(source))
                break
            if state.get('exit_code') != 75 or guard['code'] != 75:
                raise RuntimeError('single guarded process did not complete: '+step['name'])
            prior = state
    save(out/'intermittent.json', dict(evidence='verified', gpu=0, minimum_gap_s=GAP,
         jobs=len(steps), source_states=sources,
         scope='derived index of original individual scheduler receipts; no grouped reservations'))
    return subprocess.CompletedProcess(command, 0)


def install():
    def run(command, *args, **kwargs):
        words = [str(x) for x in command] if isinstance(command, (list, tuple)) else []
        if words[1:3] == ['-m', 'tilemega.bench.scheduler']:
            if args:
                raise ValueError('unexpected positional subprocess arguments')
            return dispatch(words, **kwargs)
        return RUN(command, *args, **kwargs)
    subprocess.run = run
