"""Schedule independent calibration processes on the authorized physical GPUs."""

import argparse
import concurrent.futures
from pathlib import Path
import queue
import threading
import time

from guarded import ROOT, gpu_attempt, read, sha, write


CASE = ROOT / 'runs/sm90_search_quality_v2/qwen3_1_7b_b1'
REPRO = CASE / 'phase0_reproduction'
CALIB = CASE / 'calibration'
LEGACY = ('base', 'epilogue', 'attention', 'gates', 'dsmem-merge',
          'lm-head-tp', 'moe-task', 'e64', 'q1', 'moe-merge', 'tp-merge', 'forward')


def prior_attempt(folder):
    for attempt in sorted(folder.glob('attempt*')):
        path = attempt / 'guard/guard_result.json'
        if not path.exists():
            continue
        guard = read(path)
        if guard['code'] != 75:
            return attempt, guard
    return None


def execute(spec, directory, cards, policy):
    jobs = queue.Queue()
    outcomes = {}
    halted = threading.Event()
    for job in spec:
        jobs.put(job)

    def worker(card):
        while not halted.is_set():
            try:
                job = jobs.get(timeout=1)
            except queue.Empty:
                continue
            try:
                label = job['id']
                folder = directory / label
                result = prior_attempt(folder)
                if result:
                    attempt, guard = result
                else:
                    i = max((int(p.name[7:]) for p in folder.glob('attempt*')), default=-1) + 1
                    attempt = folder / f'attempt{i:04d}'
                    guard = gpu_attempt(job['command'], attempt, policy, card, timeout_s=1800)
                if guard['code'] == 75:
                    jobs.put(job)
                    halted.wait(10)
                    continue
                if guard['code'] != 0:
                    raise RuntimeError(f'calibration job {label} failed at {attempt}')
                identity = read(attempt / 'job/physical-device-identity.json')
                if identity['nvml_uuid'] not in {c['uuid'] for c in cards}:
                    raise RuntimeError('accepted calibration attempt used an unregistered device')
                outcomes[label] = {'status': 'done', 'out': str(attempt / 'guard'),
                                   'log': str(attempt / 'run.log'), 'gpu_uuid': identity['nvml_uuid']}
                print(label, identity['nvml_uuid'], 'done', flush=True)
            except BaseException:
                halted.set()
                raise
            finally:
                jobs.task_done()
            if len(outcomes) == len(spec):
                halted.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards)) as pool:
        futures = [pool.submit(worker, card) for card in cards]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if len(outcomes) != len(spec):
        raise RuntimeError('incomplete calibration cohort')
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-binary', type=Path)
    parser.add_argument('--job-spec', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.legacy_binary) == bool(args.job_spec):
        parser.error('provide exactly one of --legacy-binary and --job-spec')
    assert read(REPRO / 'summary.json')['pass'], 'Phase 0 must pass'
    bound = read(CALIB / 'body_binding.json')['header_sha256']
    for path, digest in bound.items():
        assert sha(ROOT / path) == digest, path
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.legacy_binary:
        binary = args.legacy_binary.resolve()
        jobs = [
            {'id': f'constants-{mode}-f{i}', 'command': [str(binary), f'--{mode}-only'],
             'binary_sha256': sha(binary)}
            for i in range(5) for mode in LEGACY
        ]
    else:
        jobs = read(args.job_spec)
    n = len(jobs)
    if n == 0 or len({j['id'] for j in jobs}) != n:
        raise ValueError('calibration jobs must have unique IDs')
    frozen = output / 'jobs.json'
    if frozen.exists():
        assert read(frozen) == jobs, 'calibration manifest changed'
    else:
        write(frozen, jobs)
    for job in jobs:
        assert sha(job['command'][0]) == job['binary_sha256']
    cards = read(REPRO / 'reproduction_inputs.json')['authorized_cards']
    start = time.time()
    outcomes = execute(jobs, output / 'items', cards, REPRO / 'guard_policy.json')
    for label, result in outcomes.items():
        link = output / f'{label}.log'
        if not link.exists():
            link.symlink_to(result['log'])
        assert link.resolve() == Path(result['log']).resolve()
    write(output / 'state.json', outcomes)
    write(output / 'timing.json', {'started': start, 'ended': time.time(), 'jobs': len(jobs)})


if __name__ == '__main__':
    main()
