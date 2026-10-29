#!/usr/bin/env python3
"""Run guarded measurement units on whichever allowed card is free when they start.

A unit is one spec of `sm90r3_llm.py` (pair, sel, final or corr). It stays on
one card for its whole life, because §2.2 and §2.3 compare arms on the same
card. A unit whose card stays occupied by a foreign job while another card is
free is abandoned whole and restarted there; each card gets its own `tryN`
directory, so rounds from two cards are never mixed.

Claims are flocks on /tmp/tilemega_claim<i>.lock, taken under one mutex. Their
class encodes §1.5: `trial` (dense and new-model cells, NUMA_CONCURRENCY_TRIAL)
may share a NUMA node with other trial units; `node` (serial units, the MoE
pairs) is alone among this harness's units on its node; `machine` (MoE P-SEL
and P-FINAL) runs with no other unit of this harness on any card.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from sm90d1_common import ROOT, PYTHON, read, write

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
LLM = ROOT / 'scripts/sm90r3_llm.py'
MUTEX = Path('/tmp/tilemega_claim.mutex')
MACHINE_WAIT = Path('/tmp/tilemega_claim_machine.wait')
# Longer than one guarded MoE round with its cooldown, so a single foreign
# job that leaves again does not cost a unit its completed rounds.
STALL_S = 1800


def numa(gpu):
    return 0 if gpu < 4 else 1


def allowed(policy):
    gpus = set(range(8))
    for source in (policy, read(policy['gpu_allowlist_file']) if policy.get('gpu_allowlist_file') else {}):
        if source.get('allowed_timing_gpu_indices') is not None:
            gpus &= set(source['allowed_timing_gpu_indices'])
    return sorted(gpus)


def held(path):
    # flock is per open file description, so this also reports this
    # process's own claims as held.
    with open(path, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def live_claims():
    live = {}
    for gpu in range(8):
        if Path(f'/tmp/tilemega_claim{gpu}.lock').exists() and held(f'/tmp/tilemega_claim{gpu}.lock'):
            try:
                live[gpu] = read(f'/tmp/tilemega_claim{gpu}.json')['class']
            except (OSError, ValueError, KeyError):
                live[gpu] = 'machine'  # an unreadable holder blocks everything
    return live


def compatible(cls, gpu, live):
    if gpu in live or 'machine' in live.values():
        return False
    if cls == 'machine':
        return not live
    same = [c for g, c in live.items() if numa(g) == numa(gpu)]
    return not same if cls == 'node' else 'node' not in same


def idle_card(gpu, policy):
    query = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu,power.draw',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    apps = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True)
    if query.returncode or apps.returncode:
        return False
    used, util, power = (float(v) for v in query.stdout.split(','))
    idle_w = policy.get('idle_power_w_by_gpu', {}).get(str(gpu), policy['idle_power_w'])
    return (not apps.stdout.strip() and used <= policy['max_hidden_mib'] and util <= policy['max_util_pct']
            and power <= idle_w + policy['power_margin_w'] and not held(f'/tmp/tilemega_gpu{gpu}.lock'))


def machine_waiting():
    try:
        pid = int(MACHINE_WAIT.read_text())
    except (OSError, ValueError):
        return False
    return pid != os.getpid() and Path('/proc', str(pid)).exists()


def claim(cls, label, exclude=(), only=None, nodes=None, poll=30):
    """Block until a card is claimed; returns (gpu, handle). Closing the handle releases it."""
    policy = read(POLICY)
    if cls == 'machine':
        MACHINE_WAIT.write_text(str(os.getpid()))
    try:
        while True:
            with MUTEX.open('a') as mutex:
                fcntl.flock(mutex, fcntl.LOCK_EX)
                live = live_claims()
                # A waiting machine unit drains the machine: nothing new starts.
                candidates = [] if cls != 'machine' and machine_waiting() else \
                    [only] if only is not None else [g for g in allowed(policy) if g not in exclude and
                                                     (nodes is None or numa(g) in nodes)]
                for gpu in candidates:
                    if not compatible(cls, gpu, live) or (only is None and not idle_card(gpu, policy)):
                        continue
                    handle = open(f'/tmp/tilemega_claim{gpu}.lock', 'a')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    write(f'/tmp/tilemega_claim{gpu}.json', {'class': cls, 'pid': os.getpid(), 'unit': label,
                                                            'since': time.time()})
                    return gpu, handle
            time.sleep(poll)
    finally:
        try:
            if cls == 'machine' and int(MACHINE_WAIT.read_text()) == os.getpid():
                MACHINE_WAIT.unlink()
        except (OSError, ValueError):
            pass


def alternative(cls, gpu, policy):
    with MUTEX.open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        live = {g: c for g, c in live_claims().items() if g != gpu}
        return any(compatible(cls, g, live) and idle_card(g, policy) for g in allowed(policy) if g != gpu)


def abandon(scheduler, state_path):
    os.killpg(scheduler.pid, signal.SIGTERM)
    scheduler.wait()
    # The scheduler is only abandoned between rounds; a step it started in the
    # last moment runs in its own session and is stopped with it.
    for record in read(state_path).values():
        if record['status'] == 'running' and record.get('pid'):
            subprocess.run(['pkill', '-KILL', '-s', str(record['pid'])], check=False)


def watch(scheduler, state_path, cls, gpu, stall_s):
    policy, progress, since = read(POLICY), None, time.time()
    while scheduler.poll() is None:
        time.sleep(30)
        try:
            state = read(state_path)
        except (OSError, ValueError):
            continue
        done = sum(r['status'] == 'done' for r in state.values())
        if done != progress:
            progress, since = done, time.time()
            continue
        if (any(r['status'] == 'running' for r in state.values()) or time.time() - since < stall_s or
                idle_card(gpu, policy) or not alternative(cls, gpu, policy)):
            continue
        abandon(scheduler, state_path)
        return True
    return False


def run_unit(template, unit, cls, stall_s=STALL_S, nodes=None):
    """Measure one spec on one free card; returns the collected result."""
    unit = Path(unit)
    if (unit / 'unit.json').exists():
        return read(unit / 'result.json')
    unit.mkdir(parents=True, exist_ok=True)
    tries = sorted((p for p in unit.glob('try*') if p.is_dir()), key=lambda p: int(p.name[3:]))
    cards = [read(p / 'spec.json')['gpu'] for p in tries if (p / 'spec.json').exists()]
    while True:
        gpu, handle = claim(cls, str(unit), exclude=cards[-1:], nodes=nodes)
        attempt = unit / f'try{len(tries) + 1}'
        attempt.mkdir()
        tries.append(attempt)
        cards.append(gpu)
        try:
            spec = read(template)
            spec.pop('after', None)
            spec.update(gpu=gpu, concurrency='trial' if cls == 'trial' else 'serial')
            write(attempt / 'spec.json', spec)
            subprocess.check_call([PYTHON, str(LLM), 'queue', '--spec', str(attempt / 'spec.json'),
                                   '--queue-dir', str(attempt / 'queue')], cwd=ROOT, stdout=subprocess.DEVNULL)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
                       TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE=str(numa(gpu)),
                       PYTHONPATH=str(ROOT / 'python'))
            with (attempt / 'scheduler.log').open('w') as log:
                scheduler = subprocess.Popen([PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir',
                                              str(attempt / 'queue'), '--out', str(attempt / 'scheduler'),
                                              '--policy', str(POLICY), '--deadline-hours', '24'],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                stalled = watch(scheduler, attempt / 'scheduler/state.json', cls, gpu, stall_s)
        finally:
            handle.close()
        if stalled:
            write(attempt / 'abandoned.json', {'gpu': gpu, 'at': time.time(),
                                               'reason': f'card occupied by a foreign job for {stall_s} s '
                                                         'while another card was free'})
            continue
        state = read(attempt / 'scheduler/state.json')
        if scheduler.returncode or any(r['status'] != 'done' for r in state.values()):
            raise RuntimeError(f'unit did not complete: {attempt}')
        subprocess.check_call([PYTHON, str(LLM), 'collect', '--spec', str(attempt / 'spec.json'), '--state',
                               str(attempt / 'scheduler/state.json'), '--out', str(unit / 'result.json')],
                              cwd=ROOT, stdout=subprocess.DEVNULL)
        write(unit / 'unit.json', {'class': cls, 'gpu': gpu, 'try': attempt.name, 'cards': cards})
        return read(unit / 'result.json')


def hold(gpu, cls, pid):
    """Claim a card a process outside the dispatcher is already using, until it exits."""
    _, handle = claim(cls, f'external pid {pid}', only=gpu)
    while Path('/proc', str(pid)).exists():
        time.sleep(30)
    handle.close()


def run_units(units):
    errors = []

    def one(item):
        try:
            result = run_unit(item['spec'], item['unit'], item['class'], nodes=item.get('nodes'))
            print(json.dumps({'unit': item['unit'], **{k: result.get(k) for k in
                  ('decision', 'change_pct', 'threshold_pct', 'winner')}}), flush=True)
        except Exception as error:
            errors.append(item['unit'])
            print(json.dumps({'unit': item['unit'], 'error': str(error)}), flush=True)

    threads = [threading.Thread(target=one, args=(item,)) for item in units]
    for thread in threads:
        thread.start()
        time.sleep(5)  # claims in list order
    for thread in threads:
        thread.join()
    return 1 if errors else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run', help='measure a list of units [{spec, unit, class}]')
    run.add_argument('--units', type=Path, required=True)
    keep = sub.add_parser('hold', help='claim a card an external process is using until it exits')
    keep.add_argument('--gpu', type=int, required=True)
    keep.add_argument('--class', dest='cls', choices=('trial', 'node', 'machine'), required=True)
    keep.add_argument('--pid', type=int, required=True)
    sub.add_parser('claims', help='print the live claims')
    a = p.parse_args()
    if a.action == 'run':
        return run_units(read(a.units))
    if a.action == 'hold':
        return hold(a.gpu, a.cls, a.pid)
    with MUTEX.open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        print(json.dumps(live_claims()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
