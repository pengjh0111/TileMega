#!/usr/bin/env python3
"""The R6 single-card collection queue: one worker per timing card.

Every remaining GPU measurement in the round is one unit of work that needs
exactly one card, so the throughput limit is cards rather than anything else.
This keeps a worker per card from --gpus and hands each the next pending unit,
so a card never idles while units remain. Progress does not depend on anyone
watching: state.json records every unit's outcome and a restart resumes.

A guard rejection (rc 75) is another user's process arriving between the claim
and the preflight; it is retried rather than charged to the unit.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT

OUT = ROOT / 'runs/sm90r6/collect'
PYTHON = '/workspace/llm-stack/envs/vllm/bin/python'
# section 9.3 bytes_step, weights + KV at ISL+OSL/2 = 576 rows (appendix D).
BYTES = {('qwen3_1_7b', 1): 3506962432, ('qwen3_1_7b', 16): 4497866752,
         ('llama3_1_8b', 1): 15084814336, ('llama3_1_8b', 16): 16217276416}
MOE_CELLS = ROOT / 'configs/serving/sm90r5/ref_r4'
BUILDS = ROOT / 'runs/sm90r4/build/r6h2'


def units(a):
    """The pending work, widest-first so the long units start earliest."""
    work = []
    if 'i1' in a.phases:
        for engine in ('vllm', 'sglang', 'trtllm'):
            for model in ('llama3_1_8b', 'qwen3_1_7b'):
                for batch in (1, 16):
                    work.append(dict(name=f'i1-{engine}-{model}-B{batch}', kind='i1',
                                     engine=engine, model=model, batch=batch))
    if 'h2' in a.phases:
        for model in ('qwen3_30b_a3b', 'olmoe_1b_7b'):
            for batch in (1, 2, 4, 8, 16):
                work.append(dict(name=f'h2-{model}-B{batch}', kind='h2',
                                 model=model, batch=batch))
    return work


def command(unit, gpu):
    if unit['kind'] == 'i1':
        return [PYTHON, str(ROOT / 'scripts/sm90r6_engine_trace.py'),
                '--engine', unit['engine'], '--model', unit['model'],
                '--batch', str(unit['batch']), '--gpu', str(gpu),
                '--bytes', str(BYTES[(unit['model'], unit['batch'])])]
    cell = BUILDS / f"{unit['model']}-B{unit['batch']}"
    return ['bash', str(ROOT / 'runs/sm90r6/h2/run.sh'), str(gpu), unit['model'],
            str(unit['batch']), str(cell)]


def build_moe_cell(unit):
    """A MoE trace needs its cell; the build is CPU work and cached on disk."""
    cell = BUILDS / f"{unit['model']}-B{unit['batch']}"
    if (cell / 'decode.so').exists():
        return True
    # OLMoE's REF-R4 cells carry group_wait with a whole-row QK norm, which
    # R5's A1 made illegal, so R5 re-solved all five; those winners are the
    # buildable coordinates. Qwen3-30B-A3B's REF-R4 cells are still legal.
    resolved = Path(f"/root/TileMega-r5/runs/sm90r5/a1/olmoe-B{unit['batch']}/winner.tmexec.json")
    coordinates = (resolved if unit['model'] == 'olmoe_1b_7b' and resolved.exists()
                   else MOE_CELLS / f"{unit['model']}_b{unit['batch']}.json")
    if not coordinates.exists():
        return False
    log = OUT / f"build-{unit['name']}.log"
    with log.open('w') as stream:
        code = subprocess.call(
            [PYTHON, str(ROOT / 'scripts/sm90r4_fixed.py'), '--model', unit['model'],
             '--batch', str(unit['batch']), '--coordinates', str(coordinates),
             '--tag', 'r6h2', '--prefill',
             '/root/TileMega/runs/sm90r1/moe-wide-nondn-B16-cuda128/prefill.so'],
            cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
    return code == 0 and (cell / 'decode.so').exists()


class Queue:
    def __init__(self, work, state_path):
        self.lock = threading.Lock()
        self.pending = list(work)
        self.state_path = state_path
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for unit in work:
            self.state.setdefault(unit['name'], {'status': 'pending', 'attempts': 0})
        self.flush()

    def flush(self):
        self.state_path.write_text(json.dumps(self.state, indent=2) + '\n')

    def take(self):
        with self.lock:
            while self.pending:
                unit = self.pending.pop(0)
                if self.state[unit['name']]['status'] in ('done', 'failed'):
                    continue
                return unit
            return None

    def finish(self, unit, code, requeue=False):
        with self.lock:
            entry = self.state[unit['name']]
            entry['attempts'] += 1
            entry['returncode'] = code
            if code == 0:
                entry['status'] = 'done'
            elif requeue and entry['attempts'] < 5:
                entry['status'] = 'pending'
                self.pending.append(unit)
            else:
                entry['status'] = 'failed'
            self.flush()
            print(json.dumps({'unit': unit['name'], 'returncode': code,
                              'status': entry['status'], 'attempts': entry['attempts']}),
                  flush=True)


def worker(gpu, queue):
    while True:
        unit = queue.take()
        if unit is None:
            return
        if unit['kind'] == 'h2' and not build_moe_cell(unit):
            queue.finish(unit, 90)
            continue
        if unit['kind'] == 'h2':
            # random_checkpoint refuses an existing directory, so a retried
            # cell must start from a clean one.
            stale = ROOT / f"runs/sm90r6/h2/{unit['model']}-B{unit['batch']}/checkpoint"
            if stale.exists():
                subprocess.call(['rm', '-rf', str(stale)])
        log = OUT / f"{unit['name']}.log"
        with log.open('a') as stream:
            stream.write(f"\n=== gpu {gpu} at {time.strftime('%H:%M:%S')}\n")
            stream.flush()
            code = subprocess.call(command(unit, gpu), cwd=str(ROOT), stdout=stream,
                                   stderr=subprocess.STDOUT)
        text = log.read_text(errors='replace')[-4000:]
        rejected = 'preflight occupied' in text or '"returncode": 75' in text
        if rejected:
            time.sleep(120)
        queue.finish(unit, code, requeue=rejected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 5, 7])
    p.add_argument('--phases', nargs='+', default=['i1', 'h2'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    queue = Queue(units(a), OUT / 'state.json')
    print(json.dumps({'queued': len(queue.pending), 'gpus': a.gpus}), flush=True)
    threads = [threading.Thread(target=worker, args=(gpu, queue), daemon=False)
               for gpu in a.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done = sum(1 for v in queue.state.values() if v['status'] == 'done')
    print(json.dumps({'done': done, 'total': len(queue.state)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
