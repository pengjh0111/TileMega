#!/usr/bin/env python3
"""SM90R5 TP baselines (continuation B0.5): R16 protocol at tp 2 and 4.

One job is one fresh process over B1-16 (warmup 1, N1 and N1024 x 3) on a whole
claimed card group; rounds 0-2 are the three fresh processes R16 requires. The
command chain is `sm90r4_baselines.job()`'s, with the device identity, NUMA
binding and guard taken over the group instead of one card.

`run` is opportunistic: it waits for a group of idle, policy-allowed cards and
takes the next job that fits. A tp whose size no group can hold is deferred, not
failed (DT-B0).
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, PYTHON, read, write
import sm90r5_dispatch as dispatch

POLICY = ROOT / 'configs/bench/sm90r2_guard_policy.json'
MEASUREMENT = ROOT / 'configs/bench/measurement_policy.json'
ENVS = {'vllm': '/workspace/llm-stack/envs/vllm/bin/python',
        'sglang': '/workspace/llm-stack/envs/sglang/bin/python',
        'trtllm': '/workspace/llm-stack/envs/trtllm/bin/python'}
MODEL = 'qwen3_1_7b'
INPUTS = ROOT / 'configs/serving/sm90d1/inputs' / MODEL
PROMPTS = ROOT / 'configs/prompts/qwen_isl64.json'
OUT = ROOT / 'runs/sm90r5/tp_baselines'
TIMEOUT_S = 14400
NEEDS_FREE_MIB = 74000
RETRIES = 3


def pci_addresses():
    """index -> `0000:xx:00.0` lower case, the form cuDeviceGetPCIBusId reports."""
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id',
                                   '--format=csv,noheader'], text=True)
    result = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        index, bus = (item.strip() for item in line.split(','))
        domain, rest = bus.lower().split(':', 1)
        result[int(index)] = f'{domain[-4:]}:{rest}'
    return result


def numa_node(pci):
    return Path('/sys/bus/pci/devices', pci, 'numa_node').read_text().strip()


def job(engine, tp, round_index, group, directory):
    """The guarded command chain for one fresh process on `group`."""
    policy = read(POLICY)
    uuids = policy['idle_power_mapping_provenance']['uuid_by_gpu']
    pci = pci_addresses()
    addresses = [pci[gpu] for gpu in group]
    nodes = sorted({numa_node(address) for address in addresses}, key=int)
    visible = ','.join(str(gpu) for gpu in group)
    baseline = [ENVS[engine], '-u', str(ROOT / f'python/tilemega/serving/{engine}_baseline.py'),
                '--model', str(directory / 'checkpoint'), '--prompt-ids', str(PROMPTS),
                '--batch', 'all', '--max-tokens', '1024', '--warmup', '1', '--repeats', '3',
                '--round-index', str(round_index), '--load-format', 'auto', '--tp', str(tp),
                '--policy', str(MEASUREMENT), '--out', str(directory / 'job/measurements')]
    checkpoint = [PYTHON, str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                  '--config', str(INPUTS), '--out', str(directory / 'checkpoint'),
                  '--seed', '0', '--max-gib', '16', '--', *baseline]
    identity = [PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                '--guard-policy', str(POLICY), '--out', str(directory / 'job'),
                '--expected-uuid', ','.join(uuids[str(gpu)] for gpu in group),
                '--expected-pci', ','.join(addresses),
                '--numa-node', ','.join(numa_node(address) for address in addresses),
                '--', *checkpoint]
    inner = ['numactl', f'--cpunodebind={",".join(nodes)}', *identity]
    command = [PYTHON, '-m', 'tilemega.bench.gpu_guard', '--policy', str(POLICY),
               '--out', str(directory / 'guard'), '--needs-free-mib', str(NEEDS_FREE_MIB),
               '--timeout-s', str(TIMEOUT_S), '--', *inner]
    environment = dict(os.environ, TILEMEGA_DEVICE_GROUP=visible, CUDA_VISIBLE_DEVICES=visible,
                       TILEMEGA_DEVICE_INDEX=str(group[0]), TILEMEGA_TIMING='1',
                       TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
                       CUDA_HOME='/usr/local/cuda-12.8', TORCH_CUDA_ARCH_LIST='9.0',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8',
                       VLLM_USE_V1='1')
    for key in ('TILEMEGA_NUMA_NODE', 'TILEMEGA_GPU_LOCK', 'TILEMEGA_GPU_LOCK_HELD',
                'TLLM_WORKER_USE_SINGLE_PROCESS'):
        environment.pop(key, None)
    return command, environment, dict(group=group, numa_nodes=nodes, pci=addresses)


def units(engines, tps, rounds):
    # Round-major: every engine and tp has one fresh process before any has two.
    return [{'name': f'{engine}_tp{tp}_r{round_index}', 'engine': engine, 'tp': tp,
             'round': round_index}
            for round_index in rounds for tp in tps for engine in engines]


def queue(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pending = units(a.engines, a.tps, a.rounds)
    write(OUT / 'queue.json', pending)
    write(OUT / 'state.json', {unit['name']: {'status': 'pending', 'attempts': 0}
                               for unit in pending})
    print(json.dumps({'queued': len(pending), 'queue': str(OUT / 'queue.json')}))
    return 0


def occupancy():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in raw.splitlines():
        if line.strip():
            index, used, util = (item.strip() for item in line.split(','))
            result[int(index)] = {'memory_used_mib': int(float(used)), 'utilization_pct': int(float(util))}
    return result


def run(a):
    pending = read(OUT / 'queue.json')
    state = read(OUT / 'state.json')
    deferred = []
    while pending:
        unit = pending.pop(0)
        name = unit['name']
        if state[name]['status'] == 'done':
            continue
        policy = read(POLICY)
        # Cards outside this degree's set never join its group (user, 2026-11-01:
        # TP2 on 0/5/7 like single-card work; TP4 may take any timing card).
        exclude = tuple(g for g in range(8) if g not in a.cards[unit['tp']])
        if not dispatch.groups(policy, unit['tp'], exclude):
            # DT-B0: no group of this size exists under the standing policy.
            state[name].update(status='deferred',
                               reason=f'policy allows timing on {dispatch.allowed(policy)}')
            deferred.append(name)
            write(OUT / 'state.json', state)
            continue
        if a.yield_to_moe and dispatch.machine_waiting():
            # MoE timing has priority: wait it out rather than take the machine.
            print(json.dumps({'unit': name, 'waiting': 'a machine-class unit is queued'}), flush=True)
            pending.insert(0, unit)
            time.sleep(60)
            continue
        # Each attempt gets its own directory: the checkpoint builder refuses to
        # write into an existing one, and rounds from two attempts must not mix.
        directory = OUT / name / f'try{state[name]["attempts"] + 1}'
        directory.mkdir(parents=True, exist_ok=True)
        group, handles = dispatch.claim_group(name, unit['tp'], exclude)
        try:
            command, environment, binding = job(unit['engine'], unit['tp'], unit['round'],
                                                group, directory)
            before = occupancy()
            state[name].update(status='running', attempts=state[name]['attempts'] + 1,
                               started=time.time(), **binding)
            write(OUT / 'state.json', state)
            write(directory / 'command.json', {'command': command, 'group': group,
                                               'occupancy_before': before})
            state[name]['directory'] = str(directory)
            with (directory / 'run.log').open('a') as log:
                code = subprocess.call(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
        finally:
            for handle in handles:
                handle.close()
        state[name].update(returncode=code, finished=time.time(),
                           occupancy_after=occupancy())
        if code == 0:
            state[name]['status'] = 'done'
        elif code == 75 and state[name]['attempts'] < RETRIES:
            # Rejected or voided by the guard: back of the queue, same job.
            state[name]['status'] = 'pending'
            pending.append(unit)
        else:
            state[name]['status'] = 'failed'
        write(OUT / 'state.json', state)
        print(json.dumps({'unit': name, 'group': group, 'returncode': code,
                          'status': state[name]['status']}), flush=True)
    print(json.dumps({'deferred': deferred}), flush=True)
    return 0


ENGINE_NAMES = {'vllm': 'vllm', 'sglang': 'sglang', 'trtllm': 'tensorrt-llm'}


def collect(a):
    """SM90R5_tp_baselines.csv: R16 medians over the done fresh processes, TP1 frozen alongside."""
    import csv
    state = read(OUT / 'state.json')
    tp1 = {}
    for row in csv.DictReader(open(ROOT / 'docs/results/R16_final_baselines.csv')):
        if row['model'] == MODEL:
            tp1[(row['system'], int(row['batch']))] = float(row['tpot_ms'])
    rows = []
    for engine in ENVS:
        for tp in (2, 4):
            names = sorted(n for n in state if n.startswith(f'{engine}_tp{tp}_'))
            done = [n for n in names if state[n]['status'] == 'done']
            for batch in (1, 2, 4, 8, 16):
                records = []
                for name in done:
                    path = Path(state[name]['directory']) / f'job/measurements/B{batch}/measurements.json'
                    if path.exists():
                        records.append((name, read(path)))
                reference = tp1.get((ENGINE_NAMES[engine], batch))
                row = dict(model=MODEL, system=ENGINE_NAMES[engine], tp=tp, batch=batch,
                           fresh_processes=len(records), tp1_tpot_ms=f'{reference:.4f}' if reference else '')
                if not records:
                    row.update(tpot_ms='N/A', status='/'.join(sorted({state[n]['status'] for n in names})))
                    rows.append(row)
                    continue
                tpot = [r['tpot_seconds'] * 1e3 for _, r in records]
                median = statistics.median(tpot)
                row.update(
                    tpot_ms=f'{median:.4f}',
                    status='complete' if len(records) >= 3 else f'{len(records)} of 3 processes',
                    per_process_ms=';'.join(f'{v:.4f}' for v in tpot),
                    range_pct=f'{(max(tpot) - min(tpot)) / median * 100:.2f}',
                    ttft_ms=f'{statistics.median(r["ttft_seconds"] for _, r in records) * 1e3:.3f}',
                    e2e_ms=f'{statistics.median(r["e2e_seconds"] for _, r in records) * 1e3:.1f}',
                    groups=';'.join(','.join(map(str, state[n]['group'])) for n, _ in records),
                    checkpoint_sha256=records[0][1]['checkpoint']['checkpoint_sha256'],
                    versions=json.dumps({k: v for k, v in records[0][1]['versions'].items() if v},
                                        sort_keys=True),
                    launcher=records[0][1].get('configuration', {}).get('launcher', ''),
                    scaling_efficiency=f'{reference / (tp * median):.4f}' if reference else '')
                rows.append(row)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(a.csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), csv=str(a.csv))))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('queue', 'run'):
        one = sub.add_parser(name)
        one.add_argument('--engines', nargs='+', default=list(ENVS))
        one.add_argument('--tps', type=int, nargs='+', default=[2, 4])
        one.add_argument('--rounds', type=int, nargs='+', default=[0, 1, 2])
        if name == 'run':
            one.add_argument('--yield-to-moe', action=argparse.BooleanOptionalAction, default=True,
                             help='let a queued machine-class unit (MoE timing) go first')
            one.add_argument('--tp2-cards', type=int, nargs='+', default=[0, 5, 7])
            one.add_argument('--tp4-cards', type=int, nargs='+', default=list(range(8)))
    c = sub.add_parser('collect')
    c.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R5_tp_baselines.csv')
    a = p.parse_args()
    a.cards = {2: set(getattr(a, 'tp2_cards', range(8))), 4: set(getattr(a, 'tp4_cards', range(8)))}
    return dict(queue=queue, run=run, collect=collect)[a.action](a)


if __name__ == '__main__':
    sys.exit(main())
