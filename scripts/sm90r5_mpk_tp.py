#!/usr/bin/env python3
"""SM90R5 MPK baseline at tensor parallel degree T (prompt §5 B0.6).

One fresh guarded process per (degree, batch, round), each claiming the whole
card group so nothing else times meanwhile, on the same seed-0 random
checkpoint and the same R16 protocol as every other engine. The process shards
that checkpoint with MPK's own convert.py and becomes `mpirun -np T`; rank 0
times the request. MPK is unmodified at 63af900 plus its sm90 patch.

One process per cell because MPK's generated launcher cudaMallocs fused weight
buffers that finalize() never returns, so a --batch all process cannot afford
ten kernel builds (the R4 finding, carried over).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, PYTHON, read, write
import sm90r5_dispatch as dispatch
import sm90r5_tp_baselines as tb

MPK_PYTHON = '/root/.venvs/mpk/bin/python'
MIRAGE_HOME = '/root/mirage-mpk'
CUDA = '/usr/local/cuda-12.8'
NVSHMEM = '/root/.venvs/nvshmem/nvidia/nvshmem'
INPUTS = ROOT / 'configs/serving/sm90d1/inputs/qwen3_1_7b'
PROMPTS = ROOT / 'configs/prompts/qwen_isl64.json'
MODEL = 'qwen3_1_7b'
BATCHES = (1, 2, 4, 8, 16)
OUT = ROOT / 'runs/sm90r5/mpk_tp'
# TP2 keeps to the single-card timing set; TP4 may take any timing card but 6.
CARDS = {2: (0, 5, 7), 4: (0, 1, 2, 3, 4, 5, 7)}


# A unit that works takes about three minutes. When one MPK rank fails, the
# surviving ranks spin in the all-reduce and never return (observed: a TP4 B16
# unit held four cards at 11.7 GiB for 42 minutes with one card at 100%), so
# the guard's window is minutes rather than the whole collection's budget.
UNIT_TIMEOUT_S = 1800


def job(tp, batch, round_index, group, directory, max_tokens=1024, timeout_s=UNIT_TIMEOUT_S):
    """The guarded command chain for one fresh MPK process on `group`."""
    policy = read(tb.POLICY)
    uuids = policy['idle_power_mapping_provenance']['uuid_by_gpu']
    pci = tb.pci_addresses()
    addresses = [pci[gpu] for gpu in group]
    nodes = sorted({tb.numa_node(address) for address in addresses}, key=int)
    visible = ','.join(str(gpu) for gpu in group)
    baseline = [MPK_PYTHON, '-u', str(ROOT / 'python/tilemega/serving/mpk_baseline.py'),
                '--model', str(directory / 'checkpoint'), '--prompt-ids', str(PROMPTS),
                '--batch', str(batch), '--max-tokens', str(max_tokens), '--warmup', '1',
                '--repeats', '3', '--round-index', str(round_index), '--load-format', 'auto',
                '--tp', str(tp), '--shard-dir', str(directory / f'shards-mp{tp}'),
                '--policy', str(tb.MEASUREMENT), '--out', str(directory / 'job/measurements')]
    checkpoint = [PYTHON, str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                  '--config', str(INPUTS), '--out', str(directory / 'checkpoint'),
                  '--seed', '0', '--max-gib', '16', '--', *baseline]
    identity = [PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                '--guard-policy', str(tb.POLICY), '--out', str(directory / 'job'),
                '--expected-uuid', ','.join(uuids[str(gpu)] for gpu in group),
                '--expected-pci', ','.join(addresses),
                '--numa-node', ','.join(tb.numa_node(address) for address in addresses),
                '--', *checkpoint]
    command = [PYTHON, '-m', 'tilemega.bench.gpu_guard', '--policy', str(tb.POLICY),
               '--out', str(directory / 'guard'), '--needs-free-mib', '32768',
               '--timeout-s', str(timeout_s), '--', 'numactl',
               f'--cpunodebind={",".join(nodes)}', *identity]
    environment = dict(
        os.environ, TILEMEGA_DEVICE_GROUP=visible, CUDA_VISIBLE_DEVICES=visible,
        TILEMEGA_DEVICE_INDEX=str(group[0]), TILEMEGA_TIMING='1',
        TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
        CUDA_HOME=CUDA, CUDA_PATH=CUDA, MIRAGE_HOME=MIRAGE_HOME, TORCH_CUDA_ARCH_LIST='9.0',
        # mpirun, mpic++ and nvcc all have to be on PATH: MPK compiles its
        # NVSHMEM path with -ccbin=mpic++.
        PATH=f'/root/.venvs/mpk/bin:{CUDA}/bin:/usr/bin:/bin:' + os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=f'{NVSHMEM}/lib:{CUDA}/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
        NVSHMEM_INC_PATH=f'{NVSHMEM}/include', NVSHMEM_LIB_PATH=f'{NVSHMEM}/lib',
        MPI_INC_PATH='/usr/lib/x86_64-linux-gnu/openmpi/include',
        MPI_LIB_PATH='/usr/lib/x86_64-linux-gnu/openmpi/lib',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    for key in ('TILEMEGA_NUMA_NODE', 'TILEMEGA_GPU_LOCK', 'TILEMEGA_GPU_LOCK_HELD'):
        environment.pop(key, None)
    return command, environment, dict(group=group, numa_nodes=nodes, pci=addresses)


def units(tps, batches, rounds):
    # Round-major, so every cell has one fresh process before any has two.
    return [{'name': f'mpk_tp{tp}_B{batch}_r{index}', 'tp': tp, 'batch': batch, 'round': index}
            for index in rounds for tp in tps for batch in batches]


def queue(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pending = units(a.tps, a.batches, a.rounds)
    write(OUT / 'queue.json', pending)
    state = read(OUT / 'state.json') if (OUT / 'state.json').exists() else {}
    for unit in pending:
        state.setdefault(unit['name'], {'status': 'pending', 'attempts': 0})
    write(OUT / 'state.json', state)
    print(json.dumps({'queued': len(pending)}), flush=True)
    return 0


def run(a):
    pending = read(OUT / 'queue.json')
    state = read(OUT / 'state.json')
    deadline = time.time() + a.hours * 3600
    while pending:
        unit = pending.pop(0)
        name = unit['name']
        if state[name]['status'] == 'done':
            continue
        if time.time() > deadline:
            # §5 B0.6 boxes this collection; what is left stays pending and is
            # reported as the processes it actually has.
            state[name].setdefault('reason', f'{a.hours}h budget spent')
            state[name]['status'] = 'deferred'
            write(OUT / 'state.json', state)
            continue
        tp = unit['tp']
        policy = read(tb.POLICY)
        exclude = tuple(g for g in range(8) if g not in CARDS[tp])
        candidates = dispatch.groups(policy, tp, exclude)
        if not candidates:
            state[name].update(status='deferred', reason='no group of this size under the policy')
            write(OUT / 'state.json', state)
            continue
        # Another user takes cards 0-4 intermittently, and TP4 has exactly one
        # candidate group ([0,1,2,3], the only four allowed cards on one NUMA
        # node), so an occupied card there blocks the degree entirely. Wait for
        # a whole group to be free rather than claiming it and letting the
        # guard reject the launch, which would spend the unit's retries on
        # someone else's job.
        if not any(all(dispatch.idle_card(gpu, policy) for gpu in group)
                   for group in candidates):
            print(json.dumps({'unit': name, 'waiting': 'no idle group of this degree',
                              'candidates': candidates}), flush=True)
            pending.append(unit)
            time.sleep(120)
            continue
        directory = OUT / name / f'try{state[name]["attempts"] + 1}'
        directory.mkdir(parents=True, exist_ok=True)
        group, handles = dispatch.claim_group(name, tp, exclude)
        try:
            command, environment, binding = job(tp, unit['batch'], unit['round'], group, directory)
            state[name].update(status='running', attempts=state[name]['attempts'] + 1,
                               started=time.time(), directory=str(directory), **binding)
            write(OUT / 'state.json', state)
            write(directory / 'command.json', {'command': command, 'group': group})
            with (directory / 'run.log').open('a') as log:
                # Own process group, so that whatever survives the unit - a
                # rank still spinning in the all-reduce after a peer died - can
                # be reaped without touching this runner.
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=UNIT_TIMEOUT_S + 300)
                except subprocess.TimeoutExpired:
                    code = 124
                    log.write('runner: unit exceeded its window, killing the process group\n')
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        finally:
            for handle in handles:
                handle.close()
        state[name].update(returncode=code, finished=time.time())
        if code == 0:
            state[name]['status'] = 'done'
        elif code == 75:
            # The guard rejected or voided the launch. "preflight occupied" is
            # someone else's process, not a fault of this unit, so it does not
            # consume a retry; anything else does.
            occupied = 'occupied' in read(directory / 'guard/guard_result.json').get('reason', '')
            if occupied:
                state[name]['attempts'] -= 1
            if occupied or state[name]['attempts'] < 3:
                state[name]['status'] = 'pending'
                pending.append(unit)
            else:
                state[name]['status'] = 'failed'
        else:
            state[name]['status'] = 'failed'
        # The shards are a derived copy of a checkpoint that is rebuilt per
        # process; keeping thirty of them would cost ~100 GiB of disk.
        shards = directory / f'shards-mp{tp}'
        if shards.is_dir():
            for item in shards.iterdir():
                item.unlink()
            shards.rmdir()
        write(OUT / 'state.json', state)
        print(json.dumps({'unit': name, 'group': group, 'returncode': code,
                          'status': state[name]['status']}), flush=True)
    return 0


def collect(a):
    """SM90R5_mpk_tp.csv: R16 medians over the fresh processes of each cell."""
    state = read(OUT / 'state.json')
    tilemega = {}
    for row in csv.DictReader(open(ROOT / 'docs/results/SM90R5_tp_final.csv')):
        tilemega[(int(row['tp']), int(row['batch']))] = float(row['tpot_ms'])
    rows = []
    for tp in a.tps:
        for batch in BATCHES:
            names = sorted(n for n in state if n.startswith(f'mpk_tp{tp}_B{batch}_r'))
            records = []
            for name in names:
                if state[name].get('status') != 'done':
                    continue
                path = (Path(state[name]['directory']) /
                        f'job/measurements/B{batch}/measurements.json')
                if path.exists():
                    records.append((name, read(path)))
            reference = tilemega.get((tp, batch))
            # A TPOT only exists for a process that finished, so the median is
            # conditional on completion. Publish what it is conditional on: the
            # processes launched, and the MPK faults among them counted from
            # the run logs rather than from the final status, because a unit
            # that faulted and then succeeded on a retry ends up 'done' and
            # would otherwise hide its fault.
            attempts = sum(state[n].get('attempts', 0) for n in names)
            faults = []
            for name in names:
                for log in sorted((OUT / name).glob('try*/run.log')):
                    text = log.read_text(errors='replace')
                    for mark in ('MPK stopped early', 'did not return the exact requested'):
                        if mark in text:
                            line = next(l for l in text.splitlines() if mark in l)
                            faults.append(f'{log.parent.parent.name}/{log.parent.name}: '
                                          f'{line.strip()[:160]}')
                            break
            row = dict(model=MODEL, system='mpk', tp=tp, batch=batch, fresh_processes=len(records),
                       processes_launched=attempts, mpk_faults=len(faults),
                       guard_voided_attempts=attempts - len(records) - len(faults),
                       faults='; '.join(faults),
                       tilemega_tpot_ms=f'{reference:.4f}' if reference else '')
            if not records:
                row.update(tpot_ms='N/A',
                           status='/'.join(sorted({state[n].get('status', '?') for n in names})
                                           or {'not queued'}))
                rows.append(row)
                continue
            tpot = [r['tpot_seconds'] * 1e3 for _, r in records]
            median = statistics.median(tpot)
            row.update(tpot_ms=f'{median:.4f}',
                       status='complete' if len(records) >= 3 else f'{len(records)} of 3 processes',
                       per_process_ms=';'.join(f'{v:.4f}' for v in tpot),
                       range_pct=f'{(max(tpot) - min(tpot)) / median * 100:.2f}',
                       ttft_ms=f'{statistics.median(r["ttft_seconds"] for _, r in records) * 1e3:.1f}',
                       e2e_ms=f'{statistics.median(r["e2e_seconds"] for _, r in records) * 1e3:.1f}',
                       ratio_mpk_over_tilemega=f'{median / reference:.4f}' if reference else '',
                       groups=';'.join(','.join(map(str, state[n]['group'])) for n, _ in records),
                       launcher=records[0][1]['configuration']['launcher'],
                       world_size=records[0][1]['configuration']['world_size'],
                       tp_collective=records[0][1]['configuration']['tp_collective'],
                       source_checkpoint_sha256=(
                           (records[0][1]['configuration'].get('tp_source_checkpoint') or {})
                           .get('checkpoint', {}).get('checkpoint_sha256', '')),
                       mirage_revision=records[0][1]['configuration']['mirage_revision'])
            rows.append(row)
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with open(a.csv, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'rows': len(rows), 'csv': str(a.csv)}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('queue', 'run', 'collect'):
        one = sub.add_parser(name)
        one.add_argument('--tps', type=int, nargs='+', default=[2, 4])
        if name != 'collect':
            one.add_argument('--batches', type=int, nargs='+', default=list(BATCHES))
            one.add_argument('--rounds', type=int, nargs='+', default=[0, 1, 2])
        if name == 'run':
            one.add_argument('--hours', type=float, default=6.0,
                             help='§5 B0.6 budget for this collection')
        if name == 'collect':
            one.add_argument('--csv', type=Path, default=ROOT / 'docs/results/SM90R5_mpk_tp.csv')
    a = p.parse_args()
    return dict(queue=queue, run=run, collect=collect)[a.action](a)


if __name__ == '__main__':
    sys.exit(main())
