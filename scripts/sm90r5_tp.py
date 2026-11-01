#!/usr/bin/env python3
"""SM90R5 TileMega tensor parallel measurement (B7.3, B9): P-SEL, P-FINAL, P-CORR.

Every measured process is one fresh launch of the whole card group: the claim
(machine class, so nothing else times anywhere), the group-mode GPU guard, the
group device identity, the shared random checkpoint (seed 0, the TP baselines'
lease) and `tilemega.serving.r16_run --tp T` over the prompt mode. TPOT is the
R16 protocol's (T_1024 - T_1) / 1023 from that run, exactly as TP1 P-SEL used.

  select   the native solver's --measure-cmd: A B C D D C B A over the compiled
           top-K, one launch per position; the smallest median wins, and a gap
           within the threshold goes to the better prediction (§2.3).
  final    P-FINAL: `launches` fresh launches of one binary (§2.4, 3 for TP).
  corr     P-CORR: C-1 over 1024 positions in launch 0, C-2 over 5 launches
           (identical 1024-token tables), every rank's tokens per step (§2.5).
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sm90d1_common import ROOT, PYTHON, read, sha, write
import sm90r5_dispatch as dispatch
import sm90r5_tp_baselines as tb

INPUTS = ROOT / 'configs/serving/sm90d1/inputs/qwen3_1_7b'
PROMPTS = ROOT / 'configs/prompts/qwen_isl64.json'
CARDS = {2: (0, 5, 7), 4: (0, 1, 2, 3, 4, 5, 7)}


def launch(tp, binary, batch, directory, label, round_index=0, hf_positions=None, step_loop=False,
           timeout_s=3600):
    """One guarded fresh launch on a claimed group; returns (code, measurement or None)."""
    directory.mkdir(parents=True, exist_ok=True)
    exclude = tuple(g for g in range(8) if g not in CARDS[tp])
    group, handles = dispatch.claim_group(label, tp, exclude)
    try:
        policy = read(tb.POLICY)
        uuids = policy['idle_power_mapping_provenance']['uuid_by_gpu']
        pci = tb.pci_addresses()
        addresses = [pci[g] for g in group]
        nodes = sorted({tb.numa_node(a) for a in addresses}, key=int)
        run = [PYTHON, '-m', 'tilemega.serving.r16_run', '--tp', str(tp), '--prefill', 'prompt',
               '--decode', str(Path(binary).resolve()), '--prompt-ids', str(PROMPTS),
               '--batch', str(batch), '--max-tokens', '1024', '--warmup', '1', '--repeats', '3',
               '--mode', 'L1', '--decode-loop', '1', '--round-index', str(round_index),
               '--policy', str(tb.MEASUREMENT), '--out', str(directory / 'job/measurements')]
        if hf_positions:
            run += ['--hf-check', '--hf-positions', str(hf_positions)]
        if step_loop:
            run.append('--check-step-loop')
        checkpoint = [PYTHON, str(ROOT / 'python/tilemega/serving/random_checkpoint.py'),
                      '--config', str(INPUTS), '--out', str(directory / 'checkpoint'),
                      '--seed', '0', '--max-gib', '16', '--', *run]
        identity = [PYTHON, str(ROOT / 'scripts/run_r16_with_device_identity.py'),
                    '--guard-policy', str(tb.POLICY), '--out', str(directory / 'job'),
                    '--expected-uuid', ','.join(uuids[str(g)] for g in group),
                    '--expected-pci', ','.join(addresses),
                    '--numa-node', ','.join(tb.numa_node(a) for a in addresses), '--', *checkpoint]
        command = [PYTHON, '-m', 'tilemega.bench.gpu_guard', '--policy', str(tb.POLICY),
                   '--out', str(directory / 'guard'), '--needs-free-mib', '32768',
                   '--timeout-s', str(timeout_s), '--', 'numactl',
                   f'--cpunodebind={",".join(nodes)}', *identity]
        visible = ','.join(map(str, group))
        env = dict(os.environ, TILEMEGA_DEVICE_GROUP=visible, CUDA_VISIBLE_DEVICES=visible,
                   TILEMEGA_DEVICE_INDEX=str(group[0]), TILEMEGA_TIMING='1',
                   TILEMEGA_GPU_PID_MODE='namespace_limited', PYTHONPATH=str(ROOT / 'python'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8')
        for key in ('TILEMEGA_NUMA_NODE', 'TILEMEGA_GPU_LOCK', 'TILEMEGA_GPU_LOCK_HELD'):
            env.pop(key, None)
        write(directory / 'command.json', dict(command=command, group=group, binary=str(binary),
                                               binary_sha256=sha(binary)))
        with (directory / 'run.log').open('a') as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        for handle in handles:
            handle.close()
    found = directory / f'job/measurements/B{batch}/measurements.json'
    record = read(found) if code == 0 and found.exists() else None
    write(directory / 'result.json', dict(returncode=code, group=group,
                                          tpot_ms=record['tpot_seconds'] * 1e3 if record else None))
    print(json.dumps(dict(label=label, group=group, returncode=code,
                          tpot_ms=record['tpot_seconds'] * 1e3 if record else None)), flush=True)
    return code, record


def span(values):
    return (max(values) - min(values)) / statistics.median(values) * 100


def select(a):
    out = a.out.resolve()
    out.mkdir(parents=True)
    rows = [r for r in read(a.candidates) if r['status'] == 'compiled']
    if len(rows) < 2:
        raise ValueError('P-SEL needs at least two compiled candidates')
    labels = [f"c{r['rank']}" for r in rows]
    binary = {f"c{r['rank']}": r['binary'] for r in rows}
    predicted = {f"c{r['rank']}": r['predicted_ns'] for r in rows}
    order = labels + labels[::-1]
    tpot = {label: [] for label in labels}
    for position, label in enumerate(order):
        code, record = launch(a.tp, binary[label], a.batch, out / f'{position:02d}-{label}',
                              f'{a.name}-{position:02d}-{label}', round_index=position)
        if code:
            raise SystemExit(f'P-SEL launch {position} ({label}) failed with {code}')
        tpot[label].append(record['tpot_seconds'] * 1e3)
    medians = {label: statistics.median(v) for label, v in tpot.items()}
    ranked = sorted(labels, key=medians.get)
    first, second = ranked[0], ranked[1]
    gap = (medians[second] - medians[first]) / medians[first] * 100
    # §2.2's threshold from the two arms' process spreads.
    threshold = max(0.5, 2 * max(span(tpot[first]), span(tpot[second])))
    tie = gap <= threshold
    winner = min((first, second), key=predicted.get) if tie else first
    write(out / 'psel.json', dict(winner=winner, measured_order=order, arms=tpot, medians=medians,
                                  top_gap_pct=gap, threshold_pct=threshold, tie_broken_by_prediction=tie))
    write(out / 'result.json', dict(schema=1, protocol='P-SEL (TP)', tp=a.tp, winner_rank=int(winner[1:]),
                                    winner_binary_sha256=sha(binary[winner]), measured_order=order,
                                    arms=tpot, predicted_ns=predicted, top_gap_pct=gap,
                                    threshold_pct=threshold, tie_broken_by_prediction=tie))
    return 0


def final(a):
    tpot, records = [], []
    for index in range(a.launches):
        directory = a.out / f'f{index}'
        if (directory / 'result.json').exists() and read(directory / 'result.json')['returncode'] == 0:
            record = read(directory / f'job/measurements/B{a.batch}/measurements.json')
        else:
            code, record = launch(a.tp, a.binary, a.batch, directory, f'{a.name}-f{index}', index)
            if code:
                raise SystemExit(f'P-FINAL launch {index} failed with {code}')
        tpot.append(record['tpot_seconds'] * 1e3)
        records.append(record)
    write(a.out / 'final.json', dict(tp=a.tp, batch=a.batch, binary=str(a.binary), binary_sha256=sha(a.binary),
                                     tpot_ms=statistics.median(tpot), per_launch_ms=tpot, range_pct=span(tpot),
                                     ttft_ms=statistics.median(r['ttft_seconds'] for r in records) * 1e3,
                                     e2e_ms=statistics.median(r['e2e_seconds'] for r in records) * 1e3))
    print(json.dumps(read(a.out / 'final.json')), flush=True)
    return 0


def corr(a):
    tables = []
    for index in range(a.launches):
        directory = a.out / f'c{index}'
        # Launch 0 also runs C-1 over 1024 positions and the executor
        # comparison, which need more than the measurement's hour.
        code, _ = launch(a.tp, a.binary, a.batch, directory, f'{a.name}-c{index}', index,
                         hf_positions=1024 if index == 0 else None, step_loop=index == 0,
                         timeout_s=7200 if index == 0 else 3600)
        if code:
            raise SystemExit(f'P-CORR launch {index} failed with {code}')
        tables.append(read(directory / 'job/measurements/reference_tokens.json'))
    c1 = read(a.out / 'c0/job/measurements/c1.json')
    c2 = all(table == tables[0] for table in tables)
    write(a.out / 'corr.json', dict(tp=a.tp, batch=a.batch, binary_sha256=sha(a.binary),
                                    c1_pass=c1['pass'], c1_gap_le_0_5_ratio=c1['gap_le_0_5_ratio'],
                                    c1_max_gap=c1['max_gap'], c2_pass=c2, c2_launches=len(tables),
                                    step_loop=read(a.out / 'c0/job/measurements/c2.json')['pass'],
                                    ranks_identical='checked by TpServingEngine every generation'))
    print(json.dumps(read(a.out / 'corr.json')), flush=True)
    return 0 if c1['pass'] and c2 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('select')
    s.add_argument('--tp', type=int, choices=(2, 4), required=True)
    s.add_argument('--batch', type=int, required=True)
    s.add_argument('--name', required=True)
    s.add_argument('--candidates', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    for name in ('final', 'corr'):
        one = sub.add_parser(name)
        one.add_argument('--tp', type=int, choices=(2, 4), required=True)
        one.add_argument('--batch', type=int, required=True)
        one.add_argument('--binary', type=Path, required=True)
        one.add_argument('--name', required=True)
        one.add_argument('--out', type=Path, required=True)
        one.add_argument('--launches', type=int, default=3 if name == 'final' else 5)
    a = p.parse_args()
    return dict(select=select, final=final, corr=corr)[a.action](a)


if __name__ == '__main__':
    sys.exit(main())
