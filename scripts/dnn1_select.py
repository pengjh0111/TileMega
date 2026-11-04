#!/usr/bin/env python3
"""Measure native top-K in one inherited guard, in A B C D D C B A order."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def summarize(candidates, visits):
    """P-SEL compares two fresh process medians per compiled candidate."""
    if not candidates or len({c['rank'] for c in candidates}) != len(candidates):
        raise ValueError('P-SEL requires distinct compiled candidate ranks')
    ranks = [c['rank'] for c in candidates]
    if [v['rank'] for v in visits] != ranks + ranks[::-1]:
        raise ValueError('P-SEL visits must follow forward then reverse candidate order')
    if len({v['pid'] for v in visits}) != len(visits):
        raise ValueError('every P-SEL visit requires a fresh process')
    arms = {}
    for candidate in candidates:
        rank = candidate['rank']
        observations = [v for v in visits if v['rank'] == rank]
        medians = [v['median_ms'] for v in observations]
        p90s = [v['p90_ms'] for v in observations]
        if any(not math.isfinite(x) or x <= 0 for x in [*medians, *p90s]):
            raise ValueError('P-SEL timing values must be finite and positive')
        predicted = candidate['predicted_ns']
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError('P-SEL predictions must be finite and positive')
        arms[rank] = dict(rank=rank, median_ms=statistics.median(medians),
            p90_ms=statistics.median(p90s), process_range_ms=max(medians) - min(medians),
            process_range_pct=(max(medians) / min(medians) - 1) * 100,
            processes=2, predicted_ns=predicted, binary_sha256=candidate['binary_sha256'])
    measured = sorted(arms, key=lambda rank: (arms[rank]['median_ms'], rank))
    winner = measured[0]
    threshold = gap = None
    if len(measured) > 1:
        best, second = measured[:2]
        threshold = max(0.5, 2 * max(arms[r]['process_range_pct'] for r in (best, second)))
        gap = (arms[second]['median_ms'] / arms[best]['median_ms'] - 1) * 100
        if gap <= threshold:
            winner = min((best, second), key=lambda r: (arms[r]['predicted_ns'], r))
    return dict(protocol='P-SEL', arms=list(arms.values()), measured_order=measured,
        predicted_order=sorted(arms, key=lambda r: (arms[r]['predicted_ns'], r)),
        winner_rank=winner, winner_binary_sha256=arms[winner]['binary_sha256'],
        top_gap_pct=gap, threshold_pct=threshold, tie_broken_by_prediction=winner != measured[0],
        range_definition='100 * (max process median / min process median - 1)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--export', dest='export_dir', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--inputs', required=True, help='fixed path, optionally containing {batch}')
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'python/tilemega/serving')]
    from dnn1_baselines import require_session
    from tilemega.build.identity import verify
    gpu = int(os.environ.get('TILEMEGA_DEVICE_INDEX', '-1'))
    require_session(dict(gpu=gpu))
    # The solver invokes this command inside its guard. A nested scheduler would
    # wait on the very GPU/NUMA locks already held by this parent session.
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / 'python/tilemega/dnn/benchmark_timing_only.py',
                    ROOT / 'python/tilemega/dnn/check.py', ROOT / 'python/tilemega/dnn/baselines.py',
                    ROOT / 'scripts/dnn1_baselines.py']
    frozen = {str(p.resolve()): sha(p) for p in source_paths}
    frozen[str(args.candidates.resolve())] = sha(args.candidates)
    raw = json.loads(args.candidates.read_text())
    candidates = sorted((c for c in raw if c['status'] == 'compiled'), key=lambda c: c['rank'])
    if not 1 <= len(candidates) <= 4:
        raise ValueError('DNN1 measures one to four compiled candidates')
    batch = None
    for candidate in candidates:
        binary = Path(candidate['binary'])
        identity = verify(binary)
        if candidate['binary_sha256'] != sha(binary) or identity['trace']:
            raise ValueError('candidate binary differs or contains instrumentation')
        if identity['plan']['export_sha256'] != sha(args.bridge):
            raise ValueError('candidate was compiled from another export bridge')
        current = identity['plan']['runtime_batch']
        if current not in (1, 2, 4, 8, 16) or (batch is not None and current != batch):
            raise ValueError('candidate batch sizes differ or fall outside DNN1')
        batch = current
        folder = args.out / f"candidate-{candidate['rank']}"
        folder.mkdir()
        selection = dict(schema=1, artifact_id=identity['artifact_id'],
            binary_sha256=sha(binary), identity_sha256=sha(str(binary) + '.identity.json'),
            source_head=identity['source']['head'],
            selection=dict(protocol='P-SEL', rank=candidate['rank'], predicted_ns=candidate['predicted_ns']))
        write(folder / 'selected-config.json', selection)
    inputs = Path(args.inputs.format(batch=batch)).resolve()
    for path in (inputs, args.bridge, args.export_dir / 'manifest.json',
                 args.export_dir / 'exported_program.pt2', args.export_dir / 'checkpoint/model.safetensors'):
        frozen[str(path.resolve())] = sha(path)
    constants = os.environ.get('TILEMEGA_FORWARD_CONSTANTS_FILE')
    if constants:
        frozen[str(Path(constants).resolve())] = sha(constants)
    write(args.out / 'session.json', dict(schema=1, protocol='P-SEL', gpu=gpu, batch=batch,
        started=time.time(), guard_pid=os.environ.get('TILEMEGA_DNN1_GUARD_PID'),
        source_sha256=frozen, candidate_statuses=raw, driver_command=[sys.executable, *sys.argv],
        candidate_order=[c['rank'] for c in candidates] + [c['rank'] for c in candidates[::-1]],
        warmup=50, iterations=200, correctness='separate P-CORR after final selection'))
    visits = []
    counts = {c['rank']: 0 for c in candidates}
    for visit, candidate in enumerate(candidates + candidates[::-1]):
        require_session(dict(gpu=gpu))
        rank = candidate['rank']
        folder = args.out / f'visit-{visit:02d}-candidate-{rank}'
        config = args.out / f'candidate-{rank}/selected-config.json'
        command = [sys.executable, '-m', 'tilemega.dnn.benchmark_timing_only',
            '--protocol', 'P-SEL', '--library', candidate['binary'], '--export', str(args.export_dir),
            '--bridge', str(args.bridge), '--inputs', str(inputs), '--batch', str(batch),
            '--selection-config', str(config), '--round-index', str(counts[rank]), '--out', str(folder)]
        counts[rank] += 1
        log = args.out / f'visit-{visit:02d}.log'
        with log.open('w') as output:
            process = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if process.returncode:
            write(args.out / 'failure.json', dict(visit=visit, rank=rank, returncode=process.returncode,
                                                 command=command, log=str(log)))
            raise RuntimeError('P-SEL child failed; retain its original log: ' + str(log))
        path = folder / 'measurements/measurements.json'
        row = json.loads(path.read_text())
        if (row['gpu_index'] != gpu or row['batch'] != batch or row['protocol'] != 'P-SEL' or
                row['input_sha256'] != frozen[str(inputs)] or row['round_index'] != counts[rank] - 1 or
                row['identity']['binary_sha256'] != candidate['binary_sha256']):
            raise ValueError('P-SEL child measurement identity differs from its session')
        visits.append(dict(rank=rank, visit=visit, pid=row['pid'], median_ms=row['median_ms'],
            p90_ms=row['p90_ms'], measurement=str(path), measurement_sha256=sha(path),
            guard_log=row['guard_log']))
        write(args.out / 'visits.json', visits)
    if any(sha(path) != digest for path, digest in frozen.items()):
        raise ValueError('P-SEL source, inputs, or constants changed during the session')
    for candidate in candidates:
        if verify(candidate['binary'])['binary_sha256'] != candidate['binary_sha256']:
            raise ValueError('candidate changed during P-SEL')
    result = dict(schema=1, gpu=gpu, batch=batch, ended=time.time(),
                  guard_acceptance='pending outer guard verdict', visits=visits,
                  source_sha256=frozen, **summarize(candidates, visits))
    write(args.out / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('winner_rank', 'winner_binary_sha256', 'measured_order',
                                             'top_gap_pct', 'threshold_pct')}), flush=True)


if __name__ == '__main__':
    main()
