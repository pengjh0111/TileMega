#!/usr/bin/env python3
"""Run one Phase Z cell: P-SEL, winner P-CORR, and three-process P-DNN."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha, verify

ORDER = (0, 1, 2, 3, 3, 2, 1, 0)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def select(rows):
    if [r['rank'] for r in rows] != list(ORDER):
        raise ValueError('P-SEL requires A B C D D C B A')
    if any(not math.isfinite(r['median_ms']) or r['median_ms'] <= 0 for r in rows):
        raise ValueError('selection requires finite positive process medians')
    samples = {rank: [r['median_ms'] for r in rows if r['rank'] == rank]
               for rank in range(4)}
    medians = {rank: statistics.median(values) for rank, values in samples.items()}
    first, second = sorted(medians, key=lambda rank: (medians[rank], rank))[:2]
    threshold = max(.005, 2 * max((max(samples[r]) - min(samples[r])) / medians[r]
                                for r in (first, second)))
    gap = medians[second] / medians[first] - 1
    tied = gap <= threshold
    winner = min(first, second) if tied else first
    return dict(winner=winner, medians_ms=medians, fastest=first, runner_up=second,
                relative_gap=gap, pair_threshold=threshold, prediction_tiebreak=tied)


def candidates(args):
    gate = json.loads(args.native_gate.read_text())
    if gate['status'] != 'passed' or gate['ranks'] != list(range(4)) or len(gate['rows']) != 80:
        raise ValueError('Phase Z requires a passed 20-cell, 80-candidate native gate')
    cpu = Path(gate['cpu_gate'])
    if sha(cpu) != gate['cpu_gate_sha256']:
        raise ValueError('CPU solve receipt changed')
    matches = [r for r in gate['rows'] if r['model'] == args.model and r['batch'] == args.batch]
    if sorted(r['rank'] for r in matches) != list(range(4)):
        raise ValueError('cell lacks four unique predicted candidates')
    result = {}
    for row in matches:
        lib = Path(row['out']) / 'model.so'
        identity = verify(lib)
        entries = [r for k, r in identity['resources'].items() if 'sm90_forward_f3_kernel' in k]
        if (identity['artifact_id'] != row['artifact_id'] or identity['trace'] or
                identity['plan']['native_forward']['abi'] != 'f3' or
                len(entries) != 1 or entries[0]['registers'] > 168 or entries[0]['spill']):
            raise ValueError('selection candidate differs from accepted native gate')
        result[row['rank']] = lib.resolve()
    return result


def execute(args, libraries):
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    args.out.mkdir(parents=True, exist_ok=False)
    frozen_paths = [Path(__file__), args.native_gate, args.inputs, args.bridge,
                    args.export / 'manifest.json', args.export / 'exported_program.pt2',
                    args.export / 'checkpoint/model.safetensors',
                    ROOT / 'scripts/dnn3_model_gate.py',
                    *sorted((ROOT / 'python/tilemega/dnn').glob('*.py'))]
    for lib in libraries.values():
        frozen_paths.extend((lib, Path(str(lib) + '.identity.json'),
                             Path(str(lib) + '.cu'), Path(str(lib) + '.ptxas.log')))
    frozen = {str(p): sha(p) for p in frozen_paths}
    save(args.out / 'inputs.json', frozen)
    result = dict(evidence='verified', protocol='P-SEL; winner P-CORR; P-DNN',
                  model=args.model, batch=args.batch, gpu=args.device, status='partial',
                  qualification='provisional_container_pid_namespace', selection=[])

    def run(name, command):
        with (args.out / (name + '.log')).open('w') as stream:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=os.environ,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)

    def common(lib):
        return ['--library', str(lib), '--export', str(args.export), '--bridge', str(args.bridge),
                '--inputs', str(args.inputs), '--batch', str(args.batch)]

    # Each candidate must certify the fixed timing inputs before P-SEL. The
    # selected candidate then receives the required five-process P-CORR gate.
    gates = {}
    for rank, lib in sorted(libraries.items()):
        folder = args.out / f'precheck-{rank}'
        run(f'precheck-{rank}', [str(ROOT / 'scripts/dnn3_model_gate.py'), *common(lib),
                              '--check-one', '--out', str(folder)])
        gates[rank] = folder / 'check.json'

    def measure(name, rank, gate, index):
        folder = args.out / name
        run(name, ['-m', 'tilemega.dnn.benchmark', *common(libraries[rank]),
                   '--gate', str(gate), '--executor', 'L1', '--warmup', '50',
                   '--iterations', '200', '--round-index', str(index), '--out', str(folder)])
        path = folder / 'measurements.json'
        d = json.loads(path.read_text())
        return dict(rank=rank, median_ms=d['median_ms'], p90_ms=d['p90_ms'],
                    pid=d['pid'], source=str(path), sha256=sha(path))

    for index, rank in enumerate(ORDER):
        result['selection'].append(measure(f'select-{index + 1}-rank-{rank}', rank, gates[rank], index))
        save(args.out / 'result.json', result)
    result['decision'] = select(result['selection'])
    winner = result['decision']['winner']
    save(args.out / 'result.json', result)
    corr = args.out / 'winner_corr'
    run('winner_corr', [str(ROOT / 'scripts/dnn3_model_gate.py'), *common(libraries[winner]),
                        '--execute', '--out', str(corr)])
    receipt = json.loads((corr / 'result.json').read_text())
    if receipt['status'] != 'passed' or len(receipt['processes']) != 5:
        raise ValueError('winner P-CORR did not pass five fresh processes')
    result['final'] = [measure(f'final-{i + 1}', winner, corr / 'corr-1/check.json', i)
                       for i in range(3)]
    medians = [r['median_ms'] for r in result['final']]
    result.update(median_ms=statistics.median(medians),
                  p90_ms=statistics.median(r['p90_ms'] for r in result['final']),
                  process_range_ms=max(medians) - min(medians),
                  winner_artifact_id=verify(libraries[winner])['artifact_id'])
    if len({r['pid'] for r in result['selection'] + result['final']}) != 11:
        raise ValueError('measurement positions did not use fresh processes')
    if any(sha(path) != digest for path, digest in frozen.items()):
        raise ValueError('Phase Z inputs changed during this cell')
    result['status'] = 'passed'
    save(args.out / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-gate', 'export', 'bridge', 'inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', choices=('resnet18', 'resnet50', 'vgg16', 'bert'), required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument('--device', type=int, choices=range(4), required=True)
    parser.add_argument('--cpu-set', default='128-135')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for name in ('native_gate', 'export', 'bridge', 'inputs', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    libraries = candidates(args)
    if args.execute:
        execute(args, libraries)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    queue = args.out / 'queue'
    queue.mkdir()
    env = dict(CUDA_VISIBLE_DEVICES=str(args.device), TILEMEGA_DEVICE_INDEX=str(args.device),
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.device}.lock', TILEMEGA_NUMA_NODE='0',
               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
               PYTHONPATH=str(ROOT / 'python'))
    command = ['taskset', '-c', args.cpu_set, sys.executable, str(Path(__file__).resolve()),
               '--execute', '--model', args.model, '--batch', str(args.batch),
               '--device', str(args.device), '--out', '{attempt_dir}/cell']
    for name in ('native_gate', 'export', 'bridge', 'inputs'):
        command.extend(('--' + name.replace('_', '-'), str(getattr(args, name))))
    save(queue / 'queue_select.json', [dict(name='dnn3-phase-z-cell', gpu=True, timing=True,
         timeout_s=10800, needs_free_mib=8192, command=command, cwd=str(ROOT), env=env)])
    scheduler = [str(ROOT / 'scripts/dnn3_idle_floor_python.py'), '-m', 'tilemega.bench.scheduler',
                 '--queue-dir', str(queue), '--out', str(args.out / 'scheduler'),
                 '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6']
    with (args.out / 'dispatch.log').open('w') as stream:
        subprocess.run(scheduler, cwd=ROOT, env=dict(os.environ, **env), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    state = json.loads((args.out / 'scheduler/state.json').read_text())['dnn3-phase-z-cell']
    save(args.out / 'result.json', state)
    return int(state['status'] != 'done')


if __name__ == '__main__':
    raise SystemExit(main())
