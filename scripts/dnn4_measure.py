#!/usr/bin/env python3
"""Guard P-SMOKE or CP-1/2 P-PAIR using fixed, numerically checked inputs."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha, verify
from tilemega.dnn.benchmark import validate_gate


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'gate', 'export', 'bridge', 'inputs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol', choices=('P-SMOKE', 'P-PAIR'), default='P-SMOKE')
    parser.add_argument('--reference-library', type=Path)
    parser.add_argument('--reference-gate', type=Path)
    parser.add_argument('--reference-bridge', type=Path)
    parser.add_argument('--batch', type=int, required=True, choices=(1, 2, 4, 8, 16))
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    candidate = verify(args.library)
    if candidate['plan']['native_forward']['abi'] != 'f3' or candidate['trace']:
        raise ValueError('requires a non-trace F3 candidate')
    resources = Path(str(args.library) + '.ptxas.log').read_text()
    spills = re.findall(r'(\d+) bytes spill (?:stores|loads)', resources)
    registers = re.findall(r'Used (\d+) registers', resources)
    if not spills or any(map(int, spills)) or not registers or max(map(int, registers)) > 224:
        raise ValueError('DT-SPILL rejects candidate timing')
    artifacts = {'B': (args.library.resolve(), args.gate.resolve(), args.bridge.resolve())}
    order = ['B']
    if args.protocol == 'P-PAIR':
        if args.batch not in (1, 16) or any(value is None for value in
                (args.reference_library, args.reference_gate, args.reference_bridge)):
            raise ValueError('DT-F3 pairs require B1/B16 and frozen reference paths')
        reference = verify(args.reference_library)
        if reference['plan']['native_forward']['abi'] != 'f3' or reference['trace']:
            raise ValueError('P-PAIR reference must be an identified non-trace F3 artifact')
        artifacts['A'] = tuple(path.resolve() for path in
            (args.reference_library, args.reference_gate, args.reference_bridge))
        order = ['A', 'B', 'A', 'B', 'A', 'B']
    for library, gate, bridge in artifacts.values():
        identity = verify(library)
        validate_gate(json.loads(gate.read_text()), identity['artifact_id'],
                      args.inputs, args.export, bridge)
    environment = dict(CUDA_VISIBLE_DEVICES=str(args.device), TILEMEGA_DEVICE_INDEX=str(args.device),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.device}.lock', TILEMEGA_NUMA_NODE='0',
        TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT / 'python'))
    queue = args.out / 'queue'
    queue.mkdir()
    steps = []
    previous = None
    for index, arm in enumerate(order):
        library, gate, bridge = artifacts[arm]
        name = f'dnn4-{args.protocol.lower()}-{index+1}-{arm}'
        command = ['taskset', '-c', '104-111',
            str(ROOT / 'scripts/dnn4_guard_python.py'), '-m', 'tilemega.dnn.benchmark',
            '--library', str(library), '--gate', str(gate), '--bridge', str(bridge),
            '--export', str(args.export.resolve()), '--inputs', str(args.inputs.resolve()),
            '--batch', str(args.batch), '--executor', 'L1', '--warmup', '50', '--iterations', '200',
            '--round-index', str(index), '--out', '{attempt_dir}/measure']
        steps.append(dict(name=name, gpu=True, timing=True, timeout_s=1200,
            needs_free_mib=8192, command=command, cwd=str(ROOT), env=environment,
            after=[] if previous is None else [previous]))
        previous = name
    save(queue / 'queue_measure.json', steps)
    frozen_paths = [Path(__file__), args.inputs, args.export / 'exported_program.pt2',
                    args.export / 'checkpoint/model.safetensors']
    frozen_paths.extend((ROOT / 'python/tilemega/dnn').glob('*.py'))
    for library, gate, bridge in artifacts.values():
        frozen_paths.extend((library, gate, bridge, Path(str(library) + '.identity.json')))
    frozen = {str(path.resolve()): sha(path) for path in frozen_paths}
    save(args.out / 'inputs.json', frozen)
    scheduler = [str(ROOT / 'scripts/dnn4_guard_python.py'), '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(queue), '--out', str(args.out / 'scheduler'),
        '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6']
    with (args.out / 'dispatch.log').open('w') as stream:
        subprocess.run(scheduler, cwd=ROOT, env=dict(os.environ, **environment),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    if any(sha(path) != value for path, value in frozen.items()):
        raise ValueError('measurement inputs changed during the session')
    state = json.loads((args.out / 'scheduler/state.json').read_text())
    rows = []
    for step, arm in zip(steps, order):
        row = state[step['name']]
        if row['status'] != 'done':
            save(args.out / 'result.json', dict(protocol=args.protocol, status='failed', state=state))
            return 1
        folder = Path(row['out'])
        measured = json.loads((folder / 'measure/measurements.json').read_text())
        guard = json.loads((folder / 'guard_result.json').read_text())
        rows.append(dict(arm=arm, median_ms=measured['median_ms'], p90_ms=measured['p90_ms'],
                         measurement=str(folder / 'measure/measurements.json'), guard=guard))
    for row in rows:
        measured=json.loads(Path(row['measurement']).read_text())
        samples=[json.loads(line) for line in Path(measured['guard_log']).read_text().splitlines()]
        for label,key in [('before-measurement','start_sm_clock_mhz'),('after-measurement','end_sm_clock_mhz')]:
            found=[s for s in samples if s.get('label')==label]
            if len(found)!=1:raise ValueError('timing lacks exactly one clock boundary')
            row[key]=float(found[0]['telemetry']['clocks.sm'])
    result = dict(evidence='verified', protocol=args.protocol, batch=args.batch, gpu=args.device,
        input_sha256=sha(args.inputs), measurements=rows,
        qualification='provisional_container_pid_namespace; inspect guard receipts')
    b = [row['median_ms'] for row in rows if row['arm'] == 'B']
    result['candidate_ms'] = statistics.median(b)
    if args.protocol == 'P-PAIR':
        a = [row['median_ms'] for row in rows if row['arm'] == 'A']
        result['reference_ms'] = statistics.median(a)
        result['ratio'] = result['candidate_ms'] / result['reference_ms']
        result['threshold'] = max(.005, 2 * max((max(v)-min(v))/statistics.median(v) for v in (a,b)))
        result['slower'] = result['ratio'] - 1 > result['threshold']
    save(args.out / 'result.json', result)
    print(json.dumps({key:value for key,value in result.items() if key != 'measurements'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
