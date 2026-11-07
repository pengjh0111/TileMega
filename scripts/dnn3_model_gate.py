#!/usr/bin/env python3
"""Run F3 classification correctness in five guarded processes (DNN3 §2)."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.build.identity import sha, verify
from tilemega.dnn.execution import execution_identity


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--inputs', type=Path, help='fixed safetensors inputs shared with timing')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--python', type=Path,
                        default=Path('/root/TileMega/runs/r16/env/tensorrt-dnn/bin/python'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-one', action='store_true')
    parser.add_argument('--sanitizers', action='store_true')
    parser.add_argument('--diagnostic-tool', choices=('memcheck', 'initcheck', 'racecheck', 'synccheck'))
    args = parser.parse_args()
    args.library, args.export, args.bridge, args.out = (
        path.resolve() for path in (args.library, args.export, args.bridge, args.out))
    if args.inputs is not None:
        args.inputs = args.inputs.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    model = json.loads((args.export / 'manifest.json').read_text())['model']
    if model not in ('resnet18', 'resnet50', 'vgg16'):
        raise ValueError('this gate requires a classification export')
    identity = verify(args.library)
    if identity['plan']['native_forward']['abi'] != 'f3' or identity['trace']:
        raise ValueError('requires a non-trace F3 model artifact')
    resources = [row for entry in identity['resources'].values()
                 for row in (entry, *entry.get('callees', {}).values())]
    if any(row.get('spill_store_bytes', 0) or row.get('spill_load_bytes', 0)
           or row.get('registers', 0) > 224 for row in resources):
        raise ValueError('DT-SPILL rejects model execution')
    # ptxas may list device functions before its first entry, outside the
    # common identity parser's entry-local grouping.
    log = Path(str(args.library) + '.ptxas.log').read_text()
    spills = re.findall(r'(\d+) bytes spill (?:stores|loads)', log)
    registers = re.findall(r'Used (\d+) registers', log)
    if not spills or any(map(int, spills)) or not registers or max(map(int, registers)) > 224:
        raise ValueError('DT-SPILL rejects complete ptxas resources')
    execution = execution_identity(None, identity, 1)
    if args.check_one:
        if not os.environ.get('TILEMEGA_GPU_LOCK'):
            raise ValueError('correctness requires the scheduler/guard reservation')
        from tilemega.dnn.check_generated import check
        result = check(args.library, args.export, args.bridge, args.batch,
                       epochs=3, input_tensors=args.inputs, thresholds={'cosine': .9999})
        if any(output.get('top1_agreement') != 1.0
               for case in result['cases'] for output in case['outputs']):
            raise AssertionError('classification top-1 differs from FP32')
        save(args.out / 'check.json', result)
        return 0
    base = [str(args.python), str(Path(__file__).resolve()), '--library', str(args.library),
            '--export', str(args.export), '--bridge', str(args.bridge), '--batch', str(args.batch)]
    if args.inputs is not None:
        base.extend(['--inputs', str(args.inputs)])
    if args.execute:
        if not os.environ.get('TILEMEGA_GPU_LOCK'):
            raise ValueError('cohort requires the scheduler/guard reservation')
        inputs = [Path(__file__), args.bridge, args.library,
                  Path(str(args.library) + '.identity.json')]
        inputs.extend((ROOT / 'python/tilemega/dnn').glob('*.py'))
        if args.inputs is not None:
            inputs.append(args.inputs)
        frozen = {str(path): sha(path) for path in inputs}
        required = 1 if args.diagnostic_tool else 5
        receipt = dict(protocol='diagnostic' if args.diagnostic_tool else 'P-CORR',
                       evidence='verified', execution=execution,
                       inputs=frozen, processes=[], required_processes=required)
        save(args.out / 'result.json', receipt)
        reference = None
        jobs = [('corr-' + str(i), []) for i in range(1, 6)]
        if args.diagnostic_tool:
            name = args.diagnostic_tool + '-1'
            jobs = [(name, ['compute-sanitizer', '--tool', args.diagnostic_tool,
                '--error-exitcode', '86', '--print-limit', '32',
                '--log-file', str(args.out / (name + '.sanitizer.log'))])]
        elif args.sanitizers:
            for tool, count in (('memcheck', 1), ('initcheck', 1), ('racecheck', 5), ('synccheck', 5)):
                jobs.extend((f'{tool}-{i}', ['compute-sanitizer', '--tool', tool,
                    '--error-exitcode', '86', '--log-file', str(args.out / f'{tool}-{i}.sanitizer.log')])
                    for i in range(1, count + 1))
        for name, sanitizer in jobs:
            destination = args.out / name
            command = sanitizer + base + ['--check-one', '--out', str(destination)]
            log = args.out / (name + '.log')
            with log.open('w') as stream:
                done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            row = dict(name=name, command=command, exit_code=done.returncode,
                       log=str(log), log_sha256=sha(log))
            if sanitizer:
                raw = args.out / (name + '.sanitizer.log')
                row.update(sanitizer_log=str(raw), sanitizer_sha256=sha(raw))
            receipt['processes'].append(row)
            if done.returncode:
                receipt['status'] = 'failed'
                save(args.out / 'result.json', receipt)
                return done.returncode
            measured = json.loads((destination / 'check.json').read_text())['cases']
            if reference is None:
                reference = measured
            if measured != reference:
                raise AssertionError('fresh process metrics differ bitwise')
            save(args.out / 'result.json', receipt)
        if any(sha(path) != value for path, value in frozen.items()):
            raise ValueError('model gate inputs changed during the cohort')
        receipt.update(status='passed', passes=f'{required}/{required}', cases=reference)
        save(args.out / 'result.json', receipt)
        return 0
    environment = dict(CUDA_VISIBLE_DEVICES=str(args.device), TILEMEGA_DEVICE_INDEX=str(args.device),
        TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.device}.lock', TILEMEGA_NUMA_NODE='0',
        TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
        PYTHONPATH=str(ROOT / 'python'))
    queue = args.out / 'queue'
    queue.mkdir()
    command = ['taskset', '-c', '128-135', *base, '--execute', '--out', '{attempt_dir}/cohort']
    if args.sanitizers:
        command.append('--sanitizers')
    if args.diagnostic_tool:
        command.extend(['--diagnostic-tool', args.diagnostic_tool])
    save(queue / 'queue_model.json', [dict(name='dnn3-f3-model-gates', gpu=True, timing=False,
        timeout_s=10800, needs_free_mib=8192, command=command, cwd=str(ROOT), env=environment)])
    scheduler = [str(ROOT / 'scripts/dnn1_idle_floor_python.py'), '-m', 'tilemega.bench.scheduler',
        '--queue-dir', str(queue), '--out', str(args.out / 'scheduler'),
        '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6']
    with (args.out / 'dispatch.log').open('w') as stream:
        subprocess.run(scheduler, cwd=ROOT, env=dict(os.environ, **environment),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    state = json.loads((args.out / 'scheduler/state.json').read_text())['dnn3-f3-model-gates']
    print(json.dumps(state, indent=2))
    return 0 if state['status'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
