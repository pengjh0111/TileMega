#!/usr/bin/env python3
"""Anchor a completed DNN4 model/card to the fixed DNN3 final targets."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from dnn3_anchor import ARMS_BY_NAME, execute, frozen_measurement
from dnn1_baselines import sha

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('resnet18', 'resnet50', 'vgg16', 'bert'))
    parser.add_argument('--device', type=int, choices=(0, 2, 3), required=True)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--cells', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-set', default='112-119')
    parser.add_argument('--engine-cache', type=Path, default=ROOT / 'runs/dnn3/anchor_engine_cache')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.execute:
        execute(args)
        return
    if args.export is None or args.cells is None:
        parser.error('preparation requires --export and --cells')
    args.export = args.export.resolve()
    sources = [ROOT / 'docs/results' / name for name in
               ('DNN3_final.csv', 'DNN1_baselines.csv', 'DNN3_baselines_phase_z.csv')]
    targets = [r for r in csv.DictReader(sources[0].open()) if r['model'] == args.model]
    rows = [r for path in sources[1:] for r in csv.DictReader(path.open())]
    finals = [args.cells / f'{args.model}_b{batch}/result.json' for batch in (1, 2, 4, 8, 16)]
    for path in finals:
        result = json.loads(path.read_text())
        if result['status'] != 'passed' or result['model'] != args.model or result['gpu'] != args.device:
            raise ValueError('anchor requires five completed cells on the prescribed card')
    archive = sha(args.export / 'exported_program.pt2')
    checkpoint = sha(args.export / 'checkpoint/model.safetensors')
    frozen_paths = [Path(__file__), ROOT / 'scripts/dnn3_anchor.py', *sources, *finals,
                    args.export / 'manifest.json', args.export / 'exported_program.pt2',
                    args.export / 'checkpoint/model.safetensors', ROOT / 'scripts/dnn1_baselines.py',
                    *sorted((ROOT / 'python/tilemega/dnn').glob('*.py'))]
    anchors = []
    for batch in (1, 16):
        target = next(r for r in targets if int(r['batch']) == batch)
        matching = [r for r in rows if r['model'] == args.model and int(r['batch']) == batch
                    and r['arm'] == target['baseline_arm'] and r['status'] == 'pass'
                    and abs(float(r['median_ms'])-float(target['baseline_ms'])) < 1e-9]
        if len(matching) != 1:
            raise ValueError('fixed DNN3 target lacks a unique measured baseline source')
        path, receipt = frozen_measurement(matching[0], Path('/root/TileMega-dnn1'))
        if receipt['exported_program_sha256'] != archive or receipt['source_checkpoint_sha256'] != checkpoint:
            raise ValueError('anchor export differs from DNN3 target')
        frozen_paths.append(path)
        anchors.append(dict(batch=batch, arm=ARMS_BY_NAME[target['baseline_arm']],
                            frozen_ms=float(target['baseline_ms']), source=str(path),
                            inputs_sha256=receipt['configuration']['inputs_sha256']))
    spec = dict(model=args.model, gpu=args.device, export=str(args.export),
                exported_program_sha256=archive, source_checkpoint_sha256=checkpoint,
                anchors=anchors, frozen={str(p): sha(p) for p in frozen_paths})
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'spec.json', spec)
    queue = args.out / 'queue'
    queue.mkdir()
    env = dict(CUDA_VISIBLE_DEVICES=str(args.device), TILEMEGA_DEVICE_INDEX=str(args.device),
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.device}.lock', TILEMEGA_NUMA_NODE='0',
               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
               PYTHONPATH=str(ROOT / 'python'))
    command = ['taskset', '-c', args.cpu_set, sys.executable, str(Path(__file__).resolve()),
               '--execute', '--spec', str(args.out / 'spec.json'), '--model', args.model,
               '--device', str(args.device), '--engine-cache', str(args.engine_cache.resolve()),
               '--out', '{attempt_dir}/anchor']
    save(queue / 'queue_anchor.json', [dict(name='dnn4-phase-z-anchor', gpu=True, timing=True,
         timeout_s=18000, needs_free_mib=8192, command=command, cwd=str(ROOT), env=env)])
    with (args.out / 'dispatch.log').open('w') as log:
        subprocess.run([str(ROOT / 'scripts/dnn4_guard_python.py'), '-m', 'tilemega.bench.scheduler',
            '--queue-dir', str(queue), '--out', str(args.out / 'scheduler'), '--policy',
            str(ROOT / 'configs/bench/sm90r2_guard_policy.json'), '--deadline-hours', '6'],
            cwd=ROOT, env=dict(os.environ, **env), stdout=log, stderr=subprocess.STDOUT, check=True)
    state = json.loads((args.out / 'scheduler/state.json').read_text())['dnn4-phase-z-anchor']
    save(args.out / 'result.json', state)
    return int(state['status'] != 'done')


if __name__ == '__main__':
    raise SystemExit(main())
