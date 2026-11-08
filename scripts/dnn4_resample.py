#!/usr/bin/env python3
"""Resample five TRT/TC batch cells on a card after a failed Phase Z anchor."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tilemega.build.identity import sha
from dnn1_baselines import FIELDS, aggregate_arm, logical, strict_correctness
from dnn3_anchor import ARMS_BY_NAME, decision, fastest, frozen_measurement, save

BATCHES = (1, 2, 4, 8, 16)


def require_resample(anchor):
    if (anchor.get('status') != 'passed' or anchor.get('protocol') != 'P-ANCHOR' or
            anchor.get('action') != 'resample_trt_tc_all_batches' or
            sorted(r['batch'] for r in anchor['anchors']) != [1, 16] or
            not any(decision(r['frozen_ms'], r['median_ms'])['action'] != 'retain_frozen'
                    for r in anchor['anchors'])):
        raise ValueError('resampling requires a passed anchor with measured endpoint drift')


def order():
    arms = list(ARMS_BY_NAME.values())
    return [list(reversed(arms)) if index == 1 else arms for index in range(3)]


def prepare(args):
    anchor = json.loads(args.anchor.read_text()); require_resample(anchor)
    model, gpu = anchor['model'], anchor['gpu']
    baseline = ROOT / 'docs/results' / ('DNN1_baselines.csv' if model in
                                      ('resnet18', 'bert') else 'DNN3_baselines.csv')
    with baseline.open() as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((args.export / 'manifest.json').read_text())
    archive = sha(args.export / 'exported_program.pt2')
    checkpoint = sha(args.export / 'checkpoint/model.safetensors')
    if manifest['model'] != model or manifest['artifacts']['exported_program.pt2'] != archive:
        raise ValueError('resampling export identity differs')
    frozen = {str(p): sha(p) for p in [Path(__file__), args.anchor, baseline,
              args.export / 'manifest.json', args.export / 'exported_program.pt2',
              args.export / 'checkpoint/model.safetensors',
              ROOT / 'scripts/dnn1_baselines.py', ROOT / 'scripts/dnn3_anchor.py',
              ROOT / 'scripts/dnn4_guard_python.py',
              *sorted((ROOT / 'python/tilemega/dnn').glob('*.py'))]}
    cells = []
    for batch in BATCHES:
        reference = fastest(rows, model, batch)
        path, receipt = frozen_measurement(reference, args.dnn1_root)
        if (receipt['exported_program_sha256'] != archive or
                receipt['source_checkpoint_sha256'] != checkpoint):
            raise ValueError('resampling differs from frozen export/checkpoint')
        frozen[str(path)] = sha(path)
        cells.append(dict(model=model, batch=batch, gpu=gpu, export=str(args.export),
            exported_program_sha256=archive, source_checkpoint_sha256=checkpoint,
            inputs_sha256=receipt['configuration']['inputs_sha256'], seed=20261007,
            python=str(ROOT / 'scripts/dnn4_guard_python.py'),
            baselines=str(ROOT / 'python/tilemega/dnn/baselines.py'),
            engine_cache=str(args.out / 'engine-cache'),
            inductor_cache=str(args.out / 'inductor'), env={}, logical_timeout_s=7200,
            correctness_policy='five-independent-gates-v2', prefix=dict(available=False)))
    return dict(model=model, gpu=gpu, frozen=frozen, cells=cells)


def execute(args):
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    spec = json.loads(args.spec.read_text())
    cell = next(c for c in spec['cells'] if c['batch'] == args.batch)
    if any(sha(p) != digest for p, digest in spec['frozen'].items()):
        raise ValueError('resampling inputs changed')
    args.out.mkdir(parents=True, exist_ok=False)
    session = dict(model=cell['model'], batch=cell['batch'], gpu=cell['gpu'],
                   source=str(args.out / 'session.json'), status='partial', logical_jobs=[],
                   inputs_sha256=cell['inputs_sha256'], p_cell_complete=False,
                   runtime_libraries={})
    import torch
    library = Path(torch.__file__).parent / 'lib/libtorch_cuda.so'
    session['runtime_libraries']['torchcompile'] = {str(library): sha(library)}
    def visit(arm, role, index):
        job = logical(cell, arm, role, index, args.out)
        session['logical_jobs'].append(job); save(args.out / 'session.json', session)
        if job['status'] == 'pass':
            receipt = json.loads(Path(job['final']['measurement']).read_text())
            if receipt['configuration']['inputs_sha256'] != cell['inputs_sha256']:
                raise ValueError('resampled process used different fixed inputs')
            if not strict_correctness(cell['model'], receipt['correctness'])[0]:
                raise ValueError('resampled numerical gate failed')
    for index, arms in enumerate(order()):
        for arm in arms:
            visit(arm, 'measurement', index)
    for arm in ARMS_BY_NAME.values():
        for index in range(2):
            visit(arm, 'correctness', index)
    session['status'] = 'complete'; save(args.out / 'session.json', session)
    rows = [aggregate_arm(cell, session, name,
            [j for j in session['logical_jobs'] if j['arm']['arm'] == name], True)
            for name in ARMS_BY_NAME]
    pids = [j['final']['pid'] for j in session['logical_jobs']]
    if len(pids) != 20 or len(set(pids)) != 20:
        raise ValueError('resampling requires twenty fresh processes per batch')
    if any(sha(p) != digest for p, digest in spec['frozen'].items()):
        raise ValueError('resampling inputs changed')
    save(args.out / 'result.json', dict(evidence='verified', status='passed',
         protocol='DT-ANCHOR TRT/TC resample; 3 timing and 5 correctness processes per arm',
         model=cell['model'], batch=cell['batch'], gpu=cell['gpu'], rows=rows,
         qualification='provisional_container_pid_namespace; four existing TRT/TC BF16 arms only'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor', type=Path)
    p.add_argument('--export', type=Path)
    p.add_argument('--dnn1-root', type=Path, default=Path('/root/TileMega-dnn1'))
    p.add_argument('--cpu-set', default='0-7')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--spec', type=Path)
    p.add_argument('--batch', type=int, choices=BATCHES)
    args = p.parse_args(); args.out = args.out.resolve()
    if args.execute:
        if args.spec is None or args.batch is None:
            p.error('--execute requires --spec and --batch')
        execute(args); return 0
    if args.anchor is None or args.export is None:
        p.error('queue preparation requires --anchor and --export')
    args.anchor = args.anchor.resolve(); args.export = args.export.resolve()
    spec = prepare(args); args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'spec.json', spec)
    gpu = spec['gpu']
    env = dict(CUDA_VISIBLE_DEVICES=str(gpu), TILEMEGA_DEVICE_INDEX=str(gpu),
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{gpu}.lock', TILEMEGA_NUMA_NODE='0',
               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
               PYTHONPATH=str(ROOT / 'python'), OMP_NUM_THREADS='8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    queue = args.out / 'queue'; queue.mkdir()
    jobs = []; previous = []
    for batch in BATCHES:
        name = f'dnn4-anchor-resample-b{batch}'
        command = ['taskset', '-c', args.cpu_set, sys.executable, str(Path(__file__).resolve()),
                   '--execute', '--spec', str(args.out / 'spec.json'), '--batch', str(batch),
                   '--out', '{attempt_dir}/resample']
        jobs.append(dict(name=name, gpu=True, timing=True, after=previous, timeout_s=144000,
                         needs_free_mib=8192, command=command, cwd=str(ROOT), env=env))
        previous = [name]
    save(queue / 'queue_resample.json', jobs)
    with (args.out / 'dispatch.log').open('w') as stream:
        subprocess.run([str(ROOT / 'scripts/dnn4_guard_python.py'), '-m',
            'tilemega.bench.scheduler', '--queue-dir', str(queue), '--out',
            str(args.out / 'scheduler'), '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'),
            '--deadline-hours', '96'], cwd=ROOT, env=dict(os.environ, **env),
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    state = json.loads((args.out / 'scheduler/state.json').read_text())
    if any(state[j['name']]['status'] != 'done' for j in jobs):
        raise ValueError('resampling scheduler did not complete all five batch cells')
    rows = []; receipts = []
    for job in jobs:
        attempt = Path(state[job['name']]['out'])
        guard = json.loads((attempt / 'guard_result.json').read_text())
        path = attempt / 'resample/result.json'
        result = json.loads(path.read_text())
        if guard['code'] != 0 or result['status'] != 'passed':
            raise ValueError('resampling lacks an accepted batch receipt')
        rows.extend(result['rows'])
        receipts.append(dict(path=str(path), sha256=sha(path),
                             guard_sha256=sha(attempt / 'guard_result.json')))
    with (args.out / 'baselines.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    save(args.out / 'result.json', dict(evidence='verified', status='passed', model=spec['model'],
         gpu=gpu, rows=len(rows), baseline_csv=str(args.out / 'baselines.csv'),
         baseline_sha256=sha(args.out / 'baselines.csv'), receipts=receipts,
         baseline_scope='four existing TRT/TC BF16 arms only'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
