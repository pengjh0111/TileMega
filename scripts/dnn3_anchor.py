#!/usr/bin/env python3
"""Check one Phase Z model/card against its frozen B1 and B16 BF16 arms."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tilemega.build.identity import sha
from dnn1_baselines import ARMS, measurement_clocks, receipt_problem, strict_correctness

ARMS_BY_NAME = {a['arm']: a for a in ARMS if a['precision'] == 'bf16' and a['system'] != 'tvm'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fastest(rows, model, batch):
    eligible = [r for r in rows if r['model'] == model and int(r['batch']) == batch
                and r['precision'] == 'bf16' and r['status'] == 'pass']
    if not eligible or any(not math.isfinite(float(r['median_ms'])) or
                           float(r['median_ms']) <= 0 for r in eligible):
        raise ValueError('missing or invalid frozen BF16 baseline')
    row = min(eligible, key=lambda r: float(r['median_ms']))
    if row['arm'] not in ARMS_BY_NAME:
        raise ValueError('fastest frozen BF16 arm needs a separate baseline adapter: ' + row['arm'])
    return row


def decision(reference, measured):
    if any(not math.isfinite(v) or v <= 0 for v in (reference, measured)):
        raise ValueError('anchor requires finite positive medians')
    difference = measured / reference - 1
    return dict(relative_difference=difference,
                action='retain_frozen' if abs(measured - reference) <= .02 * reference + 1e-12
                else 'resample_trt_tc_all_batches')


def frozen_measurement(row, dnn1_root):
    source = row['source']
    if source.startswith('['):
        paths = [Path(p) for p in json.loads(source)]
    else:
        session_path = Path(source)
        if not session_path.is_absolute():
            session_path = dnn1_root / session_path
        session = json.loads(session_path.read_text())
        if 'logical_jobs' in session:
            paths = [Path(job['final']['measurement']) for job in session['logical_jobs']
                     if job['arm']['arm'] == row['arm'] and job['role'] == 'measurement'
                     and job['status'] == 'pass']
        else:
            paths = [Path(job['measurement']) for job in session['measurement']
                     if job['arm'] == row['arm'] and job['returncode'] == 0]
    for path in paths:
        value = json.loads(path.read_text())
        if value.get('timing_status') == 'measured' and value.get('correctness_passed'):
            return path, value
    raise ValueError('frozen baseline lacks a passed process identity')


def prepare(args):
    finals = [json.loads(p.read_text()) for p in args.final_results]
    if not finals or any(r.get('status') != 'passed' or r.get('model') != args.model or
                         r.get('gpu') != args.device or
                         r.get('protocol') != 'P-SEL; winner P-CORR; P-DNN' for r in finals):
        raise ValueError('anchor requires completed Phase Z cells for this model/card')
    baseline = ROOT / 'docs/results' / ('DNN1_baselines.csv' if args.model in
                                      ('resnet18', 'bert') else 'DNN3_baselines.csv')
    with baseline.open() as stream:
        rows = list(csv.DictReader(stream))
    archive = sha(args.export / 'exported_program.pt2')
    checkpoint = sha(args.export / 'checkpoint/model.safetensors')
    manifest = json.loads((args.export / 'manifest.json').read_text())
    if manifest['model'] != args.model or manifest['artifacts']['exported_program.pt2'] != archive:
        raise ValueError('anchor export model/archive differs from the manifest')
    frozen = {str(p): sha(p) for p in [Path(__file__), baseline, *args.final_results,
              args.export / 'manifest.json', args.export / 'exported_program.pt2',
              args.export / 'checkpoint/model.safetensors',
              ROOT / 'scripts/dnn1_baselines.py',
              *sorted((ROOT / 'python/tilemega/dnn').glob('*.py'))]}
    anchors = []
    for batch in (1, 16):
        row = fastest(rows, args.model, batch)
        path, receipt = frozen_measurement(row, args.dnn1_root)
        if (receipt['model'] != args.model or receipt['batch'] != batch or
                receipt['exported_program_sha256'] != archive or
                receipt['source_checkpoint_sha256'] != checkpoint):
            raise ValueError('current export/checkpoint differs from the frozen baseline')
        frozen[str(path)] = sha(path)
        anchors.append(dict(batch=batch, arm=ARMS_BY_NAME[row['arm']],
                            frozen_ms=float(row['median_ms']), source=str(path),
                            inputs_sha256=receipt['configuration']['inputs_sha256']))
    return dict(model=args.model, gpu=args.device, export=str(args.export),
                exported_program_sha256=archive, source_checkpoint_sha256=checkpoint,
                anchors=anchors, frozen=frozen)


def execute(args):
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    spec = json.loads(args.spec.read_text())
    if spec['model'] != args.model or spec['gpu'] != args.device:
        raise ValueError('anchor spec model/card differs')
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(evidence='verified', protocol='P-ANCHOR', model=args.model, gpu=args.device,
                  qualification='provisional_container_pid_namespace', status='partial', anchors=[])
    for row in spec['anchors']:
        folder = args.out / f"b{row['batch']}"
        arm = row['arm']
        command = [str(ROOT / 'scripts/dnn3_idle_floor_python.py'),
                   str(ROOT / 'python/tilemega/dnn/baselines.py'),
                   '--export', spec['export'], '--batch', str(row['batch']),
                   '--system', arm['system'], '--precision', 'bf16', '--seed', '20261007',
                   '--warmup', '50', '--iterations', '200', '--out', str(folder)]
        if arm['system'] == 'torchcompile':
            command += ['--compile-mode', arm['compile_mode'], '--fullgraph', 'true']
            if arm['freezing']:
                command.append('--freezing')
        else:
            command += ['--engine-cache', str(args.engine_cache)]
        with (args.out / f"b{row['batch']}.log").open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=os.environ, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=7200)
        path = folder / 'measurements.json'
        receipt = json.loads(path.read_text())
        cell = dict(spec, batch=row['batch'])
        reason = receipt_problem(cell, arm, 'measurement', receipt)
        passed, correctness_reason = strict_correctness(args.model, receipt['correctness'])
        if reason or not passed or receipt['configuration']['inputs_sha256'] != row['inputs_sha256']:
            raise ValueError('anchor identity/correctness failed: ' + reason + correctness_reason)
        result['anchors'].append(dict(batch=row['batch'], arm=arm['arm'],
            frozen_ms=row['frozen_ms'], median_ms=receipt['median_ms'], pid=receipt['pid'],
            measurement=str(path), sha256=sha(path), clocks=measurement_clocks(receipt, args.device),
            **decision(row['frozen_ms'], receipt['median_ms'])))
        save(args.out / 'result.json', result)
    if len({r['pid'] for r in result['anchors']}) != 2:
        raise ValueError('anchors require two fresh processes')
    if any(sha(p) != digest for p, digest in spec['frozen'].items()):
        raise ValueError('anchor inputs changed')
    result['action'] = ('retain_frozen' if all(r['action'] == 'retain_frozen' for r in result['anchors'])
                        else 'resample_trt_tc_all_batches')
    if result['action'] != 'retain_frozen':
        save(args.out / 'resampling_required.json', dict(model=args.model, gpu=args.device,
             batches=[1, 2, 4, 8, 16], arms=list(ARMS_BY_NAME), tvm='retain_without_retuning',
             reason='DT-ANCHOR absolute difference exceeds 2%; frozen CSV remains unchanged'))
    result['status'] = 'passed'
    save(args.out / 'result.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('resnet18', 'resnet50', 'vgg16', 'bert'), required=True)
    p.add_argument('--device', type=int, choices=range(4), required=True)
    p.add_argument('--export', type=Path)
    p.add_argument('--final-results', type=Path, nargs='+')
    p.add_argument('--dnn1-root', type=Path, default=Path('/root/TileMega-dnn1'))
    p.add_argument('--engine-cache', type=Path, default=ROOT / 'runs/dnn3/anchor_engine_cache')
    p.add_argument('--cpu-set', default='128-135')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spec', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(); args.out = args.out.resolve()
    if args.execute:
        if args.spec is None:
            p.error('--execute requires --spec')
        execute(args)
        return 0
    if args.export is None or not args.final_results:
        p.error('queue preparation requires --export and --final-results')
    args.export = args.export.resolve(); args.engine_cache = args.engine_cache.resolve()
    args.final_results = [path.resolve() for path in args.final_results]
    spec = prepare(args)
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'spec.json', spec)
    queue = args.out / 'queue'; queue.mkdir()
    env = dict(CUDA_VISIBLE_DEVICES=str(args.device), TILEMEGA_DEVICE_INDEX=str(args.device),
               TILEMEGA_GPU_LOCK=f'/tmp/tilemega_gpu{args.device}.lock', TILEMEGA_NUMA_NODE='0',
               TILEMEGA_GPU_PID_MODE='namespace_limited', TILEMEGA_NUMA_CONCURRENCY_TRIAL='1',
               PYTHONPATH=str(ROOT / 'python'))
    command = ['taskset', '-c', args.cpu_set, sys.executable, str(Path(__file__).resolve()),
               '--execute', '--spec', str(args.out / 'spec.json'), '--model', args.model,
               '--device', str(args.device), '--engine-cache', str(args.engine_cache),
               '--out', '{attempt_dir}/anchor']
    save(queue / 'queue_anchor.json', [dict(name='dnn3-phase-z-anchor', gpu=True, timing=True,
         timeout_s=18000, needs_free_mib=8192, command=command, cwd=str(ROOT), env=env)])
    with (args.out / 'dispatch.log').open('w') as stream:
        subprocess.run([str(ROOT / 'scripts/dnn3_idle_floor_python.py'), '-m',
            'tilemega.bench.scheduler', '--queue-dir', str(queue), '--out',
            str(args.out / 'scheduler'), '--policy', str(ROOT / 'configs/bench/sm90r2_guard_policy.json'),
            '--deadline-hours', '6'], cwd=ROOT, env=dict(os.environ, **env),
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    state = json.loads((args.out / 'scheduler/state.json').read_text())['dnn3-phase-z-anchor']
    save(args.out / 'result.json', state)
    return int(state['status'] != 'done')


if __name__ == '__main__':
    raise SystemExit(main())
