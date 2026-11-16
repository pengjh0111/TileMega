#!/usr/bin/env python3
"""Collect one explicitly unfrozen BF16 BERT batch in five fresh processes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from dnn1_baselines import measurement_clocks, receipt_problem, strict_correctness


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(specification, out):
    from tilemega.serving.baseline_common import require_guard
    gpu = int(require_guard()[0])
    spec = read(specification)
    if (spec['model'] != 'bert' or spec['batch'] not in (1,2,4,8,16) or spec['gpu'] != gpu
            or spec['roles'] != ['correctness'] + ['measurement'] * 3 + ['correctness']
            or spec['arm'] != dict(arm='torchcompile-max-autotune',
                system='torchcompile', precision='bf16', compile_mode='max-autotune',
                freezing=False, fullgraph=True)):
        raise ValueError('recheck requires the explicitly unfrozen arm and declared fresh process roles')
    if any(sha(path) != digest for path, digest in spec['frozen'].items()):
        raise ValueError('baseline inputs changed before collection')
    out.mkdir(parents=True, exist_ok=False)
    processes = []
    for index, role in enumerate(spec['roles']):
        authorization = read(ROOT / 'runs/dnn4/bert_priority/resume_allcards_20261112_v1/authorization.json')
        if authorization['status'] != 'active' or gpu not in authorization['allowed_gpu_indices']:
            raise RuntimeError('current authorization disallows this baseline process')
        folder = out / f'{index + 1}-{role}'
        command = ['taskset', '-c', spec['compile_cpu_set'] if index == 0 else spec['cpu_set'], sys.executable, '/root/python/guard_idle6.py',
            '/root/python/run_repo_script.py', '--entry',
            str(ROOT / 'python/tilemega/dnn/baselines.py'), '--export', spec['export'],
            '--batch', str(spec['batch']), '--system', 'torchcompile', '--precision', 'bf16',
            '--compile-mode', 'max-autotune', '--fullgraph', 'true',
            '--seed', '20261007', '--warmup', '50', '--iterations', '200',
            '--round-index', str(index), '--out', str(folder)]
        if role == 'correctness':
            command.append('--correctness-only')
        log = out / f'{index + 1}-{role}.log'
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=ROOT,
                env=dict(os.environ,TORCHINDUCTOR_CACHE_DIR=spec['inductor_cache'],TORCHINDUCTOR_FREEZING='0',TORCHINDUCTOR_COMPILE_THREADS='1'),
                stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=7200)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if code:
            raise RuntimeError(f'baseline {role} process {index + 1} failed: {log}')
        path = folder / 'measurements.json'
        value = read(path)
        problem = receipt_problem(spec, spec['arm'], role, value)
        passed, reason = strict_correctness('bert', value['correctness'])
        if (problem or not passed or not value['correctness_passed']
                or value['pid'] != child.pid or value['process_role'] != role
                or value['configuration']['inputs_sha256'] != spec['inputs_sha256']
                or value['configuration']['input_seed'] != 20261007
                or value['configuration']['compile_mode'] != 'max-autotune'
                or value['configuration']['freezing_requested'] is not False
                or value['configuration']['freezing'] is not False):
            raise ValueError('baseline identity/correctness differs: ' + problem + reason)
        samples = value['samples_ms']
        if role == 'measurement':
            if (len(samples) != 200 or any(not math.isfinite(x) or x <= 0 for x in samples)
                    or statistics.median(samples) != value['median_ms']):
                raise ValueError('baseline measurement lacks exactly 200 positive CUDA events')
        elif samples or value['timing_status'] != 'not_measured':
            raise ValueError('extra correctness process unexpectedly contributed timing')
        processes.append(dict(pid=child.pid, role=role, command=command, measurement=str(path),
            measurement_sha256=sha(path), log=str(log), log_sha256=sha(log),
            median_ms=value['median_ms'], p90_ms=value['p90_ms'],
            correctness=value['correctness'], actual_outputs_sha256=value['configuration']['actual_outputs_sha256'],
            reference_outputs_sha256=value['configuration']['reference_outputs_sha256'],
            versions=value['versions'], accuracy_eligible=value['accuracy_eligible'],
            random_weights=value['random_weights'], clocks=measurement_clocks(value, gpu)))
        save(out / 'progress.json', dict(status='partial', processes=processes, required=5))
    if len({row['pid'] for row in processes}) != 5:
        raise ValueError('baseline did not use five fresh processes')
    if any(sha(path) != digest for path, digest in spec['frozen'].items()):
        raise ValueError('baseline inputs changed during collection')
    measured = [row for row in processes if row['role'] == 'measurement']
    medians = [row['median_ms'] for row in measured]
    elapsed = statistics.median(medians)
    save(out / 'result.json', dict(passed=True, status='passed', evidence='verified',
        model='bert', batch=spec['batch'], gpu=gpu, arm=spec['arm']['arm'],
        protocol='original baseline: timing3x50/200; FP32 correctness5 fresh processes',
        median_ms=elapsed, p90_ms=statistics.median(row['p90_ms'] for row in measured),
        process_range_ms=max(medians) - min(medians), process_medians_ms=medians,
        frozen_ms=spec['frozen_ms'], relative_difference=elapsed / spec['frozen_ms'] - 1,
        output_hash_stable=len({row['actual_outputs_sha256'] for row in processes}) == 1,
        reference_hash_stable=len({row['reference_outputs_sha256'] for row in processes}) == 1,
        correctness_passes='5/5', processes=processes,
        qualification='provisional_container_pid_namespace; inspect original guard receipts',
        specification=str(specification), specification_sha256=sha(specification),
        scope='User-requested all-batch BF16 baseline without freezing. Same export, '
              'checkpoint, input identity, seed, max-autotune/fullgraph/freezingFalse; '
              'compilation excluded and numerical gates unchanged. P90 is median '
              'of three process p90s as in original baseline protocol. Five-process '
              'output hash stability is recorded diagnostically. Frozen DNN4 target '
              'and historical TileMega measurements are not overwritten.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    execute(args.spec.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
