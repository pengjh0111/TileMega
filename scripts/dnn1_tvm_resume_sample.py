#!/usr/bin/env python3
"""Versioned TVM baseline resumption with immutable prior tuning evidence."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.dnn import baselines as baseline
from tilemega.dnn import tvm_baseline as adapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def prior_identity(spec):
    directory = spec.get('prior_database')
    if directory is None:
        return dict(files={}, database=None)
    directory = Path(directory)
    files = {str(p): sha(p) for name in ('database_*.json', '*_results.jsonl')
             for p in sorted(directory.glob(name))}
    if not any(Path(p).name == 'database_tuning_record.json' for p in files):
        raise ValueError('the explicit prior database has no retained tuning records')
    return dict(files=files, database=str(directory.resolve()),
                prior_receipt=str(directory / 'receipt.json') if (directory / 'receipt.json').exists() else None,
                prior_receipt_sha256=sha(directory / 'receipt.json') if (directory / 'receipt.json').exists() else None)


def seed_database(files, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for filename in files:
        name = Path(filename).name
        destination = directory / name if name.startswith('database_') else directory / 'prior_journals' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename, destination)


def sample(options):
    import torch
    torch.set_num_threads(4)
    baseline.require_guard()
    spec = json.loads(options.cell.read_text())
    options.out.mkdir(parents=True, exist_ok=False)
    provenance = prior_identity(spec)
    expected = spec.get('prior_files', {})
    if expected and provenance['files'] != expected:
        raise ValueError('prior tuning evidence changed after the resume manifest was prepared')
    export = Path(spec['export'])
    if sha(export / 'exported_program.pt2') != spec['exported_program_sha256']:
        raise ValueError('unchanged exported program required')
    guard = options.out / 'guard.jsonl'
    baseline.observe(guard, 'before-initialization')
    program, manifest = baseline.load_export(export)
    args, kwargs = baseline.make_inputs(program, spec['batch'], 20261007)
    ref_args, ref_kwargs = baseline.convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        reference = [x.detach().clone() for x in baseline.leaves(baseline.portable_module(
            copy.deepcopy(program.module()), dtype=torch.float32)(*ref_args, **ref_kwargs))]
    tvm, _, _ = adapter._load_tvm()
    from tvm.s_tir import meta_schedule as ms
    adapter.RECIPE = dict(adapter.RECIPE, schema='recipe6-cancellation-resume-v1',
        official_cuda_graph=False, resume_policy=spec['policy'], source_database=provenance,
        resume_driver_sha256=sha(__file__),
        tuning=dict(requested_total_trials=2000, epoch_timeout_s=1800,
                    trial_cap_semantics='default MetaSchedule batch rounding; actual counts reported',
                    historical_wall_time='retained separately; not counted as fresh epoch time',
                    search_strategy='evolutionary', cost_model='xgb'))
    original_tuning = adapter._run_tuning

    def tuning(export_path, precision, batch, directory, *unused):
        prior = (ms.database.JSONDatabase(work_dir=provenance['database'], allow_missing=False)
                 if provenance['database'] else None)
        before = adapter._count_database(prior) if prior is not None else dict(
            committed_trials=0, successful_trials=0, failed_trials=0)
        if spec['policy'] == 'reuse' or before['committed_trials'] >= 2000:
            if prior is None:
                raise ValueError('reuse requires an explicit completed database')
            return prior, dict(before, elapsed_s=0, deadline_hit=False, timeout_s=0,
                maximum_trials=2000, resumed_from=provenance, fresh_committed_trials=0,
                work_dir=provenance['database'], policy='read-only completed database')
        directory = Path(directory)
        seed_database(provenance['files'], directory)
        maximum = 2000 - before['committed_trials']
        database, result = original_tuning(export_path, precision, batch, directory,
            maximum, 1800, spec['model'] == 'bert')
        result.update(resumed_from=provenance, prior_counts=before,
            fresh_committed_trials=result['committed_trials'] - before['committed_trials'],
            maximum_new_trials=maximum, requested_total_trials=2000,
            historical_wall_time='prior effort preserved; new epoch separately capped at1800s')
        save(directory / 'receipt.json', result)
        return database, result

    adapter._run_tuning = tuning
    metadata = dict(input_seed=20261007, inputs_sha256=baseline.inputs_digest(args, kwargs)[0],
        reference_outputs_sha256=baseline.inputs_digest(reference, {})[0],
        resume_manifest_sha256=sha(options.cell), resume_driver_sha256=sha(__file__),
        source_database=provenance, official_cuda_graph=False)
    with torch.inference_mode(), tvm.transform.PassContext(config={'relax.backend.use_cuda_graph': False}):
        run = adapter.tvm_runner(export, program, args, kwargs, 'bf16', spec['batch'],
            options.out, metadata, work_dir=Path(spec['cache']), max_trials_global=2000, timeout_s=1800)
        for _ in range(50):
            run()
        torch.cuda.synchronize()
        actual = [x.detach().clone() for x in baseline.leaves(run())]
        torch.cuda.synchronize()
        correctness = baseline.numerical_metrics(actual, reference, spec['model'])
        baseline.correctness_gate(spec['model'], correctness)
        baseline.observe(guard, 'before-measurement')
        values = [] if options.correctness_only else baseline.timed(run, 0, 200)
        baseline.observe(guard, 'after-measurement')
    if prior_identity(spec) != provenance:
        raise ValueError('original database changed during resumption')
    result = dict(schema_version=1, evidence='verified', system='tvm', model=spec['model'],
        batch=spec['batch'], precision='bf16', pid=os.getpid(), round_index=options.round_index,
        gpu_index=int(baseline.require_guard()[0]), correctness_passed=True,
        process_role='correctness' if options.correctness_only else 'measurement',
        timing_status='not_measured' if options.correctness_only else 'measured',
        correctness=correctness, configuration=metadata, versions=baseline.versions(),
        exported_program_sha256=spec['exported_program_sha256'],
        source_checkpoint_sha256=sha(export / 'checkpoint/model.safetensors'),
        actual_outputs_sha256=baseline.inputs_digest(actual, {})[0],
        samples_ms=values, median_ms=statistics.median(values) if values else None,
        p90_ms=sorted(values)[179] if values else None,
        measurement=dict(warmup=50, iterations=200, clock='CUDA events'),
        guard_log=str(guard.resolve()), driver_sha256=sha(__file__))
    save(options.out / 'measurements.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--round-index', type=int, choices=range(5), required=True)
    parser.add_argument('--correctness-only', action='store_true')
    options = parser.parse_args()
    try:
        sample(options)
    except Exception as error:
        options.out.mkdir(parents=True, exist_ok=True)
        save(options.out / 'failure.json', dict(status='failed', time=time.time(),
            error=str(error), failure_kind=getattr(error, 'failure_kind', 'runtime'),
            traceback=traceback.format_exc(), driver_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    main()
