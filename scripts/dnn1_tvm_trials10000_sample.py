#!/usr/bin/env python3
"""One guarded process of the independent fresh-DB TVM 10000/9000 campaign."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
PROFILE = 'dnn-classic-tvm-fresh10000-wall9000-v1'
WORKLOAD_ID = 'dnn-classic-v1'
MODELS = ('resnet18', 'resnet34', 'vgg16', 'alexnet', 'bert')
BATCHES = (1, 2, 4, 8, 16)
WORKLOAD = dict(id=WORKLOAD_ID, models=list(MODELS), batches=list(BATCHES))
BUDGET = dict(max_trials_global=10000, timeout_s=9000,
    max_trials_per_task=None, num_trials_per_iter=64,
    semantics='requested candidate budget; actual attempted/committed/success/failure counts retained')
COMPATIBILITY = dict(max_trials_global=2000, timeout_s=1800,
    reason='unchanged adapter admission tuple only; isolated tuning callback receives actual 10000/9000')
# Only these ids take their workload from the epoch manifest, and only they skip
# the AIT release; every other cell keeps the classic constants and gate.
SPEC_WORKLOADS = {'dnn3-v1': dict(epoch_root='runs/dnn3', release_gate='not_applicable_dnn3')}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def is_classic(value):
    return value.get('workload_id', WORKLOAD_ID) == WORKLOAD_ID


def workload_of(value):
    return WORKLOAD if is_classic(value) else value['workload']


def workload_id_of(value):
    return value.get('workload_id', WORKLOAD_ID)


def epoch_root(value, root=ROOT):
    return root / ('runs/dnn1' if is_classic(value) else SPEC_WORKLOADS[value['workload_id']]['epoch_root'])


def allowed_gpus(value):
    return range(7) if is_classic(value) else value['allowed_gpu_indices']


def release_required(value):
    if is_classic(value):
        return True
    need(value.get('workload_id') in SPEC_WORKLOADS and
         value.get('release_gate') == SPEC_WORKLOADS[value['workload_id']]['release_gate'],
         'unknown workload or release gate')
    return False


def validate_workload(cell):
    if is_classic(cell):
        need(cell.get('workload_id') == WORKLOAD_ID and cell.get('workload') == WORKLOAD and
             cell['model'] in MODELS and cell['batch'] in BATCHES,
             'the active classic five-model workload is required')
        return
    workload = cell.get('workload') or {}
    need(cell['workload_id'] in SPEC_WORKLOADS and workload.get('id') == cell['workload_id'] and
         cell['model'] in workload.get('models', ()) and cell['batch'] in workload.get('batches', ()),
         'the cell is outside its declared workload')


def validate_cell(cell):
    need(cell.get('profile') == PROFILE and cell.get('budget') == BUDGET,
         'the fixed fresh10000/9000 profile is required')
    validate_workload(cell)
    need(cell['gpu'] in allowed_gpus(cell), 'new jobs require explicitly assigned indices0–6')
    need(cell.get('fresh_database') is True and cell.get('official_cuda_graph') is False,
         'only fresh databases and uncaptured VM execution are authorized')
    need(not cell.get('prior_database') and not cell.get('prior_files') and not cell.get('library'),
         'prior DB/library reuse is forbidden')
    epoch = Path(cell['epoch']).resolve()
    need(epoch.is_relative_to(epoch_root(cell)) and
         Path(cell['cache']).resolve().is_relative_to(epoch / 'cache'),
         'cache must belong to this new independent epoch')
    need(cell['process_timeout_s'] >= 18000 and cell['cell_timeout_s'] > 5 * cell['process_timeout_s'],
         'outer timeouts must leave room beyond the 9000-second tuner')
    for name, digest in cell['protocol_sources'].items():
        need(sha(name) == digest, 'frozen experiment dependency changed: ' + name)
    for name, digest in cell['assets'].items():
        need(sha(name) == digest, 'fixed export/checkpoint/input asset changed: ' + name)
    need(sha(cell['asset_preparation']['path']) == cell['asset_preparation']['sha256'],
         'classic CPU asset proof changed')
    need(sha(cell['epoch_manifest']) == cell['epoch_manifest_sha256'], 'epoch manifest changed')
    if release_required(cell):
        need(sha(cell['release']) == cell['release_sha256'], 'AIT/push release changed')
    else:
        need(cell.get('release') is None and cell.get('release_sha256') is None,
             'this workload has no AIT/push release')


def recipe(adapter_recipe, cell):
    return dict(adapter_recipe, schema=PROFILE, workload=workload_of(cell), official_cuda_graph=False,
        fresh_database=True, prior_database=None, tuning=dict(BUDGET),
        compatibility_entry=dict(COMPATIBILITY),
        experiment_driver_sha256=sha(__file__),
        epoch_manifest_sha256=cell['epoch_manifest_sha256'])


@contextlib.contextmanager
def tuning_affinity(cell):
    """Hold the tuner's CPU mask on this thread only while it spawns and waits.

    sched_setaffinity(0) changes the calling thread, and a forked child inherits
    that thread's mask, so the tuner and its builders run on the declared cores
    while torch workers and the CUDA runtime threads keep the card's cores.
    """
    cpus = cell.get('tuning_cpu_affinity')
    if not cpus:
        yield
        return
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        need(os.sched_getaffinity(0) == set(cpus), 'tuner CPU mask was not applied')
        yield
    finally:
        os.sched_setaffinity(0, previous)


def tuning_callback(original, cell):
    """The only budget override; it never changes rules, per-task or batch defaults."""
    def fresh(export_path, precision, batch, directory, *_compatibility_args, **_kwargs):
        directory = Path(directory).resolve()
        need(precision == 'bf16' and batch == cell['batch'], 'tuner cell/precision mismatch')
        need(directory.is_relative_to(Path(cell['cache']).resolve()), 'tune directory escapes fresh cache')
        need(not directory.exists(), 'retain existing tuning data; a new cohort needs a new cache')
        with tuning_affinity(cell):
            database, result = original(export_path, precision, batch, directory,
                BUDGET['max_trials_global'], BUDGET['timeout_s'], cell['model'] == 'bert')
        need(result['maximum_trials'] == 10000 and result['timeout_s'] == 9000,
             'actual tuning callback did not receive the declared budget')
        request = read(directory / 'request.json')
        need(request['max_trials_global'] == 10000, 'actual worker request budget differs')
        raw_receipt = directory / 'receipt.json'
        augmented = dict(result, requested_total_trials=10000, initial_database_records=0,
            fresh_database=True, max_trials_per_task=None, num_trials_per_iter=64,
            actual_worker_request=str(directory / 'request.json'),
            actual_worker_request_sha256=sha(directory / 'request.json'),
            original_tuning_receipt=str(raw_receipt), original_tuning_receipt_sha256=sha(raw_receipt))
        if cell.get('tuning_cpu_affinity'):
            augmented['tuning_cpu_affinity'] = cell['tuning_cpu_affinity']
        save(directory / 'fresh_budget_contract.json', augmented)
        return database, augmented
    return fresh


def sample(options):
    cell = read(options.cell)
    options.workload_id = workload_id_of(cell)
    validate_cell(cell)
    need(options.correctness_only == (options.round_index >= 3), 'fixed P-DNN/P5 process roles required')
    from tilemega.dnn import baselines as baseline
    baseline.require_guard()
    need(os.environ.get('TILEMEGA_DEVICE_INDEX') == str(cell['gpu']) and cell['gpu'] in allowed_gpus(cell),
         'wrong physical GPU or index outside the assigned0–6 pool')
    import torch
    from tilemega.dnn import tvm_baseline as adapter
    torch.set_num_threads(4)
    options.out.mkdir(parents=True, exist_ok=False)
    guard = options.out / 'guard.jsonl'
    baseline.observe(guard, 'before-initialization')
    export = Path(cell['export'])
    program, _manifest = baseline.load_export(export)
    args, kwargs = baseline.make_inputs(program, cell['batch'], 20261007)
    ref_args, ref_kwargs = baseline.convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        reference = [x.detach().clone() for x in baseline.leaves(baseline.portable_module(
            copy.deepcopy(program.module()), dtype=torch.float32)(*ref_args, **ref_kwargs))]
    tvm, _, _ = adapter._load_tvm()
    original_recipe, original_tuning = adapter.RECIPE, adapter._run_tuning
    adapter.RECIPE = recipe(original_recipe, cell)
    adapter._run_tuning = tuning_callback(original_tuning, cell)
    metadata = dict(input_seed=20261007, inputs_sha256=baseline.inputs_digest(args, kwargs)[0],
        reference_outputs_sha256=baseline.inputs_digest(reference, {})[0],
        experiment_profile=PROFILE, experiment_budget=dict(BUDGET),
        workload_id=workload_id_of(cell), workload=workload_of(cell),
        compatibility_entry=dict(COMPATIBILITY), fresh_database=True, prior_database=None,
        official_cuda_graph=False, experiment_cell_sha256=sha(options.cell),
        experiment_driver_sha256=sha(__file__), epoch_manifest_sha256=cell['epoch_manifest_sha256'])
    need(metadata['inputs_sha256'] == cell['expected_inputs_sha256'],
         'fixed generated inputs differ from the original formal25 digest')
    try:
        with torch.inference_mode(), tvm.transform.PassContext(config={'relax.backend.use_cuda_graph': False}):
            run = adapter.tvm_runner(export, program, args, kwargs, 'bf16', cell['batch'],
                options.out, metadata, work_dir=Path(cell['cache']),
                max_trials_global=COMPATIBILITY['max_trials_global'], timeout_s=COMPATIBILITY['timeout_s'])
            for _ in range(50):
                run()
            torch.cuda.synchronize()
            actual = [x.detach().clone() for x in baseline.leaves(run())]
            torch.cuda.synchronize()
            correctness = baseline.numerical_metrics(actual, reference, cell['model'])
            baseline.correctness_gate(cell['model'], correctness)
            if cell.get('tuning_cpu_affinity'):
                # The VM forward is timed on the card's cores, never the tuner's.
                metadata['measurement_cpu_affinity'] = sorted(os.sched_getaffinity(0))
                need(metadata['measurement_cpu_affinity'] == sorted(cell['cpu_affinity']),
                     'timed forward left the card CPU affinity')
            baseline.observe(guard, 'before-measurement')
            values = [] if options.correctness_only else baseline.timed(run, 0, 200)
            baseline.observe(guard, 'after-measurement')
    finally:
        adapter.RECIPE, adapter._run_tuning = original_recipe, original_tuning
    validate_cell(cell)
    diag = metadata['tvm_diag']
    need(diag['cache_identity']['recipe']['schema'] == PROFILE and
         diag['tuning']['maximum_trials'] == 10000 and diag['tuning']['timeout_s'] == 9000 and
         diag['tuning'].get('initial_database_records') == 0, 'compiled VM has another tuning identity')
    result = dict(schema_version=1, evidence='verified', system='tvm', model=cell['model'],
        workload_id=workload_id_of(cell),
        batch=cell['batch'], precision='bf16', pid=os.getpid(), round_index=options.round_index,
        gpu_index=int(baseline.require_guard()[0]), correctness_passed=True,
        process_role='correctness' if options.correctness_only else 'measurement',
        timing_status='not_measured' if options.correctness_only else 'measured',
        correctness=correctness, configuration=metadata, versions=baseline.versions(),
        exported_program_sha256=sha(export / 'exported_program.pt2'),
        source_checkpoint_sha256=sha(export / 'checkpoint/model.safetensors'),
        actual_outputs_sha256=baseline.inputs_digest(actual, {})[0], samples_ms=values,
        median_ms=statistics.median(values) if values else None,
        p90_ms=sorted(values)[179] if values else None,
        measurement=dict(warmup=50, iterations=200, clock='CUDA events'),
        guard_log=str(guard.resolve()), driver_sha256=sha(__file__),
        experiment_budget=dict(BUDGET), fresh_database=True, prior_database=None)
    save(options.out / 'measurements.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--round-index', type=int, choices=range(5), required=True)
    parser.add_argument('--correctness-only', action='store_true')
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError('retain the original process output')
    try:
        sample(options)
    except Exception as error:
        save(options.out / 'failure.json', dict(system='tvm', status='failed',
            failure_kind=getattr(error, 'failure_kind', 'infrastructure'), error=repr(error),
            traceback=traceback.format_exc(), experiment_profile=PROFILE,
            workload_id=getattr(options, 'workload_id', WORKLOAD_ID),
            experiment_budget=dict(BUDGET), compatibility_entry=dict(COMPATIBILITY),
            fresh_database=True, driver_sha256=sha(__file__), time=time.time()))
        raise


if __name__ == '__main__':
    main()
