#!/usr/bin/env python3
"""Compare official Relax CUDA graphs using a frozen recipe6 tuning database.

This separate cache/receipt epoch never modifies its source database. An optional
fixed 8000-trial/7200s experiment creates a fresh database before graph comparison.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.dnn import baselines as baseline
from tilemega.dnn import tvm_baseline as adapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def database_identity(compiled):
    receipt = json.loads(compiled.read_text())
    if receipt['identity']['recipe']['schema'] not in (6, 'recipe6-graph-diagnostic-v2'):
        raise ValueError('the source must be a versioned recipe6 or bounded diagnostic receipt')
    tune = Path(receipt['diagnostics']['tuning']['work_dir'])
    files = {str(path): sha(path) for path in sorted(tune.glob('database_*.json'))}
    if not files:
        raise ValueError('source database is missing')
    return receipt, tune, dict(compiled_receipt=str(compiled),
        compiled_receipt_sha256=sha(compiled), database_files=files)


def sample(options):
    import torch
    torch.set_num_threads(4)
    baseline.require_guard()
    options.out.mkdir(parents=True, exist_ok=False)
    source, tune, provenance = database_identity(options.compiled.resolve())
    identity = source['identity']
    if identity['batch'] != options.batch or identity['precision'] != 'bf16':
        raise ValueError('source database cell differs')
    export = ROOT / 'runs/dnn1/exports' / options.model
    if sha(export / 'exported_program.pt2') != identity['exported_program_sha256']:
        raise ValueError('export differs from frozen tuning source')
    guard_log = options.out / 'guard.jsonl'
    baseline.observe(guard_log, 'before-initialization')
    program, manifest = baseline.load_export(export)
    args, kwargs = baseline.make_inputs(program, options.batch, 20261007)
    reference_args, reference_kwargs = baseline.convert_inputs(args, kwargs, dtype=torch.float32)
    with torch.inference_mode():
        reference = [x.detach().clone() for x in baseline.leaves(baseline.portable_module(
            copy.deepcopy(program.module()), dtype=torch.float32)(*reference_args, **reference_kwargs))]
    tvm, _, _ = adapter._load_tvm()
    from tvm.s_tir import meta_schedule as ms
    budget = (dict(max_trials_global=8000, timeout_s=7200,
                   search_strategy='evolutionary', cost_model='xgb') if options.retune else
              identity['recipe']['tuning'])
    adapter.RECIPE = dict(adapter.RECIPE, schema='recipe6-graph-diagnostic-v2',
        official_cuda_graph=options.graph, source_database=provenance,
        fresh_bounded_retune=options.retune, tuning=budget,
        diagnostic_driver_sha256=sha(__file__))
    original_tuning = adapter._run_tuning

    def retained_database(*args, **kwargs):
        if options.retune:
            export_path, precision, batch, directory = args[:4]
            return original_tuning(export_path, precision, batch, directory,
                max_trials_global=8000, timeout_s=7200, bert=options.model == 'bert')
        return ms.database.JSONDatabase(work_dir=str(tune), allow_missing=False), dict(
            source['diagnostics']['tuning'], reused_frozen_database=True,
            source_database=provenance, diagnostic_tuning_trials=0)

    adapter._run_tuning = retained_database
    original_compile = tvm.compile

    @tvm.instrument.pass_instrument
    class CaptureGraphPass:
        def run_after_pass(self, mod, info):
            if 'RewriteCUDAGraph' in info.name:
                text = mod.script()
                (options.out / 'cuda_graph_rewrite.py').write_text(text)
                capture['run_or_capture_calls'] = text.count('vm.builtin.cuda_graph.run_or_capture')
                capture['cached_alloc_calls'] = text.count('vm.builtin.cuda_graph.get_cached_alloc')

    capture = {}

    def compile_with_graph(*args, **kwargs):
        with tvm.transform.PassContext(config={'relax.backend.use_cuda_graph': options.graph},
                                       instruments=[CaptureGraphPass()]):
            return original_compile(*args, **kwargs)

    tvm.compile = compile_with_graph
    metadata = dict(source_database=provenance, official_cuda_graph=options.graph,
        input_seed=20261007, inputs_sha256=baseline.inputs_digest(args, kwargs)[0],
        reference_outputs_sha256=baseline.inputs_digest(reference, {})[0])
    try:
        with torch.inference_mode():
            run = adapter.tvm_runner(export, program, args, kwargs, 'bf16', options.batch,
                options.out, metadata, work_dir=options.cache, max_trials_global=2000, timeout_s=1800)
            for _ in range(50):
                run()
            torch.cuda.synchronize()
            actual = [x.detach().clone() for x in baseline.leaves(run())]
            torch.cuda.synchronize()
            correctness = baseline.numerical_metrics(actual, reference, manifest['model'])
            baseline.correctness_gate(manifest['model'], correctness)
            baseline.observe(guard_log, 'before-measurement')
            values = [] if options.correctness_only else baseline.timed(run, 0, 200)
            baseline.observe(guard_log, 'after-measurement')
    finally:
        tvm.compile = original_compile
    # Cached processes bind the original compile's pass receipt by its cache entry.
    diag = json.loads((options.out / 'tvm_diag.json').read_text())
    cache_entry = Path(diag['cache_directory'])
    if capture:
        (cache_entry / 'graph_rewrite_receipt.json').write_text(json.dumps(capture, indent=2) + '\n')
    else:
        capture = json.loads((cache_entry / 'graph_rewrite_receipt.json').read_text())
    if options.graph and capture.get('run_or_capture_calls', 0) == 0:
        raise ValueError('official graph pass produced no capture region')
    if database_identity(options.compiled.resolve())[2] != provenance:
        raise ValueError('source database changed during diagnostic')
    result = dict(schema=1, evidence='verified', system='tvm_graph_diagnostic',
        model=options.model, batch=options.batch, precision='bf16', pid=os.getpid(),
        gpu_index=int(baseline.require_guard()[0]), official_cuda_graph=options.graph,
        fresh_bounded_retune=options.retune,
        source_database=provenance, configuration=metadata, graph_rewrite=capture,
        correctness_passed=True, correctness=correctness,
        actual_outputs_sha256=baseline.inputs_digest(actual, {})[0],
        process_role='correctness' if options.correctness_only else 'measurement',
        samples_ms=values, median_ms=statistics.median(values) if values else None,
        p90_ms=sorted(values)[179] if values else None,
        measurement=dict(warmup=50, iterations=200, clock='CUDA events'),
        driver_sha256=sha(__file__), guard_log=str(guard_log.resolve()))
    (options.out / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'], required=True)
    p.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16], required=True)
    p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--graph', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--retune', action='store_true',
                   help='new isolated 8000-trial/7200s database; source database stays read-only')
    p.add_argument('--correctness-only', action='store_true')
    sample(p.parse_args())


if __name__ == '__main__':
    main()
