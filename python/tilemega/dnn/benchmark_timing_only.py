"""Guarded forward timing using frozen inputs; no numerical gate is run."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from .baselines import sha, timed
from baseline_common import guarded_main, observe, require_guard, versions
from tilemega.build.identity import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'export', 'bridge', 'inputs', 'selection-config', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=(1,), default=1)
    parser.add_argument('--executor', choices=('L1',), default='L1')
    parser.add_argument('--warmup', type=int, choices=(50,), default=50)
    parser.add_argument('--iterations', type=int, choices=(200,), default=200)
    parser.add_argument('--round-index', type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    gpu_index = int(require_guard()[0])
    if gpu_index not in (0, 1, 2, 3, 4, 7):
        raise ValueError('GPU6 excluded; GPU5 functional-only')
    selection = json.loads(args.selection_config.read_text())
    identity_path = Path(str(args.library) + '.identity.json')
    identity = verify(args.library)
    if (selection['artifact_id'] != identity['artifact_id'] or
            selection['binary_sha256'] != sha(args.library) or
            selection['identity_sha256'] != sha(identity_path) or
            selection['source_head'] != identity['source']['head']):
        raise ValueError('selected frozen forward artifact differs')
    if identity['trace'] or any(k.startswith(('TILEMEGA_TRACE', 'TILEMEGA_PAGE_TRACE',
            'TILEMEGA_SM90_DENSE_BOUNDARY')) and str(v).strip() not in ('0', 'false')
            for k, v in identity['defines'].items()):
        raise ValueError('instrumented artifacts cannot enter closeout timing')
    import torch
    from safetensors.torch import load_file
    from .check import NativeForward
    manifest = json.loads((args.export / 'manifest.json').read_text())
    if not manifest.get('timing_eligible', manifest.get('accuracy_eligible', False)):
        raise ValueError('export is not timing eligible')
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / 'selected-config.json'
    if receipt.exists():
        raise FileExistsError('retain prior measurement evidence')
    record = dict(manifest=str(args.selection_config.resolve()),
        manifest_sha256=sha(args.selection_config), driver_sha256=sha(__file__),
        model=manifest['model'], batch=args.batch, fresh_process=args.round_index,
        stage='measure', e3_validated=False, selection=selection['selection'],
        artifact_id=identity['artifact_id'], source_head=selection['source_head'],
        protocol=dict(warmup=50, iterations=200, fresh_processes=3),
        command=[sys.executable, *sys.argv], started=time.time(),
        user_amendment=dict(scope='TileMega dense25, MoE5 and five DNN E2E; frozen external baselines',
            solver_gate='not_met', c1='not_performed', c2='not_performed', hf='not_performed',
            full_ctest='not_performed', external_baselines='not_remeasured'))
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    measurement_out = args.out / 'measurements'
    measurement_out.mkdir()
    guard_path = measurement_out / 'guard.jsonl'
    observe(guard_path, 'before-initialization')
    inputs = load_file(str(args.inputs), device='cuda')
    with torch.inference_mode(), NativeForward(args.library, args.export, args.bridge, args.batch) as native:
        if set(native.inputs) != set(inputs):
            raise ValueError('timed input names differ from the native forward signature')
        # Same constructor/bindings/input copies as NativeForward.__call__.
        # Calling __call__ itself would run the deliberately omitted L1/L2 gate.
        for name in native.inputs:
            value = inputs[name]
            if value.shape != native.tensors[name].shape or value.dtype != native.tensors[name].dtype:
                raise ValueError('input shape/dtype differs from the bound forward plan: ' + name)
            native.tensors[name].copy_(value)
        mode = 1
        stream = torch.cuda.current_stream().cuda_stream
        run = lambda: native.plan.launch(0, mode, stream)
        observe(guard_path, 'before-measurement')
        values = timed(run, args.warmup, args.iterations)
        torch.cuda.synchronize()
        observe(guard_path, 'after-measurement')
        execution = native.plan.execution_identity(mode)
    ordered = sorted(values)
    result = dict(schema_version=1, evidence='verified', system='tilemega', model=manifest['model'],
        batch=args.batch, precision='bf16', executor=args.executor, round_index=args.round_index,
        pid=os.getpid(), gpu_index=gpu_index, versions=versions(),
        artifact_id=identity['artifact_id'], execution_identity=execution,
        input_sha256=sha(args.inputs), numerical_gate_sha256=None,
        numerical_validation='not requested or run in this closeout timing process',
        numeric_validation='not_performed',
        selection_config=str(args.selection_config.resolve()), selection_config_sha256=sha(args.selection_config),
        selected_source_head=selection['source_head'], scope_expansion='frozen SM90R1 DNN selection reused for SM90R2 closeout',
        source_checkpoint_sha256=sha(args.export / 'checkpoint/model.safetensors'),
        library_path=str(args.library.resolve()), identity_path=str(identity_path.resolve()),
        identity=identity, input_path=str(args.inputs.resolve()), export_path=str(args.export.resolve()),
        bridge_path=str(args.bridge.resolve()), bridge_sha256=sha(args.bridge),
        exported_program_sha256=sha(args.export / 'exported_program.pt2'),
        random_weights=manifest.get('random_weights', False),
        accuracy_eligible=False, runner_sha256=sha(__file__),
        native_forward_source_sha256=sha(Path(__file__).with_name('check.py')),
        timing_helper_source_sha256=sha(Path(__file__).with_name('baselines.py')),
        measurement=dict(clock='CUDA events', warmup=args.warmup, iterations=args.iterations,
            cuda_graph=False, cuda_graph_reason='host launch tickets advance each inference; static replay would reuse synchronization epochs',
            scope='full model forward; input and weight transfers excluded'),
        median_ms=statistics.median(values), p90_ms=ordered[math.ceil(0.9 * len(ordered)) - 1],
        samples_ms=values, guard_log=str(guard_path.resolve()))
    (measurement_out / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'execution.json').write_text(json.dumps(dict(record, exit_code=0, ended=time.time()), indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples_ms'}), flush=True)


if __name__ == '__main__':
    guarded_main(main)
