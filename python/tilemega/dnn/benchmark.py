"""Guarded full-model forward timing after an identified numerical gate."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from .baselines import sha, timed
from baseline_common import guarded_main, observe, require_guard, versions


def validate_gate(receipt, artifact_id, inputs, export_dir, bridge):
    if not receipt.get('passed') or receipt.get('artifact_id') != artifact_id:
        raise ValueError('forward timing requires a passed numerical receipt for this artifact')
    if receipt.get('synthetic_weights'):
        raise ValueError('a synthetic replacement state cannot certify the stored random checkpoint')
    if receipt.get('input_tensors_sha256') != sha(inputs):
        raise ValueError('numerical receipt and timed input identity differ')
    if receipt.get('archive_sha256') != sha(Path(export_dir) / 'exported_program.pt2'):
        raise ValueError('numerical receipt and timed export archive differ')
    if receipt.get('source_checkpoint_sha256') != sha(Path(export_dir) / 'checkpoint/model.safetensors'):
        raise ValueError('numerical receipt and timed source checkpoint differ')
    if receipt.get('bridge_sha256') != sha(bridge):
        raise ValueError('numerical receipt and timed bridge differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--executor', choices=('L1', 'L2'), default='L2')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.warmup < 1 or args.iterations < 1:
        parser.error('warmup and iterations must be positive')
    import torch
    from safetensors.torch import load_file
    from .check import NativeForward
    identity = json.loads(Path(str(args.library) + '.identity.json').read_text())
    validate_gate(json.loads(args.gate.read_text()), identity['artifact_id'], args.inputs, args.export, args.bridge)
    args.out.mkdir(parents=True, exist_ok=True)
    guard_path = args.out / 'guard.jsonl'
    observe(guard_path, 'before-initialization')
    inputs = load_file(str(args.inputs), device='cuda')
    with torch.inference_mode(), NativeForward(args.library, args.export, args.bridge, args.batch) as native:
        if set(native.inputs) != set(inputs):
            raise ValueError('timed input names differ from the native forward signature')
        # NativeForward verifies finite outputs and L1/L2 bitwise equality once;
        # its per-call copies and checks remain outside the timing region.
        reference = native([inputs[name] for name in native.inputs])
        mode = {'L1': 1, 'L2': 2}[args.executor]
        stream = torch.cuda.current_stream().cuda_stream
        run = lambda: native.plan.launch(0, mode, stream)
        observe(guard_path, 'before-measurement')
        values = timed(run, args.warmup, args.iterations)
        torch.cuda.synchronize()
        if any(not torch.equal(value, native.tensors[name])
               for name, value in zip(native.outputs, reference)):
            raise AssertionError('timed forward output differs from the checked execution')
        observe(guard_path, 'after-measurement')
        execution = native.plan.execution_identity(mode)
    manifest = json.loads((args.export / 'manifest.json').read_text())
    ordered = sorted(values)
    result = dict(schema_version=1, evidence='verified', system='tilemega', model=manifest['model'],
        batch=args.batch, precision='bf16', executor=args.executor, round_index=args.round_index,
        pid=os.getpid(), gpu_index=int(require_guard()), versions=versions(),
        artifact_id=identity['artifact_id'], execution_identity=execution,
        input_sha256=sha(args.inputs), numerical_gate_sha256=sha(args.gate),
        source_checkpoint_sha256=sha(args.export / 'checkpoint/model.safetensors'),
        library_path=str(args.library.resolve()), identity_path=str(Path(str(args.library)+'.identity.json').resolve()),
        identity=identity, input_path=str(args.inputs.resolve()), numerical_gate_path=str(args.gate.resolve()),
        export_path=str(args.export.resolve()),
        exported_program_sha256=manifest.get('artifacts', {}).get('exported_program.pt2'),
        random_weights=manifest.get('random_weights', False),
        accuracy_eligible=manifest.get('accuracy_eligible', False),
        measurement=dict(clock='CUDA events', warmup=args.warmup, iterations=args.iterations,
            cuda_graph=False, cuda_graph_reason='host launch tickets advance each inference; static replay would reuse synchronization epochs',
            scope='full model forward; input and weight transfers excluded'),
        median_ms=statistics.median(values), p90_ms=ordered[math.ceil(0.9 * len(ordered)) - 1],
        samples_ms=values, guard_log=str(guard_path.resolve()))
    (args.out / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples_ms'}), flush=True)


if __name__ == '__main__':
    guarded_main(main)
