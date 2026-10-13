"""Locate the first observed per-image excess error in retained DNN buffers.

Run after check_generated --diagnostics on a build with --reuse none. This is
a diagnostic against that check's exported BF16 checkpoint and its FP32
promotion, not the dataset gate against original upstream FP32 weights.
"""
import argparse
import json
import math
from pathlib import Path

from tilemega.build.identity import digest, verify
from .gates import BF16_ERROR_FACTOR


def reference_aliases(nodes, buffer_names, captured_names):
    """Resolve surviving fused outputs, proving SimpleGate's paired halves."""
    aliases, terminal, chunks, halves = {}, {}, {}, {}
    order = {node['name']: index for index, node in enumerate(nodes)}
    unary = {'batch_norm', '_native_batch_norm_legit_no_training', 'relu', 'relu_',
             'hardtanh', 'hardtanh_', 'view', 'reshape', 'flatten', 'dropout',
             'clone', 'contiguous', 'tanh', 'gelu', 'silu'}
    for node in nodes:
        name, target = node['name'], node['target']
        op = target.split('.')[1] if target.startswith('aten.') else ''
        inputs = node.get('inputs', [])
        args = node.get('args', {}).get('v', [])
        if op == 'chunk' and len(args) >= 3 and inputs:
            if args[1] == {'t':'int','v':2} and args[2] == {'t':'int','v':1}:
                chunks[name] = inputs[0]
        if target in {'operator.getitem','<built-in function getitem>'} and inputs:
            if inputs[0] in chunks and len(args) == 2 and args[1].get('t') == 'int':
                halves[name] = (inputs[0], args[1]['v'])
        if name in buffer_names:
            aliases[name] = name
        elif (op in unary or target == 'operator.getitem') and inputs:
            if inputs[0] in aliases:
                aliases[name] = aliases[inputs[0]]
        elif op in {'add','add_'}:
            sources = [aliases[source] for source in inputs if source in aliases]
            if sources:
                aliases[name] = max(sources, key=lambda source: order.get(source,-1))
        elif op == 'mul' and len(inputs) == 2 and all(source in halves for source in inputs):
            left, right = (halves[source] for source in inputs)
            if left[0] == right[0] and {left[1],right[1]} == {0,1}:
                source = chunks[left[0]]
                if source in aliases:
                    aliases[name] = aliases[source]
        if name in aliases and name in captured_names:
            terminal[aliases[name]] = name
    return terminal


def summarize(intermediates, bridge):
    nodes = {node['name']: (index, node['target'])
             for index, node in enumerate(bridge['nodes'])}
    observations = {}
    seen = set()
    for row in intermediates['buffers']:
        name = row['reference']
        if name not in nodes:
            raise ValueError('diagnostic refers to an unknown exported node: ' + name)
        if row.get('shape_mismatch'):
            raise ValueError('diagnostic buffer shape differs from its reference: ' + name)
        order, operator = nodes[name]
        for sample in row['per_batch']:
            batch = sample['batch']
            if type(batch) is not int or batch < 0 or (batch, name) in seen:
                raise ValueError('invalid or duplicate per-image diagnostic')
            seen.add((batch, name))
            values = {key: float(sample[key]) for key in
                      ('tm_fp32_mse', 'bf16_fp32_mse', 'tm_bf16_mse', 'tm_bf16_max_error')}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError('diagnostic error must be finite and nonnegative')
            tm, bf16 = values['tm_fp32_mse'], values['bf16_fp32_mse']
            observations.setdefault(batch, []).append(dict(
                buffer=row['buffer'], reference=name, operator=operator,
                graph_order=order, **values,
                excess_mse=tm - BF16_ERROR_FACTOR * bf16,
                error_ratio=tm / bf16 if bf16 else None))
    if not observations:
        raise ValueError('no per-image intermediate observations')
    samples = []
    for batch, rows in sorted(observations.items()):
        rows.sort(key=lambda row: row['graph_order'])
        first = next((row for row in rows if row['excess_mse'] > 0), None)
        samples.append(dict(batch=batch, observed_buffers=len(rows),
                            first_observed_excess=first, final_observation=rows[-1]))
    return dict(scope='diagnostic only; not G-DNN-BF16', evidence='verified',
                criterion=f'TM MSE > {BF16_ERROR_FACTOR:g} * BF16 MSE against the same FP32 promotion',
                limitation='Only retained buffers are observed; fused or unexposed intermediate nodes are not localized.',
                samples=samples)


def check_binding(intermediates, bridge, identity):
    if (intermediates.get('artifact_id') != identity['artifact_id'] or
            intermediates.get('binary_sha256') != identity['binary_sha256'] or
            intermediates.get('bridge_digest') != digest(bridge)):
        raise ValueError('intermediate receipt is not bound to this artifact and bridge')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediates', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    identity = verify(args.library)
    manifest = json.loads(Path(str(args.library) + '.plan.json').read_text())
    if manifest.get('reuse', 'none') != 'none':
        raise ValueError('first-divergence diagnostics require --reuse none')
    intermediates = json.loads(args.intermediates.read_text())
    bridge = json.loads(args.bridge.read_text())
    check_binding(intermediates, bridge, identity)
    result = summarize(intermediates, bridge)
    from tilemega.cache import file_sha
    result.update(artifact_id=identity['artifact_id'],
                  intermediates_sha256=file_sha(args.intermediates),
                  bridge_sha256=file_sha(args.bridge))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['samples']:
        first = row['first_observed_excess']
        print(f'batch {row["batch"]}: ' + ('no observed excess' if first is None else
              f'{first["reference"]} ({first["operator"]}), excess MSE={first["excess_mse"]:.6g}'))


if __name__ == '__main__':
    main()
