#!/usr/bin/env python3
"""Record and evaluate upstream BERT's unmasked SDPA shape fragment on CPU."""
import argparse
import json
from pathlib import Path

import torch
from tilemega.dnn import load_program


def probe(path, batches):
    program = load_program(path)
    roots = {s.arg.name: s.kind.name for s in program.graph_signature.input_specs}
    by_name = {n.name: n for n in program.graph.nodes}
    masks, fragments = [], {}
    for node in program.graph.nodes:
        if str(node.target) != 'aten.scaled_dot_product_attention.default':
            continue
        mask = node.args[3]
        masks.append(dict(sdpa=node.name, mask=mask.name,
                          shape=[str(d) for d in mask.meta['val'].shape]))
        if mask.name in fragments:
            continue
        pending, seen, boundaries = [mask], {}, {}
        while pending:
            ancestor = pending.pop()
            if ancestor.name in seen:
                continue
            seen[ancestor.name] = ancestor
            if str(ancestor.target).startswith('aten.sym_size.'):
                source, axis = ancestor.args
                boundaries[ancestor.name] = dict(input=source.name, axis=axis,
                    shape=[str(d) for d in source.meta['val'].shape],
                    dtype=str(source.meta['val'].dtype))
            else:
                pending.extend(ancestor.all_input_nodes)
        values = {n.name: roots.get(n.name) for n in seen.values()
                  if n.op == 'placeholder'}
        if values:
            raise ValueError('SDPA mask depends on tensor values: ' + str(values))
        fragment = dict(shape_boundaries=boundaries,
                        value_dependent_placeholders=values,
                        fragment=[dict(name=n.name, target=str(n.target),
                                       args=str(n.args), kwargs=str(n.kwargs))
                                  for n in program.graph.nodes if n.name in seen],
                        cpu_bound_checks=[])
        for batch in batches:
            interpreter = torch.fx.Interpreter(program.graph_module)
            for boundary in boundaries.values():
                source = by_name[boundary['input']]
                shape = list(source.meta['val'].shape)
                if boundary['axis'] != 0 or any(not isinstance(d, int) for d in shape[1:]):
                    raise ValueError('probe expects batch-only symbolic BERT inputs')
                shape[0] = batch
                interpreter.env[source] = torch.empty(shape, device='cpu',
                                                       dtype=source.meta['val'].dtype)
            with torch.no_grad():
                for ancestor in program.graph.nodes:
                    if ancestor.name in seen:
                        interpreter.env[ancestor] = interpreter.run_node(ancestor)
            value = interpreter.env[mask]
            if value.device.type != 'cpu' or value.dtype != torch.bool or not bool(value.all()):
                raise ValueError('unmasked BERT shape fragment is not all true')
            fragment['cpu_bound_checks'].append(dict(batch=batch, shape=list(value.shape),
                elements=value.numel(), dtype=str(value.dtype), all_true=bool(value.all())))
        fragments[mask.name] = fragment
    if not masks:
        raise ValueError('no SDPA masks found')
    return dict(evidence='verified',
        scope='FX dependency transcription and isolated CPU Interpreter evaluation',
        program=str(Path(path).resolve()), masks=masks, mask_fragments=fragments,
        all_sdpa_nodes_share_mask=len(fragments) == 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 8, 32, 64])
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = probe(args.program, args.batches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(masks=len(result['masks']),
                          unique_fragments=len(result['mask_fragments']),
                          batches=args.batches)), flush=True)


if __name__ == '__main__':
    main()
