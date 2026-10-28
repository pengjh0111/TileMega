#!/usr/bin/env python3
"""N-2 diagnostic: every native forward buffer against the exported graph.

`--chain` (default) compares each buffer with the graph's own tensor, so an
early defect shows in every later stage. `--isolate` recomputes each node in
torch from the *native* input buffers, so the cosine measures that stage's own
arithmetic alone and the first low row is the defect.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from tilemega.serving.plan import PlanLibrary
from tilemega.serving.weights import load_weights

DTYPES = {0: (torch.bfloat16, 2), 1: (torch.float32, 4), 2: (torch.int32, 4)}


class Capture(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.values = {}

    def run_node(self, node):
        result = super().run_node(node)
        if isinstance(result, torch.Tensor):
            self.values[node.name] = result.detach()
        return result


def native_view(raw, row, batch):
    """One buffer's elements as [N, C, H, W] (NHWC) or [rows, C] (row major)."""
    logical = [int(v) for v in row['logical']]
    physical = [int(v) for v in row['physical']]
    if row['layout'] == 'nhwc' and row['rank'] == 4:
        top, left = row['halo_top'], row['halo_left']
        value = raw[:batch * physical[1] * physical[2] * physical[3]]
        value = value.reshape(batch, physical[1], physical[2], physical[3])
        value = value[:, top:top + logical[1], left:left + logical[2], :logical[3]]
        return value.permute(0, 3, 1, 2).contiguous()
    rows = physical[0] if row['rank'] else raw.numel()
    width = physical[1] if row['rank'] > 1 else 1
    return raw[:rows * width].reshape(rows, width)[:, :max(1, logical[1] if row['rank'] > 1 else 1)]


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        return None
    scale = a.norm() * b.norm()
    return float(torch.dot(a, b) / scale) if float(scale) > 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'export', 'bridge', 'inputs'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--threshold', type=float, default=0.99)
    p.add_argument('--isolate', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    manifest = json.loads(Path(str(a.library) + '.plan.json').read_text())
    rows = {row['name']: row for row in manifest['buffers']}
    # A fused stage writes one buffer for a whole group of exported nodes
    # (conv + BN + ReLU + residual): the buffer holds the group's last node.
    aliases = {}
    for name, index in manifest.get('node_buffers', {}).items():
        aliases.setdefault(int(index), []).append(name)
    library = PlanLibrary(a.library)
    bridge = json.loads(a.bridge.read_text())
    nodes = {node['name']: node for node in bridge['nodes']}
    signature = bridge['signature']
    inputs = [i['name'] for i in signature['inputs'] if i['kind'] == 'USER_INPUT']
    outputs = [o['name'] for o in signature['outputs'] if o['kind'] == 'USER_OUTPUT']
    values = load_file(str(a.inputs), device='cuda')

    program = torch.export.load(str(a.export / 'exported_program.pt2'))
    module = program.module().to('cuda').to(torch.bfloat16)
    arguments = [values[name].to('cuda') for name in inputs]
    with torch.inference_mode():
        capture = Capture(module)
        capture.run(*arguments)
    reference = capture.values

    tensors = load_weights(a.export / 'checkpoint', library, device='cuda')
    for name in inputs:
        tensors[name] = values[name].contiguous()
    for name in outputs:
        node = nodes[name]
        tensors[name] = torch.zeros([a.batch, *map(int, node['shape'][1:])],
                                    dtype=torch.bfloat16, device='cuda')
    plan = library.create(a.batch, {n: v.data_ptr() for n, v in tensors.items()}, 0)
    read = library.lib.tm_plan_read_buffer
    read.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64]
    read.restype = C.c_int
    native = {}
    with plan:
        plan.set_steps([0])
        plan.launch(0, 1, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        order = {node.name: position for position, node in enumerate(module.graph.nodes)}
        for index, info in enumerate(library.buffers):
            row = rows.get(info.name)
            held = sorted((n for n in aliases.get(index, []) if n in reference),
                          key=lambda n: order.get(n, -1))
            if row is None or not held or info.dtype == 2:
                continue
            dtype, size = DTYPES[info.dtype]
            count = info.elements_constant + a.batch * info.elements_per_batch
            host = torch.empty(int(count), dtype=dtype)
            if read(plan.handle, index, host.data_ptr(), host.numel() * size):
                continue
            native[info.name] = (index, native_view(host, row, a.batch).cuda(), held)

    report = []
    graph = {node.name: node for node in module.graph.nodes}
    for name, (index, actual, held) in native.items():
        # A fused group's buffer may alias several nodes; report the closest.
        scored = [(cosine(actual, reference[n]), n) for n in held
                  if reference[n].shape.numel() == actual.numel()]
        scored = [(value, n) for value, n in scored if value is not None]
        best = max(scored, default=(None, held[-1]))
        node_name = best[1]
        truth = reference[node_name]
        row = dict(index=index, name=name, node=node_name, aliases=held,
                   shape=list(actual.shape),
                   chain_cosine=best[0] if best[0] is not None else cosine(actual, truth),
                   absmean=float(actual.float().abs().mean()),
                   reference_absmean=float(truth.float().abs().mean()))
        if a.isolate and node_name in graph:
            node = graph[node_name]

            def value(item):
                if isinstance(item, torch.fx.Node):
                    for other, (_, value_tensor, held) in native.items():
                        if held == item.name:
                            return value_tensor.to(reference[item.name].dtype)
                    return reference.get(item.name, capture.env.get(item))
                if isinstance(item, (list, tuple)):
                    return type(item)(value(entry) for entry in item)
                return item

            try:
                with torch.inference_mode():
                    expected = node.target(*[value(x) for x in node.args],
                                           **{k: value(v) for k, v in node.kwargs.items()})
                if isinstance(expected, torch.Tensor):
                    row['isolated_cosine'] = cosine(actual, expected)
                    row['isolated_absmean'] = float(expected.float().abs().mean())
            except Exception as error:  # a node the interpreter cannot replay alone
                row['isolated_error'] = type(error).__name__ + ': ' + str(error)[:80]
        report.append(row)
    key = 'isolated_cosine' if a.isolate else 'chain_cosine'
    first = next((r for r in report if r.get(key) is None or r[key] < a.threshold), None)
    for r in report:
        value = r.get(key)
        mark = '' if value is not None and value >= a.threshold else (
            '  <-- FIRST BAD' if r is first else '  <-- bad')
        print(f"{r['index']:>3} {r['name'][:20]:20} -> {r.get('node','')[:18]:18} {key.split('_')[0]}="
              f"{'none' if value is None else f'{value:.6f}'}"
              f" chain={r['chain_cosine'] if r['chain_cosine'] is None else round(r['chain_cosine'],4)}"
              f" mean={r['absmean']:.4g}/{r.get('isolated_absmean', r['reference_absmean']):.4g}"
              f"{' ' + r['isolated_error'] if 'isolated_error' in r else ''}{mark}")
    if a.out:
        a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
