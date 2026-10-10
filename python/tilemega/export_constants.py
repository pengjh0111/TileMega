"""CPU evaluation of value-independent FX fragments; no operator lowering."""
from __future__ import annotations

import base64
import math
from typing import Any

import torch

MAX_CONSTANT_ELEMENTS = 1 << 20
INLINE_CONSTANT_BYTES = 4096


def argument(value: Any) -> dict:
    if isinstance(value, torch.fx.Node):
        return dict(t='node', v=value.name)
    if value is None:
        return dict(t='none', v=None)
    if isinstance(value, bool):
        return dict(t='bool', v=value)
    if isinstance(value, int):
        return dict(t='int', v=value)
    if isinstance(value, float):
        encoded = value if math.isfinite(value) else (
            'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf')
        return dict(t='float', v=encoded)
    if isinstance(value, str):
        return dict(t='str', v=value)
    if isinstance(value, (tuple, list)):
        return dict(t='list', v=[argument(item) for item in value])
    for kind in ('dtype', 'device', 'layout', 'memory_format'):
        if isinstance(value, getattr(torch, kind)):
            return dict(t=kind, v=str(value))
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return dict(t='symbol', v=str(value))
    raise TypeError(f'unsupported FX argument {type(value).__name__}: {value!r}')


def constant(value: Any) -> dict | None:
    if isinstance(value, (bool, int, float)):
        return dict(dtype=type(value).__name__, shape=[], scalar=argument(value))
    if not isinstance(value, torch.Tensor) or value.numel() > MAX_CONSTANT_ELEMENTS:
        return None
    value = value.detach().contiguous().cpu()
    result = dict(dtype=str(value.dtype), shape=list(value.shape), elements=value.numel())
    if value.dtype == torch.bool or not value.is_floating_point() and not value.is_complex():
        if value.dtype == torch.bool:
            result['all_true'] = bool(value.all())
        if value.numel():
            first = value.reshape(-1)[0].item()
            result['all_equal'] = dict(equal=bool((value == first).all()), value=first)
    if value.numel() * value.element_size() <= INLINE_CONSTANT_BYTES:
        raw = bytes(value.reshape(-1).view(torch.uint8).tolist())
        result.update(data_base64=base64.b64encode(raw).decode('ascii'), byte_order='little')
    return result


def _symbols(value):
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return {str(symbol) for symbol in value.node.expr.free_symbols}
    return set()


def _literal_symbols(value):
    if isinstance(value, (tuple, list)):
        return set().union(*(_literal_symbols(item) for item in value))
    if isinstance(value, dict):
        return _literal_symbols(list(value.values()))
    return _symbols(value)


def _shape_node(node):
    return node.op == 'call_function' and str(node.target) in (
        'aten.sym_size.int', 'aten.sym_numel.default')


def _dimensions(node):
    source = node.args[0].meta.get('val')
    if not isinstance(source, torch.Tensor):
        raise ValueError('shape query has no tensor metadata: ' + node.name)
    return [source.shape[node.args[1]]] if str(node.target) == 'aten.sym_size.int' else list(source.shape)


def _resolve(value, bindings):
    if isinstance(value, int):
        return value
    expression = value.node.expr
    substituted = expression.subs({symbol: bindings[str(symbol)]
                                  for symbol in expression.free_symbols})
    if not substituted.is_integer or substituted.free_symbols:
        raise ValueError('shape binding is not an integer: ' + str(substituted))
    return int(substituted)


def _bounded(value, bindings):
    if isinstance(value, torch.Tensor):
        try:
            return math.prod(_resolve(d, bindings) for d in value.shape) <= MAX_CONSTANT_ELEMENTS
        except KeyError:
            return False
    if isinstance(value, (tuple, list)):
        return all(_bounded(item, bindings) for item in value)
    return True


class _CpuInterpreter(torch.fx.Interpreter):
    def __init__(self, module, bindings):
        super().__init__(module)
        self.bindings = bindings

    def run_node(self, node):
        if _shape_node(node):
            return math.prod(_resolve(d, self.bindings) for d in _dimensions(node))
        return super().run_node(node)

    def call_function(self, target, args, kwargs):
        # Device placement is irrelevant to a constant's values and shape;
        # execution must stay on CPU even for an archive exported on CUDA.
        def bound_literal(value):
            if isinstance(value, torch.SymInt):
                return _resolve(value, self.bindings)
            if isinstance(value, (torch.SymFloat, torch.SymBool)):
                expression = value.node.expr
                expression = expression.subs({s: self.bindings[str(s)] for s in expression.free_symbols})
                return bool(expression) if isinstance(value, torch.SymBool) else float(expression)
            return torch.device('cpu') if isinstance(value, torch.device) else value
        args = torch.fx.node.map_aggregate(args, bound_literal)
        kwargs = dict(torch.fx.node.map_aggregate(kwargs, bound_literal))
        if 'device' in kwargs and kwargs['device'] is not None:
            kwargs['device'] = torch.device('cpu')
        if kwargs.get('pin_memory'):
            kwargs['pin_memory'] = False
        return target(*args, **kwargs)


def annotate(program, records, bindings=None):
    """Annotate constants, retaining an FX fragment for unbound shape facts."""
    bindings = dict(bindings or {})
    for symbol, value in bindings.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('shape binding must be an integer: ' + symbol)
        ranges = {str(s): r for s, r in program.range_constraints.items()}
        if symbol not in ranges:
            raise ValueError('unknown shape symbol: ' + symbol)
        bounds = ranges[symbol]
        if not bool(bounds.lower <= value) or not bool(value <= bounds.upper):
            raise ValueError('shape binding is outside the export range: ' + symbol)
    roles = {spec.arg.name: spec for spec in program.graph_signature.input_specs
             if hasattr(spec.arg, 'name')}
    mutations = {spec.target for spec in program.graph_signature.output_specs
                 if spec.kind.name == 'BUFFER_MUTATION'}
    records_by_name = {record['name']: record for record in records}
    eligibility, symbols = {}, {}
    interpreter = _CpuInterpreter(program.graph_module, bindings)
    for node in program.graph.nodes:
        record = records_by_name[node.name]
        if node.op == 'placeholder':
            spec = roles.get(node.name)
            # Nonpersistent buffers travel in the archive's constants bank.
            # Preserve their frozen values separately: a buffer read is still
            # value-dependent and must not become a constant-subgraph ancestor.
            if spec is not None and spec.kind.name == 'BUFFER' and \
                    spec.persistent is False and spec.target not in mutations:
                value = program.constants.get(spec.target)
                if isinstance(value, torch.Tensor) and _bounded(value, bindings):
                    record['immutable_buffer_value'] = constant(value)
            eligibility[node] = spec is not None and spec.kind.name == 'CONSTANT_TENSOR'
            symbols[node] = set()
            if eligibility[node]:
                value = program.constants[spec.target]
                if _bounded(value, bindings):
                    interpreter.env[node] = value.detach().cpu().clone()
                    record['constant'] = constant(interpreter.env[node])
            continue
        if node.op == 'output':
            eligibility[node], symbols[node] = False, set()
            continue
        if _shape_node(node):
            eligibility[node] = True
            symbols[node] = set().union(*(_symbols(d) for d in _dimensions(node)))
        else:
            eligibility[node] = all(eligibility.get(n, False) for n in node.all_input_nodes)
            symbols[node] = set().union(*(symbols.get(n, set()) for n in node.all_input_nodes))
            symbols[node] |= _literal_symbols(node.args) | _literal_symbols(node.kwargs)
        if not eligibility[node]:
            continue
        if symbols[node]:
            pending, fragment = [node], set()
            while pending:
                parent = pending.pop()
                if parent in fragment:
                    continue
                fragment.add(parent)
                if not _shape_node(parent):
                    pending.extend(parent.all_input_nodes)
            record['shape_constant'] = dict(symbols=sorted(symbols[node]),
                fragment=[dict(name=n.name, op=n.op, target=str(n.target),
                               args=argument(n.args),
                               kwargs={k: argument(v) for k, v in sorted(n.kwargs.items())},
                               input_shape=[str(d) for d in n.args[0].meta['val'].shape]
                                   if _shape_node(n) else None)
                          for n in program.graph.nodes if n in fragment])
            if not symbols[node].issubset(bindings):
                continue
            record['shape_constant']['bindings'] = {s: bindings[s] for s in sorted(symbols[node])}
        if not _bounded(node.meta.get('val'), bindings):
            record['constant_evaluation'] = 'output exceeds element limit'
            continue
        if not _shape_node(node) and any(n not in interpreter.env for n in node.all_input_nodes):
            record['constant_evaluation'] = 'ancestor could not be evaluated'
            continue
        try:
            with torch.no_grad():
                value = interpreter.run_node(node)
        except (RuntimeError, TypeError, ValueError, NotImplementedError) as error:
            record['constant_evaluation'] = type(error).__name__ + ': ' + str(error)
            continue
        interpreter.env[node] = value
        result = constant(value)
        if result is not None:
            record['constant'] = result
