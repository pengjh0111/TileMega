"""Generic rank-four layout propagation for the pinned FX2AIT frontend."""
import torch
from aitemplate.compiler import ops
from aitemplate.compiler.base import Tensor
from fx2ait.converters.converter_registry import AIT_CONVERTERS

ORIGINALS = dict(AIT_CONVERTERS)
COUNTS = dict(to_nhwc=0, to_nchw=0, axis_rewrite=0, broadcast_promote=0, constant_to_nhwc=0, runtime_to_nhwc=0, logical_boundary_by_op={})
PHYSICAL = (0, 2, 3, 1)
LOGICAL = (0, 3, 1, 2)
TAG = 'dnn1_nhwc'


def tagged(value):
    return isinstance(value, Tensor) and value._attrs.get(TAG, False)


def mark(value):
    if isinstance(value, Tensor) and len(value.shape()) == 4:
        value._attrs[TAG] = True
    elif isinstance(value, (list, tuple)):
        for item in value: mark(item)
    return value


def nhwc(value, broadcast=False):
    if not isinstance(value, Tensor) or tagged(value): return value
    rank = len(value.shape())
    if broadcast and 0 < rank < 4:
        value = ops.reshape()(value, [1] * (4-rank) + value.shape())
        COUNTS['broadcast_promote'] += 1
        rank = 4
    if rank == 4:
        COUNTS['to_nhwc'] += 1
        COUNTS['constant_to_nhwc' if value._attrs.get('data') is not None else 'runtime_to_nhwc'] += 1
        return mark(ops.permute()(value, list(PHYSICAL)))
    return value


def nchw(value):
    if tagged(value):
        COUNTS['to_nchw'] += 1
        return ops.permute()(value, list(LOGICAL))
    if isinstance(value, (list, tuple)):
        return type(value)(nchw(item) for item in value)
    return value


def dim(value):
    COUNTS['axis_rewrite'] += 1
    return LOGICAL[value % 4]


POINTWISE = {'add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'rsqrt', 'neg', 'abs',
             'relu', 'sigmoid', 'tanh', 'hardtanh', 'clamp', 'gelu', 'silu',
             'exp', 'log', '_to_copy', 'clone', 'contiguous', 'detach'}
NHWC_BODIES = {'convolution', 'batch_norm', 'max_pool2d', 'avg_pool2d', 'adaptive_avg_pool2d'}


def wrap(target, converter):
    name = str(target).split('.')[1] if str(target).startswith('aten.') else ''
    def translate(target, args, kwargs, node_name):
        values = list(args)
        if name in NHWC_BODIES:
            values[0] = nhwc(values[0])
            return mark(converter(target, tuple(values), kwargs, node_name))
        if name in POINTWISE and any(tagged(x) for x in values):
            values = [nhwc(x, broadcast=True) for x in values]
            return mark(converter(target, tuple(values), kwargs, node_name))
        if name in {'mean', 'sum'} and tagged(values[0]):
            values[1] = [dim(x) for x in values[1]]
            return mark(converter(target, tuple(values), kwargs, node_name))
        if name == 'slice' and tagged(values[0]):
            defaults=[values[0],0,0,9223372036854775807,1];defaults[:len(values)]=values
            defaults[1]=dim(defaults[1])
            return mark(converter(target, tuple(defaults), kwargs, node_name))
        if name in {'split', 'split_with_sizes', 'chunk'} and tagged(values[0]):
            axis = values[2] if len(values)>2 else 0
            if len(values)<3: values.append(axis)
            values[2] = dim(axis)
            return mark(converter(target, tuple(values), kwargs, node_name))
        if name == 'cat' and any(tagged(x) for x in values[0]):
            values[0] = [nhwc(x) for x in values[0]]
            values[1] = dim(values[1] if len(values)>1 else 0)
            return mark(converter(target, tuple(values), kwargs, node_name))
        if name == 'sym_size' and tagged(values[0]):
            values[1] = dim(values[1])
            return converter(target, tuple(values), kwargs, node_name)
        # Reshape/permutation and other operators retain ATen logical ordering.
        if any(tagged(x) for x in values):
            counts = COUNTS['logical_boundary_by_op']
            counts[name] = counts.get(name, 0) + 1
        values = [nchw(x) for x in values]
        return converter(target, tuple(values), kwargs, node_name)
    return translate


def install():
    for target, converter in list(AIT_CONVERTERS.items()):
        if str(target).startswith('aten.'):
            AIT_CONVERTERS[target] = wrap(target, converter)
