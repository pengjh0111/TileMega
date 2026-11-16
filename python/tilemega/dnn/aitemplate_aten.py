"""Static ATen converters shared by all DNN1 AITemplate models."""
import operator
import torch
from aitemplate.compiler import ops
from aitemplate.compiler.public import FuncEnum, Tensor
from fx2ait.converters.converter_registry import AIT_CONVERTERS as REG
from fx2ait.utils import dtype_to_str


def register(target):
    def decorate(fn):
        REG[target] = fn
        return fn
    return decorate


def dims(value):
    return [d.value() for d in value.shape()]


@register(torch.ops.aten.sym_size)
def static_size(target, args, kwargs, name):
    return args[0].shape()[args[1]].value()


@register(operator.eq)
def scalar_equal(target, args, kwargs, name):
    assert all(isinstance(x, (int, bool)) for x in args), 'static shape equality only'
    return operator.eq(*args)


@register(torch.ops.aten._assert_scalar.default)
def scalar_assert(target, args, kwargs, name):
    assert args[0] is True, args[1]
    return None


@register(torch.ops.aten._assert_tensor_metadata.default)
def metadata_assert(target, args, kwargs, name):
    if len(args) > 3 and args[3] is not None:
        assert args[0].dtype() == dtype_to_str(args[3])
    if len(args) > 1 and args[1] is not None:
        assert dims(args[0]) == list(args[1])
    if len(args) > 2 and args[2] is not None:
        raise ValueError('nonempty stride metadata assertion not implemented')
    return None


@register(torch.ops.aten.embedding.default)
def embedding(target, args, kwargs, name):
    weight, indices = args[:2]
    shape = dims(indices)
    count = 1
    for d in shape: count *= d
    width = dims(weight)[1]
    flat = ops.reshape()(indices, [count, 1])
    expanded = ops.expand()(flat, [count, width])
    return ops.reshape()(ops.gather()(weight, 0, expanded), shape + [width])


@register(torch.ops.aten._to_copy.default)
def cast(target, args, kwargs, name):
    dtype = dtype_to_str(kwargs.get('dtype', args[0].dtype()))
    return ops.ne()(args[0], 0) if dtype == 'bool' else ops.cast()(args[0], dtype)


@register(torch.ops.aten.full.default)
def full(target, args, kwargs, name):
    return ops.full()(args[0], args[1], dtype_to_str(kwargs.get('dtype', torch.float32)))


@register(torch.ops.aten.scalar_tensor.default)
def scalar_tensor(target, args, kwargs, name):
    return Tensor(shape=[], value=args[0], dtype=dtype_to_str(kwargs.get('dtype', torch.float32)))


@register(torch.ops.aten.full_like.default)
def full_like(target, args, kwargs, name):
    return ops.full()(args[0].shape(), args[1], args[0].dtype())


@register(torch.ops.aten._softmax.default)
def softmax(target, args, kwargs, name):
    assert not args[2], 'half_to_float softmax needs explicit input promotion'
    return ops.softmax()(args[0], args[1])


@register(torch.ops.aten.eq.Scalar)
def equal(target, args, kwargs, name):
    return ops.eq()(args[0], args[1])


@register(torch.ops.aten.logical_not.default)
def logical_not(target, args, kwargs, name):
    x = ops.cast()(args[0], 'float32') if args[0].dtype() == 'bool' else args[0]
    return ops.eq()(x, 0)


@register(torch.ops.aten.any.dim)
def any_dim(target, args, kwargs, name):
    x = ops.cast()(args[0], 'float32')
    return ops.ne()(ops.reduce_sum(args[1], keepdim=args[2])(x), 0)


@register(torch.ops.aten.where.self)
def where(target, args, kwargs, name):
    shape = list(torch.broadcast_shapes(*(dims(x) for x in args)))
    values = [x._attrs['value'] if x.is_a_const_num()
              else (ops.expand()(x, shape) if dims(x) != shape else x) for x in args]
    return ops.where()(*values)


@register(torch.ops.aten.gelu.default)
def gelu(target, args, kwargs, name):
    approx = kwargs.get('approximate', args[1] if len(args) > 1 else 'none')
    assert approx in ('none', 'tanh')
    return ops.elementwise(FuncEnum.GELU if approx == 'none' else FuncEnum.FASTGELU)(args[0])


REG[torch.ops.aten.mul.Scalar] = REG[torch.ops.aten.mul.Tensor]

@register(torch.ops.aten.layer_norm.default)
def layernorm(target, args, kwargs, name):
    return ops.layernorm()(args[0], args[2], args[3], args[1], eps=float(args[4]))


def identity_adaptive_avg_pool(converter):
    def convert(target, args, kwargs, name):
        value = args[0]
        size = args[1] if len(args) > 1 else kwargs.get("output_size")
        size = (size, size) if isinstance(size, int) else size
        shape = value.shape()
        if len(shape) == 4 and isinstance(size, (tuple, list)) and len(size) == 2:
            spatial = [dim._attrs["values"] for dim in shape[1:3]]
            if all(len(set(values)) == 1 and (want is None or want == values[0])
                   for values, want in zip(spatial, size)):
                return value
        if converter is None:
            raise NotImplementedError("nonidentity adaptive average pooling has no converter")
        return converter(target, args, kwargs, name)
    return convert


def promoted_binary(converter):
    def convert(target, args, kwargs, name):
        values = list(args)
        if all(isinstance(x, Tensor) for x in values[:2]):
            # ATen scalar tensor precedence differs from plain promote_types.
            probes = [torch.empty([1] if x._rank() else [], dtype=getattr(torch, x.dtype()))
                      for x in values[:2]]
            dtype = dtype_to_str(torch.result_type(*probes))
            values[:2] = [ops.cast()(x, dtype) if x.dtype() != dtype else x for x in values[:2]]
        return converter(target, tuple(values), kwargs, name)
    return convert


for _op in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar, torch.ops.aten.div.Tensor):
    REG[_op] = promoted_binary(REG[_op])
