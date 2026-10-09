"""DNN packing recipes; parameters retain their exported checkpoint FQNs."""
from __future__ import annotations

import math
from typing import Mapping
import torch


DNN_RECIPES = frozenset(('conv_bn_fold', 'bn_bias', 'fold_layernorm',
                         'linear_bias', 'qkv_concat_bias', 'gate_pair_interleave'))


def recipe_sources(recipe: Mapping, nested) -> tuple[str, ...]:
    kind = recipe['kind']
    if kind == 'qkv_concat_bias':
        return tuple(str(name) for name in recipe['sources'])
    value = recipe.get('source')
    result = nested(value) if isinstance(value, dict) else ((str(value),) if value is not None else ())
    if kind in ('conv_bn_fold', 'bn_bias'):
        result += tuple(str(recipe[key]) for key in ('gamma', 'mean', 'variance'))
        if kind == 'bn_bias':
            result += (str(recipe['beta']),)
            if recipe.get('bias') is not None:
                result += (str(recipe['bias']),)
    if kind == 'fold_layernorm':
        result += tuple(str(recipe[key]) for key in ('gamma', 'beta'))
        if recipe.get('bias') is not None:
            result += (str(recipe['bias']),)
    return result


def _vector(source, name, width, label):
    value = source(str(name))
    if value.ndim != 1 or value.numel() != width:
        raise ValueError(label + ' must have one value per channel')
    return value.float()


def pack(recipe: Mapping, source, nested):
    kind = recipe['kind']
    value = recipe.get('source')
    weight = nested(value) if isinstance(value, dict) else source(str(value)) if value is not None else None
    if kind in ('conv_bn_fold', 'bn_bias'):
        gamma = source(str(recipe['gamma']))
        if gamma.ndim != 1 or not gamma.numel():
            raise ValueError('batch norm gamma must be a nonempty vector')
        width = gamma.numel()
        mean = _vector(source, recipe['mean'], width, 'batch norm mean')
        variance = _vector(source, recipe['variance'], width, 'batch norm variance')
        epsilon = float(recipe['epsilon'])
        if not math.isfinite(epsilon) or epsilon <= 0 or torch.any(variance < 0):
            raise ValueError('batch norm needs positive epsilon and nonnegative variance')
        scale = gamma.float() * torch.rsqrt(variance + epsilon)
        if kind == 'bn_bias':
            beta = _vector(source, recipe['beta'], width, 'batch norm beta')
            bias = _vector(source, recipe['bias'], width, 'conv bias') if recipe.get('bias') is not None else 0
            return beta + (bias - mean) * scale
        if weight is None or weight.ndim != 4 or weight.shape[0] != width:
            raise ValueError('conv_bn_fold requires matching OIHW weight and batch norm')
        folded = (weight.float() * scale[:, None, None, None]).to(torch.bfloat16)
        from tilemega.serving.weights import _conv_krsc
        return _conv_krsc(folded, int(recipe['padded_channels']))
    if kind == 'fold_layernorm':
        if weight is None or weight.ndim != 2:
            raise ValueError('fold_layernorm requires a matrix weight')
        outputs, width = weight.shape
        gamma = _vector(source, recipe['gamma'], width, 'LayerNorm gamma')
        beta = _vector(source, recipe['beta'], width, 'LayerNorm beta')
        # The correction sum must describe the stored BF16 weight, otherwise
        # a constant row leaks the folded-weight quantization error through mu.
        folded = (weight.float() * gamma[None, :]).to(torch.bfloat16)
        part = recipe['part']
        if part == 'weight':
            return folded
        if part == 'u':
            return folded.float().sum(dim=1)
        if part == 'v':
            bias = _vector(source, recipe['bias'], outputs, 'linear bias') if recipe.get('bias') is not None else 0
            return torch.mv(weight.float(), beta) + bias
        raise ValueError('fold_layernorm part must be weight, u or v')
    if kind == 'linear_bias':
        if weight is None or weight.ndim != 1:
            raise ValueError('linear_bias requires a vector')
        return weight.float()
    if kind == 'qkv_concat_bias':
        items = [source(str(name)) for name in recipe['sources']]
        heads, dim = int(recipe['heads']), int(recipe['head_dim'])
        part = recipe['part']
        rank = 2 if part == 'weight' else 1 if part == 'bias' else 0
        if heads <= 0 or dim <= 0 or len(items) != 3 or not rank or any(
                item.ndim != rank or item.shape[0] != heads * dim for item in items):
            raise ValueError('QKV recipe requires three equal head-major weights or biases')
        if len({tuple(item.shape) for item in items}) != 1:
            raise ValueError('QKV sources disagree on shape')
        tail = tuple(items[0].shape[1:])
        output = torch.stack([item.reshape(heads, dim, *tail) for item in items], dim=1)
        output = output.reshape(3 * heads * dim, *tail)
        return output.to(torch.bfloat16) if part == 'weight' else output.float()
    if kind == 'gate_pair_interleave':
        unit = int(recipe['u'])
        if weight is None or weight.ndim not in (1, 2, 4) or unit <= 0 or weight.shape[0] % (2 * unit):
            raise ValueError('gate_pair_interleave requires complete channel pairs')
        first, second = weight.chunk(2, dim=0)
        shape = (-1, unit, *weight.shape[1:])
        return torch.stack((first.reshape(shape), second.reshape(shape)), dim=1).reshape(weight.shape)
    raise ValueError('unsupported DNN recipe ' + str(kind))
