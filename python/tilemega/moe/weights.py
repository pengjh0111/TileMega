"""Pack original expert FQNs directly into one deployment weight allocation."""
from __future__ import annotations

import torch


def is_expert_recipe(recipe):
    while recipe.get('kind') in ('tile_pages','fold_rmsnorm'):
        recipe = recipe.get('source', {})
        if not isinstance(recipe, dict):
            return False
    return recipe.get('kind') == 'expert_stack'


def _parts(recipe):
    tile = None;norm = None
    while recipe.get('kind') in ('tile_pages','fold_rmsnorm'):
        if recipe['kind'] == 'tile_pages':
            if tile is not None:
                raise ValueError('expert weights cannot be tiled twice')
            tile = (int(recipe['tile_n']),int(recipe['tile_k']))
        else:
            if norm is not None:
                raise ValueError('expert weights cannot fold two normalizations')
            norm = str(recipe['norm'])
        recipe = recipe['source']
    if recipe.get('kind') != 'expert_stack':
        raise ValueError('expected an expert_stack leaf recipe')
    prefix = str(recipe['prefix']);part = recipe['part']
    experts,hidden,intermediate,unit = (int(recipe[key]) for key in ('experts','hidden','intermediate','u'))
    if part not in ('gate_up','down') or min(experts,hidden,intermediate,unit) <= 0 or intermediate % unit:
        raise ValueError('invalid expert stack geometry or interleave unit')
    if norm is not None and part != 'gate_up':
        raise ValueError('post-attention normalization folds only into gate/up and router')
    n,k = (2*intermediate,hidden) if part == 'gate_up' else (hidden,intermediate)
    if tile is not None:
        tn,tk = tile
        if tn <= 0 or tk not in (16,32,64,128) or n % tn or k % tk or \
                (part == 'gate_up' and tn % (2*unit)):
            raise ValueError('expert page tiles must divide the expert and own complete gate pairs')
    if not prefix or not prefix.endswith('.'):
        raise ValueError('expert prefix must end at the original experts namespace')
    return prefix,part,experts,hidden,intermediate,unit,norm,tile


def recipe_sources(recipe):
    prefix,part,experts,_,_,_,norm,_ = _parts(recipe)
    kinds = ('gate','up') if part == 'gate_up' else ('down',)
    names = tuple(prefix+f'{expert}.{kind}_proj.weight' for expert in range(experts) for kind in kinds)
    return names+((norm,) if norm is not None else ())


def pack_experts(recipe, source, tile_pack):
    prefix,part,experts,hidden,intermediate,unit,norm_name,tile = _parts(recipe)
    gamma = None
    if norm_name is not None:
        gamma = source(norm_name)
        if gamma.shape != (hidden,) or gamma.dtype != torch.bfloat16:
            raise ValueError('expert normalization shape/dtype differs from hidden width')
        gamma = gamma.float()[None,:]
    def read(expert, kind, shape):
        value = source(prefix+f'{expert}.{kind}_proj.weight')
        if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
            raise ValueError('original expert projection has incompatible shape or dtype')
        return value
    def one(expert):
        if part == 'gate_up':
            gate = read(expert,'gate',(intermediate,hidden))
            up = read(expert,'up',(intermediate,hidden))
            matrix = torch.stack((gate.reshape(-1,unit,hidden),up.reshape(-1,unit,hidden)),dim=1).reshape(2*intermediate,hidden)
        else:
            matrix = read(expert,'down',(hidden,intermediate))
        if gamma is not None:
            matrix = (matrix.float()*gamma).to(torch.bfloat16)
        return tile_pack(matrix,*tile) if tile is not None else matrix
    first = one(0)
    shape = (experts,*first.shape) if tile is None else (experts,first.numel())
    output = first.new_empty(shape)
    output[0].copy_(first.reshape_as(output[0]));del first
    # Sources and temporary layouts die per expert; no whole untiled expert
    # stack coexists with the deployment's tiled stack.
    for expert in range(1,experts):
        value = one(expert)
        output[expert].copy_(value.reshape_as(output[expert]));del value
    return output if tile is None else output.reshape(-1)
