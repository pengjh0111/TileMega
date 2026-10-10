"""Check serving packing geometry using BF16 checkpoint headers only."""
from __future__ import annotations

import math


def describe(recipe, tensors, config):
    sources = set()

    def integer(value, label):
        if type(value) is not int or value <= 0:
            raise ValueError('invalid packing integer: '+label)
        return value

    def source(name):
        if not isinstance(name, str) or name not in tensors:
            raise ValueError('packing source absent from checkpoint: '+str(name))
        entry = tensors[name]
        if entry['dtype'] != 'BF16':
            raise ValueError('packing source must be BF16: '+name)
        sources.add(name)
        return tuple(entry['shape'])

    def visit(value):
        if not isinstance(value, dict):
            raise ValueError('packing recipe must be an object')
        kind = value.get('kind')
        if kind == 'alias':
            shape = source(value['source'])
            return shape, shape, None, False, False
        if kind == 'expert_stack':
            e, h, i, u = (integer(value[k], k) for k in ('experts', 'hidden', 'intermediate', 'u'))
            configured = config.get('num_experts', config.get('num_local_experts'))
            if e != configured or h != config['hidden_size'] or i != config['moe_intermediate_size'] or i % u:
                raise ValueError('expert recipe dimensions differ from checkpoint config')
            prefix, part = value['prefix'], value['part']
            if not isinstance(prefix, str) or not prefix.endswith('.') or part not in ('gate_up', 'down'):
                raise ValueError('invalid expert namespace or projection')
            for expert in range(e):
                for projection in (('gate', 'up') if part == 'gate_up' else ('down',)):
                    expected = (i, h) if projection != 'down' else (h, i)
                    if source(prefix+f'{expert}.{projection}_proj.weight') != expected:
                        raise ValueError('expert recipe source shape differs')
            shape = (e, 2*i, h) if part == 'gate_up' else (e, h, i)
            return shape, shape, (part, u), False, False
        if kind in ('fold_rmsnorm', 'tile_pages'):
            shape, logical, expert, tiled, folded = visit(value['source'])
            if kind == 'fold_rmsnorm':
                if folded or (tiled and expert is None) or len(logical) not in (2, 3) or \
                        (expert is not None and expert[0] != 'gate_up') or \
                        source(value['norm']) != (logical[-1],):
                    raise ValueError('invalid normalization fold shape or order')
                return shape, logical, expert, tiled, True
            tn, tk = (integer(value[k], k) for k in ('tile_n', 'tile_k'))
            if tiled or tn % 8 or (tk not in (16, 32) and tk % 64) or \
                    len(logical) != (3 if expert else 2):
                raise ValueError('invalid tile_pages geometry or source')
            n, k = logical[-2:]
            if expert and (tk not in (16, 32, 64, 128) or n % tn or k % tk or
                    (expert[0] == 'gate_up' and tn % (2*expert[1]))):
                raise ValueError('expert tiles split a projection or gate pair')
            elements = ((n+tn-1)//tn)*((k+tk-1)//tk)*tn*tk
            if expert:
                elements *= logical[0]
            return (elements,), logical, expert, True, folded
        if kind in ('qkv_group_interleave', 'gate_up_interleave'):
            names = value['sources']
            if not isinstance(names, list) or len(names) != (3 if kind == 'qkv_group_interleave' else 2):
                raise ValueError('projection packing source count differs')
            shapes = [source(name) for name in names]
            if any(len(shape) != 2 for shape in shapes) or len({shape[1] for shape in shapes}) != 1:
                raise ValueError('projection packing input widths differ')
            if kind == 'qkv_group_interleave':
                kv, q, d = (integer(value[k], k) for k in ('hkv', 'qperkv', 'head_dim'))
                model_kv = config['num_key_value_heads']
                model_q = config['num_attention_heads']
                model_d = config.get('head_dim', config['hidden_size']//model_q)
                if (kv, q, d) != (model_kv, model_q//model_kv, model_d):
                    raise ValueError('QKV head geometry differs from checkpoint config')
                if shapes != [(kv*q*d, shapes[0][1]), (kv*d, shapes[0][1]), (kv*d, shapes[0][1])]:
                    raise ValueError('QKV recipe dimensions differ from checkpoint shapes')
            else:
                unit = integer(value['u'], 'u')
                if shapes[0] != shapes[1] or shapes[0][0] % unit:
                    raise ValueError('gate/up recipe dimensions differ')
            shape = (sum(shape[0] for shape in shapes), shapes[0][1])
            return shape, shape, None, False, False
        raise ValueError('unsupported serving checkpoint recipe: '+str(kind))

    shape, _, _, _, _ = visit(recipe)
    return dict(shape=list(shape), elements=math.prod(shape), sources=sorted(sources))
