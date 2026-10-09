"""Reshard original Qwen3-MoE checkpoints without changing HF model code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_tensors(config):
    if config.get('model_type') != 'qwen3_moe' or config.get('mlp_only_layers', []) or \
            config.get('decoder_sparse_step', 1) != 1:
        raise ValueError('checkpoint tools require every decoder layer to be Qwen3 MoE')
    config = dict(config)
    if 'num_experts' not in config:
        config['num_experts'] = config['num_local_experts']
    elif 'num_local_experts' in config and config['num_experts'] != config['num_local_experts']:
        raise ValueError('checkpoint expert-count aliases disagree')
    h, i, e, l, v, q, kv = (int(config[key]) for key in (
        'hidden_size', 'moe_intermediate_size', 'num_experts', 'num_hidden_layers',
        'vocab_size', 'num_attention_heads', 'num_key_value_heads'))
    d = int(config.get('head_dim', h//q))
    if min(h,i,e,l,v,q,kv,d) <= 0 or not 0 < int(config['num_experts_per_tok']) <= e or q % kv:
        raise ValueError('invalid Qwen3-MoE checkpoint geometry')
    result = {'model.embed_tokens.weight': [v,h], 'model.norm.weight': [h]}
    if not config.get('tie_word_embeddings', False):
        result['lm_head.weight'] = [v,h]
    for layer in range(l):
        prefix = f'model.layers.{layer}.'
        for name in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
            result[prefix+name] = [h]
        for name, shape in [('q_proj', [q*d,h]), ('k_proj', [kv*d,h]),
                            ('v_proj', [kv*d,h]), ('o_proj', [h,q*d])]:
            result[prefix+'self_attn.'+name+'.weight'] = shape
            if config.get('attention_bias', False):
                result[prefix+'self_attn.'+name+'.bias'] = [shape[0]]
        for name in ('q_norm', 'k_norm'):
            result[prefix+'self_attn.'+name+'.weight'] = [d]
        result[prefix+'mlp.gate.weight'] = [e,h]
        for expert in range(e):
            for name, shape in [('gate', [i,h]), ('up', [i,h]), ('down', [h,i])]:
                result[prefix+f'mlp.experts.{expert}.{name}_proj.weight'] = shape
    return result


def inventory(directory):
    directory = Path(directory).resolve()
    index_path = directory/'model.safetensors.index.json'
    document = json.loads(index_path.read_text())
    index = document['weight_map']
    if not isinstance(index, dict) or not index:
        raise ValueError('checkpoint has no tensor index')
    headers = {}; total = 0; result = {}
    for shard in sorted(set(index.values())):
        path = directory/shard
        if not isinstance(shard, str) or path.resolve().parent != directory:
            raise ValueError('unsafe checkpoint shard path')
        with path.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('truncated safetensors length')
            length = struct.unpack('<Q', raw)[0]
            if length > min(path.stat().st_size-8, 100*1024*1024):
                raise ValueError('invalid safetensors header size')
            header = json.loads(stream.read(length))
        tensors = {name: value for name,value in header.items() if name != '__metadata__'}
        expected = {name for name,file in index.items() if file == shard}
        if set(tensors) != expected:
            raise ValueError('shard header and checkpoint index disagree: '+shard)
        intervals = []
        for name, entry in tensors.items():
            shape, offsets = entry['shape'], entry['data_offsets']
            if entry['dtype'] != 'BF16' or any(type(x) is not int or x < 1 for x in shape):
                raise ValueError('checkpoint tensor must be nonempty BF16: '+name)
            if len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
                raise ValueError('invalid tensor byte interval: '+name)
            size = 2*math.prod(shape)
            if offsets[1]-offsets[0] != size or 8+length+offsets[1] > path.stat().st_size:
                raise ValueError('checkpoint tensor byte interval differs from shape: '+name)
            intervals.append((*offsets,name));total += size
            result[name] = dict(shape=shape, dtype='BF16', bytes=size, shard=shard)
        end = 0
        for begin,stop,name in sorted(intervals):
            if begin != end:
                raise ValueError('checkpoint tensor intervals overlap or leave a gap: '+name)
            end = stop
        if 8+length+end != path.stat().st_size:
            raise ValueError('checkpoint shard contains trailing bytes: '+shard)
        headers[shard] = dict(header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),
                              bytes=path.stat().st_size)
    if document.get('metadata', {}).get('total_size', total) != total:
        raise ValueError('checkpoint total_size differs from tensor headers')
    return dict(index_sha256=_sha(index_path), total_bytes=total, tensors=result, shards=headers)


def index_check(directory, recipes=None, expected_bytes=None):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    expected = expected_tensors(config); actual = inventory(directory)
    if set(expected) != set(actual['tensors']):
        raise ValueError('checkpoint names differ: missing='+str(sorted(set(expected)-set(actual['tensors'])))+
                         ' extra='+str(sorted(set(actual['tensors'])-set(expected))))
    for name, shape in expected.items():
        if actual['tensors'][name]['shape'] != shape:
            raise ValueError('checkpoint shape differs from config: '+name)
    if expected_bytes is not None and actual['total_bytes'] != expected_bytes:
        raise ValueError('checkpoint byte total differs from required identity')
    if recipes is not None:
        from tilemega.serving.weights import _recipe_sources
        used = set()
        for recipe in recipes.values():
            used.update(_recipe_sources(recipe))
        if used-set(expected):
            raise ValueError('plan recipes name absent checkpoint tensors: '+str(sorted(used-set(expected))))
        actual['recipe_sources'] = sorted(used)
    return dict(evidence='verified', passed=True, config_sha256=_sha(directory/'config.json'), **actual)


def rewrite(source, output, *, layers=None, experts=None, shard_bytes=2*1024**3):
    """Hold at most one output shard plus the current original tensor in host RAM."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or shard_bytes < 1:
        raise ValueError('output must be a new directory and shard_bytes must be positive')
    verified = index_check(source)
    config = json.loads((source/'config.json').read_text())
    # HF 5.19 serializes its attribute-map destination; original hub configs
    # use num_experts. Emit the original spelling with one unambiguous value.
    config['num_experts'] = int(config.get('num_experts', config.get('num_local_experts', 0)))
    config.pop('num_local_experts', None)
    if (layers is None) == (experts is None):
        raise ValueError('choose exactly one of layer truncation or expert slicing')
    if layers is not None:
        if type(layers) is not int or not 1 <= layers <= config['num_hidden_layers']:
            raise ValueError('truncated layer count outside checkpoint')
        config['num_hidden_layers'] = layers
    if experts is not None:
        if type(experts) is not int or not config['num_experts_per_tok'] <= experts <= config['num_experts']:
            raise ValueError('sliced expert count outside checkpoint/top-k')
        config['num_experts'] = experts
    wanted = expected_tensors(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.json','.txt','.model') and path.name != 'model.safetensors.index.json':
                shutil.copy2(path, temporary/path.name)
        (temporary/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        weight_map = {}; pending = {}; pending_bytes = 0; total = 0; shards = []
        def flush():
            nonlocal pending, pending_bytes
            if not pending:
                return
            name = f'model-{len(shards)+1:05d}.safetensors'
            save_file(pending, str(temporary/name), metadata={'format':'pt'})
            weight_map.update({key:name for key in pending})
            shards.append(dict(name=name, sha256=_sha(temporary/name), tensor_bytes=pending_bytes))
            pending = {};pending_bytes = 0
        for name in sorted(wanted):
            original = verified['tensors'][name]
            with safe_open(source/original['shard'], framework='pt', device='cpu') as stream:
                tensor = stream.get_tensor(name)
                if experts is not None and re.fullmatch(r'model\.layers\.\d+\.mlp\.gate\.weight', name):
                    tensor = tensor[:experts].clone()
                else:
                    tensor = tensor.contiguous()
            if list(tensor.shape) != wanted[name]:
                raise ValueError('rewritten tensor disagrees with config: '+name)
            size = tensor.numel()*tensor.element_size()
            if pending and pending_bytes+size > shard_bytes:
                flush()
            pending[name] = tensor;pending_bytes += size;total += size
        flush()
        (temporary/'model.safetensors.index.json').write_text(json.dumps(
            dict(metadata=dict(total_size=total), weight_map=weight_map), indent=2, sort_keys=True)+'\n')
        result = index_check(temporary)
        identity = dict(schema='tilemega.moe.checkpoint.v1', evidence='verified',
            source=str(source), source_index_sha256=verified['index_sha256'],
            source_config_sha256=verified['config_sha256'], layers=config['num_hidden_layers'],
            experts=config['num_experts'], top_k=config['num_experts_per_tok'], total_bytes=total,
            shards=shards, output_index_sha256=result['index_sha256'])
        (temporary/'tilemega_checkpoint.json').write_text(json.dumps(identity, indent=2)+'\n')
        temporary.rename(output)
        return identity
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('truncate','expert_slice'):
        sub = commands.add_parser(name);sub.add_argument('--source', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--layers' if name == 'truncate' else '--experts', type=int,
                         default=12 if name == 'truncate' else 16)
        sub.add_argument('--shard-bytes', type=int, default=2*1024**3)
    sub = commands.add_parser('index_check');sub.add_argument('--source', type=Path, required=True)
    sub.add_argument('--recipes', type=Path);sub.add_argument('--expected-bytes', type=int)
    args = parser.parse_args()
    if args.command == 'index_check':
        result = index_check(args.source, json.loads(args.recipes.read_text()) if args.recipes else None,
                             args.expected_bytes)
        result = {key:value for key,value in result.items() if key not in ('tensors','shards')}
    else:
        result = rewrite(args.source, args.output, layers=args.layers if args.command == 'truncate' else None,
                         experts=args.experts if args.command == 'expert_slice' else None,
                         shard_bytes=args.shard_bytes)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
