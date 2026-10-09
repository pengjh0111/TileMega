"""Collect real HF routing and region inputs with one decoder layer resident.

The queue has a separate, locked process for embedding and every decoder
layer. Hidden states cross these boundaries on disk, without retaining an
earlier layer's device weights. Windows share a causal 4096-token context;
their token positions and dependence are retained in the sampling manifest.
No latency measurements are made here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

TOKENS = tuple(1 << i for i in range(13))
BLOCK_ROWS = (16, 32, 64, 128)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def routing_statistics(selections, experts, top_k, tokens=TOKENS, block_rows=BLOCK_ROWS):
    """Count disjoint windows within each original sequence, including idle experts."""
    import torch
    if experts < 1 or not 1 <= top_k <= experts:
        raise ValueError('invalid expert/top-k counts')
    selected = [torch.as_tensor(x, device='cpu') for x in selections]
    if not selected or not tokens or len(set(tokens)) != len(tokens):
        raise ValueError('empty routing observations or duplicate token coordinates')
    if not block_rows or len(set(block_rows)) != len(block_rows) or any(
            type(b) is not int or b < 1 for b in block_rows):
        raise ValueError('invalid or duplicate binding block sizes')
    for x in selected:
        if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('routing observations must contain integer expert indices')
        if x.ndim != 2 or x.shape[1] != top_k or not x.shape[0]:
            raise ValueError('routing observation must have shape [sequence, top_k]')
        if (x < 0).any() or (x >= experts).any():
            raise ValueError('expert index outside checkpoint')
        if top_k > 1 and (x.sort(dim=-1).values.diff(dim=-1) == 0).any():
            raise ValueError('duplicate expert within one token')
    result = {}
    for t in tokens:
        if t < 1 or any(len(x) % t for x in selected):
            raise ValueError('token windows must divide every source sequence')
        distinct = Counter()
        per_expert = [Counter() for _ in range(experts)]
        group_blocks = {b: Counter() for b in block_rows}
        virtual_rows = {b: [0]*((t*top_k+b-1)//b + min(experts,t*top_k)) for b in block_rows}
        row_histograms = {b: [Counter() for _ in sums] for b, sums in virtual_rows.items()}
        windows = 0
        for x in selected:
            for block in x.reshape(-1, t * top_k):
                counts = torch.bincount(block.to(torch.int64), minlength=experts).tolist()
                distinct[sum(n > 0 for n in counts)] += 1
                for e, n in enumerate(counts):
                    per_expert[e][n] += 1
                # Marginals determine mean work, but static placement also
                # needs the joint distribution of the active virtual prefix.
                active_counts = [n for n in counts if n]
                for b, histogram in group_blocks.items():
                    histogram[sum((n + b - 1) // b for n in active_counts)] += 1
                    cursor = 0
                    for n in active_counts:
                        full, tail = divmod(n, b)
                        for _ in range(full):
                            virtual_rows[b][cursor] += b
                            row_histograms[b][cursor][b] += 1
                            cursor += 1
                        if tail:
                            virtual_rows[b][cursor] += tail
                            row_histograms[b][cursor][tail] += 1
                            cursor += 1
                windows += 1
        histogram = lambda c: {str(k): v for k, v in sorted(c.items())}
        result[str(t)] = dict(tokens=t, windows=windows,
            distinct_experts_histogram=histogram(distinct),
            expected_distinct_experts=sum(k * n for k, n in distinct.items()) / windows,
            tokens_per_expert_histograms=[histogram(c) for c in per_expert],
            group_blocks_histograms={str(b): histogram(c) for b, c in group_blocks.items()},
            virtual_rows_totals={str(b): sums for b, sums in virtual_rows.items()},
            virtual_rows_histograms={str(b): [histogram(h) for h in hs] for b, hs in row_histograms.items()},
            assignments_per_window=t * top_k)
    return result


class IndexedCheckpoint:
    """Read individual original FQNs without materializing a shard on the device."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.index_path = self.directory / 'model.safetensors.index.json'
        self.index = json.loads(self.index_path.read_text())['weight_map']
        self.used = {}
        for name, shard in self.index.items():
            if not isinstance(name, str) or not isinstance(shard, str):
                raise ValueError('checkpoint index entries must be strings')
            path = self.directory / shard
            if path.resolve().parent != self.directory or not path.is_file():
                raise ValueError(f'missing or unsafe checkpoint shard: {shard}')

    def tensor(self, name):
        import torch
        from safetensors import safe_open
        path = self.directory / self.index[name]
        with safe_open(path, framework='pt', device='cpu') as stream:
            tensor = stream.get_tensor(name)
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f'expected original BF16 weight: {name}')
        digest = hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()
        self.used[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest)
        return tensor

    def identity(self):
        return dict(directory=str(self.directory), index_sha256=sha(self.index_path),
                    config_sha256=sha(self.directory / 'config.json'), tensors=self.used)


def load_hf_layer(checkpoint, config, index, device):
    """Bind the unchanged HF layer's stacked parameters to original expert FQNs."""
    import torch
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
    if not 0 <= index < config.num_hidden_layers:
        raise ValueError('layer index outside model')
    with torch.device('meta'):
        layer = Qwen3MoeDecoderLayer(config, index).to(dtype=torch.bfloat16)
    prefix = f'model.layers.{index}.'
    consumed = set()
    def read(name):
        consumed.add(name)
        return checkpoint.tensor(name)
    for name, parameter in list(layer.named_parameters()):
        target = torch.empty(parameter.shape, dtype=torch.bfloat16, device=device)
        if name in ('mlp.experts.gate_up_proj', 'mlp.experts.down_proj'):
            for expert in range(config.num_experts):
                expert_prefix = prefix + f'mlp.experts.{expert}.'
                if name.endswith('gate_up_proj'):
                    for half, kind in enumerate(('gate', 'up')):
                        value = read(expert_prefix + kind + '_proj.weight')
                        destination = target[expert, half * config.moe_intermediate_size:
                                             (half + 1) * config.moe_intermediate_size]
                        if destination.shape != value.shape:
                            raise ValueError(f'checkpoint shape differs: {expert_prefix}{kind}')
                        destination.copy_(value)
                        del value
                else:
                    value = read(expert_prefix + 'down_proj.weight')
                    if target[expert].shape != value.shape:
                        raise ValueError(f'checkpoint shape differs: {expert_prefix}down')
                    target[expert].copy_(value)
                    del value
        else:
            value = read(prefix + name)
            if target.shape != value.shape:
                raise ValueError(f'checkpoint shape differs: {prefix}{name}')
            target.copy_(value)
            del value
        parent, _, attribute = name.rpartition('.')
        layer.get_submodule(parent).register_parameter(
            attribute, torch.nn.Parameter(target, requires_grad=False))
    expected = {name for name in checkpoint.index if name.startswith(prefix)}
    if expected != consumed:
        raise ValueError('HF layer parameters do not cover original checkpoint layer')
    return layer.eval()


def token_sequences(prompts, tokenizer, wiki_texts, context=4096):
    """Retain each R10 ID sequence, then append Qwen-tokenized validation text."""
    if not prompts or context < 1 or any(not row for row in prompts):
        raise ValueError('nonempty original prompts and positive context required')
    filler = []
    needed = sum(max(0, context - len(row)) for row in prompts)
    if needed:
        for text in wiki_texts:
            if text:
                filler.extend(tokenizer.encode(text, add_special_tokens=False))
                if len(filler) >= needed:
                    break
        if len(filler) < needed:
            raise ValueError('WikiText-103 validation data cannot fill requested contexts')
    offset = 0
    result = []
    for row in prompts:
        extra = max(0, context - len(row))
        result.append(list(row[:context]) + filler[offset:offset + extra])
        offset += extra
    vocab = len(tokenizer)
    if any(not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < vocab
           for row in result for t in row):
        raise ValueError('prompt token outside Qwen3 tokenizer vocabulary')
    return result


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompts = json.loads(args.prompts.read_text())
    wiki = pq.read_table(args.wiki, columns=['text']).column('text').to_pylist()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_moe' or config.get('num_experts', 0) < 1:
        raise ValueError('routing collection requires Qwen3-MoE configuration')
    if any(not 0 <= index < config['num_hidden_layers'] for index in args.capture_layers):
        raise ValueError('region capture layer outside model configuration')
    sequences = token_sequences(prompts, tokenizer, wiki)
    write_json(args.work / 'tokens.json', sequences)
    tokenizer_files = {p.name: sha(p) for p in sorted(args.checkpoint.iterdir())
                       if p.is_file() and ('token' in p.name or p.name in ('vocab.json', 'merges.txt'))
                       and not p.name.endswith(('.log', '.json.tmp')) and '.download' not in p.name}
    write_json(args.work / 'sampling.json', dict(schema='tilemega.dm1.routing.sampling.v1',
        prompts=dict(path=str(args.prompts.resolve()), sha256=sha(args.prompts),
                     lengths=[len(row) for row in prompts]),
        wiki=dict(path=str(args.wiki.resolve()), sha256=sha(args.wiki)),
        config_sha256=sha(args.checkpoint / 'config.json'), tokenizer_files=tokenizer_files,
        tokens_sha256=sha(args.work / 'tokens.json'), context=4096, sequences=len(sequences),
        windows='disjoint contiguous windows per sequence in a shared causal context',
        filler='consecutive non-repeated WikiText-103 validation tokens after each R10 prompt',
        capture_layers=args.capture_layers, layers=config['num_hidden_layers'],
        experts=config['num_experts'], top_k=config['num_experts_per_tok']))


def capture_identity(checkpoint, args, inputs):
    import inspect
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from transformers.integrations import moe
    return dict(schema='tilemega.dm1.routing.identity.v1', checkpoint=checkpoint.identity(),
        implementation=dict(script_sha256=sha(__file__), torch=version('torch'),
            transformers=version('transformers'), attention='sdpa', experts='grouped_mm',
            hf_model_source_sha256=sha(inspect.getfile(modeling_qwen3_moe)),
            hf_experts_source_sha256=sha(inspect.getfile(moe))),
        inputs={str(p): sha(p) for p in inputs}, device=str(args.device))


def embed(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    checkpoint = IndexedCheckpoint(args.checkpoint)
    sequences = json.loads((args.work / 'tokens.json').read_text())
    with torch.inference_mode():
        weight = checkpoint.tensor('model.embed_tokens.weight').to(args.device)
        states = {f'sequence_{i:03d}': F.embedding(
            torch.tensor(row, dtype=torch.int64, device=args.device), weight).cpu()
            for i, row in enumerate(sequences)}
        save_file(states, str(args.work / 'hidden_00.safetensors'))
    identity = capture_identity(checkpoint, args, [args.work / 'tokens.json', args.work / 'sampling.json'])
    identity['output_sha256'] = sha(args.work / 'hidden_00.safetensors')
    write_json(args.work / 'embedding.identity.json', identity)


def collect_layer(args):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3MoeConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
    config = Qwen3MoeConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config._attn_implementation = 'sdpa'
    config._experts_implementation = 'grouped_mm'
    checkpoint = IndexedCheckpoint(args.checkpoint)
    incoming = args.work / f'hidden_{args.layer:02d}.safetensors'
    states = load_file(str(incoming))
    outputs, observations, region = {}, [], {}
    with torch.inference_mode():
        layer = load_hf_layer(checkpoint, config, args.layer, args.device)
        rotary = Qwen3MoeRotaryEmbedding(config).to(args.device)
        captured = {}
        def before_norm(module, inputs):
            if args.layer in args.capture_layers:
                captured['hidden'] = inputs[0].squeeze(0).cpu().contiguous()
        def after_router(module, inputs, output):
            captured['logits'], captured['weights'], captured['indices'] = [x.cpu().contiguous() for x in output]
        hooks = [layer.post_attention_layernorm.register_forward_pre_hook(before_norm),
                 layer.mlp.gate.register_forward_hook(after_router)]
        try:
            for name, cpu_hidden in sorted(states.items()):
                captured.clear()
                hidden = cpu_hidden.unsqueeze(0).to(args.device)
                positions = torch.arange(hidden.shape[1], device=args.device).unsqueeze(0)
                mask = create_causal_mask(config=config, inputs_embeds=hidden,
                    attention_mask=None, past_key_values=None, position_ids=positions)
                result = layer(hidden, attention_mask=mask, position_ids=positions,
                    position_embeddings=rotary(hidden, positions), use_cache=False)
                if not torch.isfinite(result).all():
                    raise ValueError(f'nonfinite HF output at layer {args.layer}, {name}')
                outputs[name] = result.squeeze(0).cpu().contiguous()
                observations.append(captured['indices'])
                if args.layer in args.capture_layers:
                    region.update({f'{name}.{key}': value for key, value in captured.items()})
                del hidden, result, mask
        finally:
            for hook in hooks:
                hook.remove()
        torch.cuda.synchronize(args.device)
        peak = dict(allocated=torch.cuda.max_memory_allocated(args.device),
                    reserved=torch.cuda.max_memory_reserved(args.device))
    output = args.work / f'hidden_{args.layer + 1:02d}.safetensors'
    save_file(outputs, str(output))
    save_file({f'sequence_{i:03d}': x for i, x in enumerate(observations)},
              str(args.work / f'routing_{args.layer:02d}.safetensors'))
    if region:
        save_file(region, str(args.work / f'region_{args.layer:02d}.safetensors'))
    identity = capture_identity(checkpoint, args, [incoming, args.work / 'sampling.json'])
    identity.update(layer=args.layer, peak_device_bytes=peak, output_sha256=sha(output),
                    routing_sha256=sha(args.work / f'routing_{args.layer:02d}.safetensors'))
    if region:
        identity['region_sha256'] = sha(args.work / f'region_{args.layer:02d}.safetensors')
    write_json(args.work / f'layer_{args.layer:02d}.json', dict(layer=args.layer,
        coordinates=routing_statistics(observations, config.num_experts, config.num_experts_per_tok),
        identity=identity))


def finish(args):
    sampling = json.loads((args.work / 'sampling.json').read_text())
    embedding = json.loads((args.work / 'embedding.identity.json').read_text())
    layers = [json.loads((args.work / f'layer_{i:02d}.json').read_text())
              for i in range(sampling['layers'])]
    previous = embedding['output_sha256']
    sampling_hash = sha(args.work / 'sampling.json')
    if embedding['inputs'][str(args.work / 'sampling.json')] != sampling_hash:
        raise ValueError('sampling manifest changed after embedding collection')
    if embedding['inputs'][str(args.work / 'tokens.json')] != sha(args.work / 'tokens.json'):
        raise ValueError('tokens changed after embedding collection')
    for i, layer in enumerate(layers):
        identity = layer['identity']
        if layer['layer'] != i or identity['inputs'][str(args.work / f'hidden_{i:02d}.safetensors')] != previous:
            raise ValueError('broken streamed hidden-state identity chain')
        if sha(args.work / f'hidden_{i:02d}.safetensors') != previous:
            raise ValueError('hidden-state artifact changed after collection')
        if identity['inputs'][str(args.work / 'sampling.json')] != sampling_hash:
            raise ValueError('layer used a different sampling manifest')
        if identity['checkpoint']['index_sha256'] != embedding['checkpoint']['index_sha256']:
            raise ValueError('checkpoint index changed between layers')
        if identity['checkpoint']['config_sha256'] != sampling['config_sha256']:
            raise ValueError('checkpoint configuration changed between layers')
        if identity['implementation'] != embedding['implementation']:
            raise ValueError('HF/tool implementation changed between layers')
        if sha(args.work / f'routing_{i:02d}.safetensors') != identity['routing_sha256']:
            raise ValueError('routing artifact changed after collection')
        if 'region_sha256' in identity and sha(args.work / f'region_{i:02d}.safetensors') != identity['region_sha256']:
            raise ValueError('region-input artifact changed after collection')
        previous = identity['output_sha256']
    if sha(args.work / f'hidden_{sampling["layers"]:02d}.safetensors') != previous:
        raise ValueError('final hidden-state artifact changed after collection')
    result = dict(schema='tilemega.dm1.routing.profile.v1', evidence='verified',
        scope='HF BF16 grouped_mm routing; causal token windows; no TileMega numerical or timing gate',
        sampling=sampling, embedding_identity=embedding, layers=layers)
    result['profile_id'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write_json(args.output, result)


def make_queue(args):
    config = json.loads((args.checkpoint / 'config.json').read_text())
    command = [sys.executable, '-m', 'tilemega.moe.routing_profile']
    common = ['--checkpoint', str(args.checkpoint.resolve()), '--work', str(args.work.resolve()),
              '--prompts', str(args.prompts.resolve()), '--wiki', str(args.wiki.resolve()),
              '--output', str(args.output.resolve()), '--device', args.device,
              '--capture-layers', *map(str, args.capture_layers)]
    stages = [('prepare', None), ('embed', None)] + [('layer', i) for i in range(config['num_hidden_layers'])] + [('finish', None)]
    queue = []
    for stage, index in stages:
        name = 'routing_' + stage + (f'_{index:02d}' if index is not None else '')
        invocation = command + common + ['--stage', stage]
        gpu = stage in ('embed', 'layer')
        if index is not None:
            invocation += ['--layer', str(index)]
        if gpu:
            invocation = ['flock', str(args.lock), *invocation]
        step = dict(name=name, command=invocation, cwd=str(Path.cwd()),
                    env={'PYTHONPATH': str(Path(__file__).resolve().parents[2])},
                    gpu=False, timeout_s=7200)
        if queue:
            step['after'] = [queue[-1]['name']]
        queue.append(step)
    write_json(args.queue / 'queue_all.json', queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('queue', 'prepare', 'embed', 'layer', 'finish'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--wiki', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lock', type=Path, default=Path('/root/r14_work/gpu.lock'))
    parser.add_argument('--capture-layers', nargs='+', type=int, default=[0, 24, 47])
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == 'queue' and args.queue is None:
        parser.error('--queue is required for queue generation')
    if args.stage == 'layer' and args.layer is None:
        parser.error('--layer is required for a layer task')
    if args.stage in ('embed', 'layer') and not args.device.startswith('cuda:'):
        parser.error('production collection requires a CUDA device')
    actions = dict(queue=make_queue, prepare=prepare, embed=embed, layer=collect_layer, finish=finish)
    actions[args.stage](args)
    print(json.dumps(dict(event='routing_stage_complete', stage=args.stage, layer=args.layer)), flush=True)


if __name__ == '__main__':
    main()
