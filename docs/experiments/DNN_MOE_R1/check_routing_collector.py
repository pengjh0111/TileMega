#!/usr/bin/env python3
"""Check the streamed CUDA collector against a complete unchanged tiny HF model."""
import argparse
import hashlib
import json
from pathlib import Path

from tilemega.moe.routing_profile import sha, write_json


def prepare(work):
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(9167)
    config = Qwen3MoeConfig(hidden_size=64, intermediate_size=128,
        moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
        vocab_size=128, tie_word_embeddings=False)
    config._experts_implementation = 'eager'
    config._attn_implementation = 'sdpa'
    model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.save_pretrained(work / 'reference')
    tensors = {}
    for name, tensor in model.state_dict().items():
        if name.endswith('mlp.experts.gate_up_proj'):
            prefix = name.removesuffix('gate_up_proj')
            for e, weights in enumerate(tensor):
                for kind, value in zip(('gate', 'up'), weights.chunk(2, dim=0)):
                    tensors[prefix + f'{e}.{kind}_proj.weight'] = value.contiguous().clone()
        elif name.endswith('mlp.experts.down_proj'):
            prefix = name.removesuffix('down_proj')
            for e, weights in enumerate(tensor):
                tensors[prefix + f'{e}.down_proj.weight'] = weights.contiguous().clone()
        else:
            tensors[name] = tensor.contiguous().clone()
    source = work / 'source'
    source.mkdir()
    config.save_pretrained(source)
    shards = [{}, {}]
    index = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        shard = i % 2
        shards[shard][name] = tensor
        index[name] = f'shard{shard}.safetensors'
    for i, shard in enumerate(shards):
        save_file(shard, str(source / f'shard{i}.safetensors'))
    write_json(source / 'model.safetensors.index.json', dict(weight_map=index))
    states = work / 'states'
    states.mkdir()
    write_json(states / 'tokens.json', [torch.randint(0, config.vocab_size, (4096,)).tolist()])
    write_json(states / 'sampling.json', dict(config_sha256=sha(source / 'config.json'),
        layers=2, experts=4, top_k=2, capture_layers=[0, 1], context=4096,
        source='seeded tiny HF model and fixed IDs; no real-weight routing estimate'))


def check(work):
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3MoeForCausalLM
    states = work / 'states'
    model = Qwen3MoeForCausalLM.from_pretrained(work / 'reference',
        dtype=torch.bfloat16, attn_implementation='sdpa', experts_implementation='grouped_mm').to('cuda:0').eval()
    inputs = torch.tensor(json.loads((states / 'tokens.json').read_text()), device='cuda:0')
    captured = {}
    hooks = []
    for index, layer in enumerate(model.model.layers):
        def norm(module, args, index=index):
            captured[f'{index}.hidden'] = args[0].squeeze(0).cpu()
        def router(module, args, outputs, index=index):
            for name, value in zip(('logits', 'weights', 'indices'), outputs):
                captured[f'{index}.{name}'] = value.cpu()
        def output(module, args, value, index=index):
            captured[f'{index}.output'] = value.squeeze(0).cpu()
        hooks.extend([layer.post_attention_layernorm.register_forward_pre_hook(norm),
                      layer.mlp.gate.register_forward_hook(router),
                      layer.register_forward_hook(output)])
    with torch.inference_mode():
        model.model(inputs, use_cache=False)
    for hook in hooks:
        hook.remove()
    for index in range(2):
        region = load_file(str(states / f'region_{index:02d}.safetensors'))
        for name in ('hidden', 'logits', 'weights', 'indices'):
            expected = captured[f'{index}.{name}']
            actual = region[f'sequence_000.{name}']
            if not torch.equal(actual, expected):
                raise AssertionError(f'layer {index}: streamed {name} differs from full HF model')
        hidden = load_file(str(states / f'hidden_{index + 1:02d}.safetensors'))['sequence_000']
        if not torch.equal(hidden, captured[f'{index}.output']):
            raise AssertionError(f'layer {index}: streamed output differs from full HF model')
    profile = json.loads((work / 'profile.json').read_text())
    if len(profile['layers']) != 2 or any(len(row['coordinates']) != 13 for row in profile['layers']):
        raise AssertionError('missing routing coordinates')
    result = dict(evidence='verified', passed=True,
        scope='two-layer seeded HF model; actual CUDA grouped_mm; streamed hidden/router/output values bitwise equal to full HF; all 13 token coordinates; no real Qwen3 weight/model gate',
        profile_sha256=sha(work / 'profile.json'), profile_id=profile['profile_id'],
        device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        inputs={str(p): sha(p) for p in (Path(__file__), states / 'tokens.json', states / 'sampling.json')})
    write_json(work / 'result.json', result)
    print(json.dumps(dict(event='routing_collector_check_complete', passed=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'check'), required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    dict(prepare=prepare, check=check)[args.stage](args.work.resolve())


if __name__ == '__main__':
    main()
