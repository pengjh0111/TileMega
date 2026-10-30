"""Check pinned decoder shapes on meta tensors without checkpoint weights.

This checks the exporter's actual modules and tensor operations. It does not
validate generated CUDA, numerical values, model quality or performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from .export import Decoder
from .model_registry import ROOT, load_model, model_ids, verify_config
from .workload import Workload


def _shape(tensor, expected, label):
    actual = tuple(tensor.shape)
    if actual != tuple(expected):
        raise ValueError(f'{label}: expected {tuple(expected)}, got {actual}')
    if tensor.device.type != 'meta':
        raise ValueError(f'{label}: structural checks must not allocate weights')


def inspect(config: dict, points: tuple[Workload, ...]) -> dict:
    """Run all layers at the requested shapes, retaining no tensor values."""
    hidden, vocab = config['hidden_size'], config['vocab_size']
    heads, kv = config['num_attention_heads'], config['num_key_value_heads']
    dim = config.get('head_dim', hidden // heads)
    if heads % kv or not 1 <= heads // kv <= 8:
        raise ValueError('model requires an integral QPerKV in [1, 8]')
    layers = config['num_hidden_layers']
    moe = config['architectures'][0] in ('Qwen3MoeForCausalLM','OlmoeForCausalLM')
    with torch.device('meta'):
        model = Decoder(config).eval().to(torch.bfloat16)
    if len(model.model.layers) != layers:
        raise ValueError('exported layer count differs from config')
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    if tied != config.get('tie_word_embeddings', False):
        raise ValueError('exported lm_head tying differs from config')
    _shape(model.lm_head.weight, (vocab, hidden), 'lm_head')
    for index, layer in enumerate(model.model.layers):
        for name, shape in (('q_proj', (heads * dim, hidden)),
                            ('k_proj', (kv * dim, hidden)),
                            ('v_proj', (kv * dim, hidden)),
                            ('o_proj', (hidden, heads * dim))):
            _shape(getattr(layer.self_attn, name).weight, shape, f'layer {index} {name}')
        if layer.self_attn.has_qk_norm:
            for name in ('q_norm', 'k_norm'):
                _shape(getattr(layer.self_attn, name).weight, (dim,), f'layer {index} {name}')
        if layer.self_attn.row_qk_norm:
            _shape(layer.self_attn.q_norm.weight,(heads*dim,),f'layer {index} row q norm')
            _shape(layer.self_attn.k_norm.weight,(kv*dim,),f'layer {index} row k norm')
        if moe:
            experts = config.get('num_experts', config.get('num_local_experts'))
            inner = config.get('moe_intermediate_size') or config['intermediate_size']
            _shape(layer.mlp.gate.weight, (experts, hidden), f'layer {index} router')
            _shape(layer.mlp.experts.gate_up_proj, (experts, 2 * inner, hidden),
                   f'layer {index} expert gate/up')
            _shape(layer.mlp.experts.down_proj, (experts, hidden, inner),
                   f'layer {index} expert down')
        else:
            inner = config['intermediate_size']
            for name, shape in (('gate_proj', (inner, hidden)),
                                ('up_proj', (inner, hidden)),
                                ('down_proj', (hidden, inner))):
                _shape(getattr(layer.mlp, name).weight, shape, f'layer {index} {name}')
    rows = []
    with torch.inference_mode(), torch.device('meta'):
        for point in points:
            phases = [('prefill', point.isl, 0)]
            phases += [('decode', 1, past) for past in
                       sorted({*point.pasts, point.isl + point.osl - 2})
                       if point.osl > 1]
            for batch in point.batches:
                for phase, seq, past in phases:
                    tokens = torch.empty((batch, seq), dtype=torch.int32)
                    caches = tuple(torch.empty((batch, kv, past, dim), dtype=torch.bfloat16)
                                   for _ in range(2 * layers))
                    rope = torch.empty((point.capacity, dim), dtype=torch.bfloat16)
                    outputs = model.execute(tokens, caches, rope, rope)
                    if len(outputs) != 1 + 2 * layers:
                        raise ValueError('exported cache count differs from layer count')
                    _shape(outputs[0], (batch, seq, vocab), 'logits')
                    for index, value in enumerate(outputs[1:]):
                        _shape(value, (batch, kv, past + seq, dim), f'cache {index}')
                    rows.append(dict(isl=point.isl, osl=point.osl, capacity=point.capacity,
                                     batch=batch, phase=phase, seq=seq, past=past,
                                     logits_shape=list(outputs[0].shape),
                                     cache_shape=list(outputs[1].shape)))
    parameters = list(model.parameters())  # Shared embeddings count once.
    return dict(scope='meta structure only; no generated CUDA or numerical gate',
                structure_passed=True, model_gate_passed=False, layers=layers, qperkv=heads // kv,
                q_width=heads * dim, hidden_size=hidden, tied_lm_head=tied,
                attention_signature=dict(head_dim=dim, qperkv=heads // kv,
                                         qk_norm=model.model.layers[0].self_attn.has_qk_norm),
                lm_head_fqn='model.embed_tokens.weight' if tied else 'lm_head.weight',
                logical_bf16_parameter_bytes=sum(p.numel() * p.element_size() for p in parameters),
                allocated_weight_bytes=0, checkpoint_weight_reads=0, cases=rows)


def run(model_root: Path, names: tuple[str, ...], points: tuple[Workload, ...],
        repo_root: Path = ROOT) -> dict:
    reports = {}
    for name in names:
        record = load_model(name, repo_root)
        config = verify_config(record, model_root / record['id'] / 'config.json')
        reports[record['id']] = dict(inspect(config, points),
            config_sha256=record['config_sha256'], revision=record['revision'])
    source_paths = ('python/tilemega/serving/export.py',
                    'python/tilemega/moe/export.py',
                    'python/tilemega/serving/workload.py',
                    'python/tilemega/serving/structure_check.py')
    head = subprocess.check_output(['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
                                   text=True).strip()
    return dict(scope='structure_only', head=head, model_gate_passed=False,
                source_sha256={name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
                               for name in source_paths}, models=reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=Path('/root/models'))
    parser.add_argument('--models', default=','.join(model_ids()))
    parser.add_argument('--points', default='64:1024,128:2048')
    parser.add_argument('--batches', default='1,16')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    batches = tuple(map(int, args.batches.split(',')))
    points = tuple(Workload(*map(int, point.split(':')), batches=batches)
                   for point in args.points.split(','))
    report = run(args.model_root, tuple(args.models.split(',')), points)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Checked {len(report["models"])} model structures without weights; G-MODEL not evaluated')


if __name__ == '__main__':
    main()
