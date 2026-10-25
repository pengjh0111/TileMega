#!/usr/bin/env python3
"""Teacher-forced MoE routing check for a native decode (prompt §8 item 5).

Runs inside `tilemega.serving.random_checkpoint` under the scheduler guard:
prefill + 64 decode steps one launch at a time, reading every layer's top-k
index buffer (the FX names `routing_check` uses) after each launch, then the
HF comparison. Mismatched sets are listed with HF's 8th/9th router logits so
true ties are separated from defects (U12). Diagnostic; not a timing path.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from tilemega.cache import atomic_json
from tilemega.moe.full import generation_tokens
from tilemega.moe.routing_check import agreement, capture, compare_hf, index_names
from tilemega.serving.engine import ServingEngine

ROOT = Path(__file__).resolve().parents[1]


def hf_ties(checkpoint, prompts, tokens, mismatches):
    """HF router logits ranked 8 and 9 at each mismatched (step, layer, batch)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='grouped_mm').cuda().eval()
    rows = []
    for b, prompt in enumerate(prompts):
        wanted = [m for m in mismatches if m['batch'] == b]
        if not wanted:
            continue
        logits = {}
        def hook(layer):
            def save(module, args, output):
                logits[layer] = output[0].detach().float().cpu()
            return save
        handles = [layer.mlp.gate.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]
        try:
            sequence = torch.tensor([prompt + tokens[b][:-1]], device='cuda', dtype=torch.long)
            with torch.inference_mode():
                model(sequence, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for m in wanted:
            row = logits[m['layer']][len(prompt) - 1 + m['step']].to(torch.bfloat16).float()
            top = row.topk(9).values.tolist()
            rows.append(dict(m, hf_logit_8=top[7], hf_logit_9=top[8], tie=top[7] == top[8]))
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--decode', type=Path, required=True)
    p.add_argument('--bridge', type=Path, required=True, help='native decode export.json')
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('GPU work requires the scheduler guard')
    checkpoint = Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    config = json.loads((checkpoint / 'config.json').read_text())
    layers, top_k = config['num_hidden_layers'], config['num_experts_per_tok']
    # The legacy prefill and the native decode name the index buffers alike.
    names = index_names(json.loads(a.bridge.read_text()), layers)
    rows = torch.tensor(json.loads((ROOT / 'configs/prompts/qwen_isl64.json').read_text())[:a.batch],
                        dtype=torch.int64)
    routing = []
    def capture_step(plan, phase):
        if len(routing) < a.steps:
            routing.append(capture(plan, names, a.batch, rows.shape[1] if phase == 'prefill' else 1, top_k))
    with ServingEngine(checkpoint, a.prefill, a.decode, a.batch, prompt_len=rows.shape[1],
                       max_new_tokens=a.steps, mode='L1', prefill_mode='L1', decode_loop=False,
                       step_events=False) as engine:
        tokens = generation_tokens(engine, rows, a.steps, 1, config['vocab_size'], capture_step)
    del engine
    torch.cuda.empty_cache()
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out / 'tokens.json', tokens)
    atomic_json(a.out / 'routing_indices.json', routing)
    prompts = rows.tolist()
    report = compare_hf(checkpoint, prompts, tokens, routing, a.out / 'routing.json')
    # compare_hf keeps no per-position table; rebuild HF's only when needed.
    expected, mismatches = None, []
    if report['exact_set_ratio'] < 1:
        expected = hf_indices(checkpoint, prompts, tokens, a.steps, layers)
        for s in range(a.steps):
            for l in range(layers):
                for b in range(a.batch):
                    if sorted(routing[s][l][b]) != sorted(expected[s][l][b]):
                        mismatches.append(dict(step=s, layer=l, batch=b, native=routing[s][l][b],
                                               hf=expected[s][l][b]))
        mismatches = hf_ties(checkpoint, prompts, tokens, mismatches)
    report.update(batch=a.batch, steps=a.steps, mismatches=mismatches,
                  ties=sum(m['tie'] for m in mismatches),
                  defects=sum(not m['tie'] for m in mismatches),
                  agreement_recheck=agreement(routing, expected) if expected else None)
    atomic_json(a.out / 'routing.json', report)
    print(json.dumps({k: report[k] for k in ('exact_set_ratio', 'exact_rank_ratio', 'ties', 'defects')}))
    return 0 if report['defects'] == 0 else 1


def hf_indices(checkpoint, prompts, tokens, steps, layers):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='grouped_mm').cuda().eval()
    table = [[[None] * len(prompts) for _ in range(layers)] for _ in range(steps)]
    for b, prompt in enumerate(prompts):
        captures = {}
        def hook(layer):
            def save(module, args, output):
                captures[layer] = output[2].detach().cpu()
            return save
        handles = [layer.mlp.gate.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]
        try:
            sequence = torch.tensor([prompt + tokens[b][:-1]], device='cuda', dtype=torch.long)
            with torch.inference_mode():
                model(sequence, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for s in range(steps):
            for l in range(layers):
                table[s][l][b] = captures[l][len(prompt) - 1 + s].tolist()
    del model
    torch.cuda.empty_cache()
    return table


if __name__ == '__main__':
    raise SystemExit(main())
