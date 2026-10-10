"""Qwen3 MoE export boundary and an original-FQN standalone MoE region."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


@torch.library.custom_op('tilemega::moe_experts', mutates_args=())
def moe_experts(h: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor,
                gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Reference with HF grouped_mm's BF16 projection/gate/product boundaries.

    Grouping changes row placement only. Restoring (token, rank) before the
    FP32 reduction avoids the BF16 index_add accumulation of HF eager experts.
    """
    tokens, hidden = h.shape
    experts, twice_inner, width = gate_up.shape
    if width != hidden or twice_inner % 2 or down.shape != (experts, hidden, twice_inner // 2):
        raise ValueError('MoE expert matrices disagree on geometry')
    if topk_idx.ndim != 2 or topk_idx.shape != topk_w.shape or topk_idx.shape[0] != tokens:
        raise ValueError('MoE routing tensors disagree on geometry')
    k = topk_idx.shape[1]
    if k < 1 or k > experts or torch.any((topk_idx < 0) | (topk_idx >= experts)):
        raise ValueError('MoE expert index outside the expert table')
    partials = torch.empty((tokens, k, hidden), device=h.device, dtype=h.dtype)
    for expert in range(experts):
        token, rank = torch.where(topk_idx == expert)
        if token.numel():
            gate, up = torch.nn.functional.linear(h[token], gate_up[expert]).chunk(2, -1)
            projected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down[expert])
            partials[token, rank] = projected * topk_w[token, rank, None]
    return partials.sum(1, dtype=torch.float32).to(h.dtype)


@moe_experts.register_fake
def _fake_experts(h, topk_idx, topk_w, gate_up, down):
    torch._check(h.ndim == 2)
    torch._check(gate_up.ndim == 3)
    torch._check(down.ndim == 3)
    torch._check(topk_idx.ndim == 2)
    torch._check(topk_w.shape == topk_idx.shape)
    torch._check(topk_idx.shape[0] == h.shape[0])
    torch._check(gate_up.shape[2] == h.shape[1])
    torch._check(down.shape[0] == gate_up.shape[0])
    torch._check(down.shape[1] == h.shape[1])
    torch._check(2 * down.shape[2] == gate_up.shape[1])
    return torch.empty_like(h)


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        e = int(config.get('num_experts', config.get('num_local_experts', 0)))
        h, i = int(config['hidden_size']), int(config['moe_intermediate_size'])
        if min(e, h, i) <= 0:
            raise ValueError('MoE export requires positive expert dimensions')
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2*i, h))
        self.down_proj = nn.Parameter(torch.empty(e, h, i))


class SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = Experts(config)
        e = self.experts.gate_up_proj.shape[0]
        self.gate = nn.Linear(config['hidden_size'], e, bias=False)
        self.top_k = int(config['num_experts_per_tok'])
        self.normalize = bool(config.get('norm_topk_prob', False))
        if not 1 <= self.top_k <= e:
            raise ValueError('MoE top-k exceeds the expert table')

    def forward(self, h):
        shape = h.shape
        rows = h.reshape(-1, shape[-1])
        logits = self.gate(rows)
        probability = torch.softmax(logits, -1, dtype=torch.float32)
        weight, index = torch.topk(probability, self.top_k, dim=-1)
        if self.normalize:
            weight = weight / weight.sum(-1, keepdim=True)
        weight = weight.to(logits.dtype)
        return moe_experts(rows, index, weight, self.experts.gate_up_proj,
                           self.experts.down_proj).reshape(shape)


class Region(nn.Module):
    """The nesting preserves model.layers.L parameter names in region exports."""
    def __init__(self, config, layer):
        super().__init__()
        from tilemega.serving.export import RMSNorm
        self.layer = int(layer)
        if not 0 <= self.layer < int(config['num_hidden_layers']):
            raise ValueError('region layer outside the model')
        block = nn.Module()
        block.post_attention_layernorm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        block.mlp = SparseMLP(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(layer): block})

    def forward(self, h):
        block = self.model.layers[str(self.layer)]
        return h + block.mlp(block.post_attention_layernorm(h))


def export_region(config, layer, out):
    # Export metadata only: even the full 128-expert layer allocates no weights.
    with torch.device('meta'):
        model = Region(config, layer).eval().to(torch.bfloat16)
        h = torch.empty((2, config['hidden_size']), dtype=torch.bfloat16)
    tokens = torch.export.Dim('tokens', min=1, max=4096)
    program = torch.export.export(model, (h,), dynamic_shapes=({0: tokens},), strict=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, out/'exported_program.pt2')
    manifest = dict(phase='forward', batch=1, tokens_range=[1, 4096], layer=layer,
        config=config, torch_version=torch.__version__, weight_fixture=None,
        inputs=['h'], outputs=['h + MoE(RMSNorm(h))'], custom_op='tilemega.moe_experts.default')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_region(json.loads(args.config.read_text()), args.layer, args.out)), flush=True)


if __name__ == '__main__':
    main()
