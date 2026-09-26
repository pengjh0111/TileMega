"""Export a serving decoder without loading checkpoint tensors.

The exported graph retains HF parameter FQNs and functional cache outputs.
The serving frontend will replace those outputs with persistent, in-place
state; no random weights or fixtures are written beside the export.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.epsilon)
        return self.weight * value.to(x.dtype)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        heads = config["num_attention_heads"]
        kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", hidden // heads)
        self.heads = heads
        self.kv_heads = kv_heads
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.has_qk_norm = config["architectures"][0].startswith("Qwen3")
        if self.has_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])
            self.k_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])

    def forward(self, x: torch.Tensor, past_k: torch.Tensor,
                past_v: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        past = past_k.shape[2]
        q = self.q_proj(x).view(batch, seq, self.heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim)
        if self.has_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        positions = torch.arange(past, past + seq, device=x.device)
        cosine = rope_cos.index_select(0, positions).view(1, seq, 1, self.head_dim)
        sine = rope_sin.index_select(0, positions).view(1, seq, 1, self.head_dim)
        q = q * cosine + rotate_half(q) * sine
        k = k * cosine + rotate_half(k) * sine
        full_k = torch.cat((past_k, k.transpose(1, 2)), dim=2)
        full_v = torch.cat((past_v, v.transpose(1, 2)), dim=2)
        query = q.transpose(1, 2)
        keys = torch.repeat_interleave(full_k, self.heads // self.kv_heads, dim=1)
        values = torch.repeat_interleave(full_v, self.heads // self.kv_heads, dim=1)
        score = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        key_position = torch.arange(past + seq, device=x.device)[None, :]
        query_position = torch.arange(past, past + seq, device=x.device)[:, None]
        score = score.masked_fill(key_position > query_position, float("-inf"))
        probability = torch.softmax(score.float(), dim=-1).to(x.dtype)
        context = torch.matmul(probability, values).transpose(1, 2).contiguous()
        return self.o_proj(context.view(batch, seq, self.heads * self.head_dim)), full_k, full_v


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor,
                rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        update, full_k, full_v = self.self_attn(
            self.input_layernorm(x), past_k, past_v, rope_cos, rope_sin)
        x = x + update
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, full_k, full_v


class Backbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(Layer(config) for _ in range(config["num_hidden_layers"]))
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def execute(self, input_ids: torch.Tensor, caches: tuple[torch.Tensor, ...],
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.model.embed_tokens(input_ids)
        outputs = []
        for layer, past_k, past_v in zip(self.model.layers, caches[::2], caches[1::2]):
            x, full_k, full_v = layer(x, past_k, past_v, rope_cos, rope_sin)
            outputs.extend((full_k, full_v))
        return (self.lm_head(self.model.norm(x)), *outputs)


def export(config: dict, phase: str, capacity: int, out: Path) -> dict:
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    seq = 1 if phase == "decode" else 64
    past = 3 if phase == "decode" else 0
    batch_example = 2
    dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    kv_heads = config["num_key_value_heads"]
    model = Decoder(config).eval().to(torch.bfloat16)
    names = [f"past_{kind}{layer}" for layer in range(config["num_hidden_layers"])
             for kind in "kv"]
    namespace = {}
    signature = ", ".join(["input_ids", *names, "rope_cos", "rope_sin"])
    expression = ", ".join(names)
    exec(f"def forward(self, {signature}):\n"
         f" return self.execute(input_ids, ({expression},), rope_cos, rope_sin)\n",
         namespace)
    object.__setattr__(model, "forward", namespace["forward"].__get__(model, Decoder))
    # Serving's persistent token state and argmax writer are both int32.
    inputs = [torch.randint(0, config["vocab_size"], (batch_example, seq), dtype=torch.int32)]
    for _ in model.model.layers:
        for _ in range(2):
            inputs.append(torch.zeros((batch_example, kv_heads, past, dim), dtype=torch.bfloat16))
    inputs.extend((torch.ones((capacity, dim), dtype=torch.bfloat16),
                   torch.zeros((capacity, dim), dtype=torch.bfloat16)))
    batch = torch.export.Dim("batch", min=1, max=64)
    dynamic = [{0: batch}]
    if phase == "decode":
        length = torch.export.Dim("past", min=1, max=capacity - 1)
        dynamic.extend({0: batch, 2: length} for _ in model.model.layers for _ in range(2))
    else:
        dynamic.extend({0: batch} for _ in model.model.layers for _ in range(2))
    dynamic.extend(({}, {}))
    program = torch.export.export(model, tuple(inputs), dynamic_shapes=tuple(dynamic), strict=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, out / "exported_program.pt2")
    manifest = {"phase": phase, "seq": seq, "capacity": capacity,
                "batch_range": [1, 64], "past_range": [1, capacity - 1] if phase == "decode" else [0, 0],
                "config": config, "torch_version": torch.__version__,
                "weight_fixture": None, "inputs": len(inputs)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {key: value for key, value in manifest.items() if key != "config"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("llama", "qwen3"), required=True)
    parser.add_argument("--phase", choices=("decode", "prefill"), required=True)
    parser.add_argument("--capacity", type=int, default=1088)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--test-small", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.test_small:
        config.update(hidden_size=128, intermediate_size=256, num_attention_heads=4,
                      num_key_value_heads=2, num_hidden_layers=1, vocab_size=256,
                      head_dim=32)
    out = args.out or Path(f"/root/r10_work/export/{args.model}_{args.phase}")
    print(json.dumps(export(config, args.phase, args.capacity, out)), flush=True)


if __name__ == "__main__":
    main()
