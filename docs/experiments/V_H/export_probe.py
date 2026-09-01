#!/usr/bin/env python3
"""Export a weight-free two-layer Llama decoder and inventory its ATen graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.export import Dim, export
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class LlamaLayer(nn.Module):
    """Llama decoder structure: pre-norm, GQA+RoPE attention, SwiGLU."""

    def __init__(self, hidden: int, intermediate: int, heads: int, kv_heads: int):
        super().__init__()
        assert hidden % heads == 0 and heads % kv_heads == 0
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = hidden // heads
        self.input_norm = RMSNorm(hidden)
        self.post_norm = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=True)

    def forward(
        self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        residual = x
        h = self.input_norm(x)
        q = self.q_proj(h).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        past = past_k.shape[2]
        positions = torch.arange(past, past + seq, device=x.device, dtype=self.inv_freq.dtype)
        phase = torch.outer(positions, self.inv_freq)
        phase = torch.cat((phase, phase), dim=-1)[None, None, :, :]
        q = q * phase.cos() + rotate_half(q) * phase.sin()
        k = k * phase.cos() + rotate_half(k) * phase.sin()

        full_k = torch.cat((past_k, k), dim=2)
        full_v = torch.cat((past_v, v), dim=2)
        repeat = self.heads // self.kv_heads
        attn_k = torch.repeat_interleave(full_k, repeat, dim=1)
        attn_v = torch.repeat_interleave(full_v, repeat, dim=1)
        score = torch.matmul(q, attn_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        query_pos = torch.arange(past, past + seq, device=x.device)[:, None]
        key_pos = torch.arange(past + seq, device=x.device)[None, :]
        score = score.masked_fill(key_pos > query_pos, float("-inf"))
        probability = torch.softmax(score.float(), dim=-1).to(q.dtype)
        context = torch.matmul(probability, attn_v)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.hidden)
        x = residual + self.o_proj(context)

        residual = x
        h = self.post_norm(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x, full_k, full_v


class TwoLayerLlama(nn.Module):
    def __init__(self, hidden: int = 512, intermediate: int = 1024):
        super().__init__()
        self.layers = nn.ModuleList(
            [LlamaLayer(hidden, intermediate, heads=4, kv_heads=2) for _ in range(2)]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        past_k0: torch.Tensor,
        past_v0: torch.Tensor,
        past_k1: torch.Tensor,
        past_v1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, next_k0, next_v0 = self.layers[0](hidden, past_k0, past_v0)
        hidden, next_k1, next_v1 = self.layers[1](hidden, past_k1, past_v1)
        return hidden, next_k0, next_v0, next_k1, next_v1


CATEGORIES = {
    "pointwise": (
        "aten.add.", "aten.mul.", "aten.neg.", "aten.pow.", "aten.rsqrt.",
        "aten.silu.", "aten.gt.", "aten.cos.", "aten.sin.", "aten.div.",
        "aten.masked_fill.",
    ),
    "reduction": ("aten.mean.", "aten.softmax."),
    "matmul": ("aten.linear.", "aten.matmul.", "aten.mm.", "aten.bmm.",
               "aten.addmm.", "aten.outer."),
    "broadcast": ("aten.expand.", "aten.repeat_interleave.", "aten.unsqueeze."),
    "concat": ("aten.cat.",),
    "slice": ("aten.slice.", "aten.select.", "aten.narrow.", "aten.chunk."),
    "transpose": ("aten.transpose.", "aten.permute.", "aten.t."),
}


def classify(name: str) -> str:
    for category, needles in CATEGORIES.items():
        if any(name.startswith(needle) for needle in needles):
            return category
    if name == "<built-in function getitem>":
        return "slice"
    if name == "<built-in function add>":
        return "pointwise"
    return "other"


def tensor_shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(str(dim) for dim in value.shape))
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(tensor_shape(item) for item in value) + "]"
    return "-"


def iter_fx_nodes(value: Any):
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_fx_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_fx_nodes(item)


def build_cg(program: torch.export.ExportedProgram) -> dict[str, Any]:
    tasks = []
    dependencies = []
    node_to_task: dict[torch.fx.Node, str] = {}
    for index, node in enumerate(program.graph.nodes):
        if node.op != "call_function":
            continue
        target = str(node.target)
        task_id = f"t{index}"
        node_to_task[node] = task_id
        tasks.append(
            {
                "id": task_id,
                "fx_name": node.name,
                "operator": target,
                "category": classify(target),
                "shape": tensor_shape(node.meta.get("val")),
                "granularity": "symbolic_g",
            }
        )
    for node, dst in node_to_task.items():
        seen = set()
        for source_node in iter_fx_nodes((node.args, node.kwargs)):
            src = node_to_task.get(source_node)
            if src is None or src in seen:
                continue
            seen.add(src)
            category = next(task["category"] for task in tasks if task["id"] == dst)
            dependencies.append(
                {
                    "src": src,
                    "dst": dst,
                    "C": f"fixed_rule:{category}",
                    "tier": 0,
                    "sync": "global",
                }
            )
    return {
        "theta": [str(symbol) for symbol in program.range_constraints],
        "range_constraints": {
            str(symbol): str(constraint)
            for symbol, constraint in program.range_constraints.items()
        },
        "task_spaces": tasks,
        "couplings": dependencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260901)
    model = TwoLayerLlama().eval()
    hidden = torch.randn(1, 4, 512)
    caches = [torch.randn(1, 2, 3, 128) for _ in range(4)]
    inputs = (hidden, *caches)
    seq = Dim("seq_len", min=1, max=8)
    past = Dim("past_len", min=1, max=8)
    dynamic_shapes = (
        {1: seq},
        {2: past},
        {2: past},
        {2: past},
        {2: past},
    )

    programs = [export(model, inputs, dynamic_shapes=dynamic_shapes, strict=True) for _ in range(3)]
    program = programs[0]
    graph_texts = [str(item.graph) for item in programs]
    stable = len({hashlib.sha256(text.encode()).hexdigest() for text in graph_texts}) == 1

    operators = sorted(
        {str(node.target) for node in program.graph.nodes if node.op == "call_function"}
    )
    rows = [(name, classify(name)) for name in operators]
    other = [name for name, category in rows if category == "other"]
    call_shapes = [
        (node.name, tensor_shape(node.meta.get("val")))
        for node in program.graph.nodes
        if node.op == "call_function" and tensor_shape(node.meta.get("val")) != "-"
    ]
    symbolic_shapes = [(name, shape) for name, shape in call_shapes if "s" in shape]
    static_shapes = [(name, shape) for name, shape in call_shapes if "s" not in shape]
    tensor_shapes = [
        (node.name, node.op, tensor_shape(node.meta.get("val")))
        for node in program.graph.nodes
        if tensor_shape(node.meta.get("val")) != "-"
    ]
    all_symbolic_shapes = [row for row in tensor_shapes if "s" in row[2]]
    all_static_shapes = [row for row in tensor_shapes if "s" not in row[2]]
    # ExportedProgram owns the executable input guards. This is private in the
    # installed release, so record both the version and exact strings rather
    # than treating it as a stable frontend API.
    guards = list(getattr(program, "_guards_code", []))
    exported_module = program.module()
    runtime_validation = []
    for runtime_seq, runtime_past in ((1, 8), (8, 1), (2, 5)):
        runtime_inputs = (
            torch.randn(1, runtime_seq, 512),
            *[torch.randn(1, 2, runtime_past, 128) for _ in range(4)],
        )
        with torch.no_grad():
            runtime_outputs = exported_module(*runtime_inputs)
        runtime_validation.append(
            {
                "seq_len": runtime_seq,
                "past_len": runtime_past,
                "output_shapes": [list(value.shape) for value in runtime_outputs],
                "status": "pass",
            }
        )
    unequal_cache_guard = {"status": "not_checked", "error": ""}
    invalid_inputs = (
        torch.randn(1, 2, 512),
        torch.randn(1, 2, 3, 128),
        torch.randn(1, 2, 4, 128),
        torch.randn(1, 2, 3, 128),
        torch.randn(1, 2, 3, 128),
    )
    try:
        with torch.no_grad():
            exported_module(*invalid_inputs)
        unequal_cache_guard = {"status": "unexpected_pass", "error": ""}
    except Exception as error:  # Runtime input guards intentionally reject it.
        unequal_cache_guard = {
            "status": "rejected",
            "error": f"{type(error).__name__}: {error}"[:1000],
        }

    with (args.out / "aten_ops.tsv").open("w", encoding="utf-8") as stream:
        stream.write("operator\tcategory\n")
        for name, category in rows:
            stream.write(f"{name}\t{category}\n")

    with (args.out / "node_shapes.tsv").open("w", encoding="utf-8") as stream:
        stream.write("node\top\ttarget\tshape\n")
        for node in program.graph.nodes:
            stream.write(
                f"{node.name}\t{node.op}\t{node.target}\t{tensor_shape(node.meta.get('val'))}\n"
            )

    (args.out / "exported_program.txt").write_text(str(program), encoding="utf-8")
    torch.export.save(program, args.out / "exported_program.pt2")
    cg = build_cg(program)
    (args.out / "cg.json").write_text(json.dumps(cg, indent=2), encoding="utf-8")
    report = {
        "torch_version": torch.__version__,
        "strict_export": True,
        "stable_three_exports": stable,
        "model": {
            "layers": 2,
            "hidden": 512,
            "intermediate": 1024,
            "heads": 4,
            "kv_heads": 2,
            "rope": True,
        },
        "kv_cache": {
            "representation": "explicit graph inputs and outputs",
            "mutation": False,
            "input_names": ["past_k0", "past_v0", "past_k1", "past_v1"],
        },
        "range_constraints": cg["range_constraints"],
        "guards": guards,
        "guards_api": "ExportedProgram._guards_code (private in this torch release)",
        "runtime_validation": runtime_validation,
        "unequal_cache_guard": unequal_cache_guard,
        "shape_propagation": {
            "symbolic_tensor_nodes": len(all_symbolic_shapes),
            "static_tensor_nodes": len(all_static_shapes),
            "symbolic_call_function_nodes": len(symbolic_shapes),
            "static_call_function_nodes": len(static_shapes),
            "symbolic": symbolic_shapes,
            "static": static_shapes,
            "static_tensor_examples": all_static_shapes,
            "specialized_dimensions": {
                "batch": 1,
                "hidden": 512,
                "intermediate": 1024,
                "query_heads": 4,
                "kv_heads": 2,
                "head_dim": 128,
            },
        },
        "operators": rows,
        "unclassified": other,
        "task_space_count": len(cg["task_spaces"]),
        "coupling_count": len(cg["couplings"]),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not stable:
        raise RuntimeError("three strict exports produced different FX graphs")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
