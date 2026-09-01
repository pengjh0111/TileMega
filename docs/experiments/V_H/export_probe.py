#!/usr/bin/env python3
"""V-H: export a tiny randomly initialized Llama with a dynamic sequence dim."""

import json
import sys

import torch
from torch.export import Dim, export
from transformers import LlamaConfig, LlamaForCausalLM


def op_name(node):
    target = getattr(node, "target", None)
    return str(target) if node.op == "call_function" else None


def main():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    seq = Dim("seq", min=2, max=128)
    program = export(model, (input_ids,), dynamic_shapes=({1: seq},), strict=True)

    ops = sorted({name for node in program.graph.nodes if (name := op_name(node))})
    signature = str(program.graph_signature)
    report = {
        "torch_version": torch.__version__,
        "dynamic_shapes": {"seq": [2, 128]},
        "range_constraints": [str(x) for x in program.range_constraints],
        "graph_signature": signature,
        "buffer_mutation": "BUFFER_MUTATION" in signature,
        "aten_ops": ops,
    }
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
