#!/usr/bin/env python3
"""Export the structurally distinct four-layer MHA acceptance model.

This is a test fixture producer, not compiler logic.  Model structure reaches
TileMega only through the stable export bridge and the verified CG ModuleOp.
"""
from __future__ import annotations

import argparse
import array
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export


def _load_probe(path: Path):
    spec = importlib.util.spec_from_file_location("tilemega_vh_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FourLayerMHA(nn.Module):
    """Four layers and MHA (kv_heads == heads), unlike V-H's two-layer GQA."""

    def __init__(self, layer_type):
        super().__init__()
        self.layers = nn.ModuleList(
            [layer_type(512, 1024, heads=4, kv_heads=4) for _ in range(4)]
        )

    def forward(self, hidden, past_k0, past_v0, past_k1, past_v1,
                past_k2, past_v2, past_k3, past_v3):
        caches = (past_k0, past_v0, past_k1, past_v1,
                  past_k2, past_v2, past_k3, past_v3)
        outputs = []
        for layer, (past_k, past_v) in zip(self.layers,
                                           zip(caches[0::2], caches[1::2])):
            hidden, full_k, full_v = layer(hidden, past_k, past_v)
            outputs.extend((full_k, full_v))
        return (hidden, *outputs)


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous().float()
    with path.open("wb") as stream:
        array.array("f", value.reshape(-1).tolist()).tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / "fixture"
    fixture.mkdir(exist_ok=True)

    probe = _load_probe(args.repo / "docs/experiments/V_H/export_probe.py")
    torch.manual_seed(20260902)
    model = FourLayerMHA(probe.LlamaLayer).eval()
    hidden = torch.randn(1, 4, 512)
    caches = [torch.randn(1, 4, 3, 128) for _ in range(8)]
    inputs = (hidden, *caches)
    seq, past = Dim("seq_len", min=1, max=8), Dim("past_len", min=1, max=8)
    dynamic = ({1: seq}, *({2: past} for _ in caches))
    program = export(model, inputs, dynamic_shapes=dynamic, strict=True)
    torch.export.save(program, args.out / "exported_program.pt2")

    with torch.no_grad():
        outputs = program.module()(*inputs)
    for spec, value in zip(
        (item for item in program.graph_signature.input_specs
         if item.kind.name == "USER_INPUT"), inputs
    ):
        write_f32(fixture / f"input_{spec.arg.name}.bin", value)
    for name, value in program.state_dict.items():
        write_f32(fixture / ("state_" + name.replace(".", "_") + ".bin"), value)
    for index, value in enumerate(outputs):
        write_f32(fixture / f"reference_{index}.bin", value)
    (fixture / "manifest.json").write_text(json.dumps({
        "seq": 4, "past": 3, "layers": 4, "heads": 4,
        "kv_heads": 4, "structure": "MHA without GQA",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"layers": 4, "attention": "MHA", "outputs": len(outputs)}))


if __name__ == "__main__":
    main()
