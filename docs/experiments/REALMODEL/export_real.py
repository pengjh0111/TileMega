#!/usr/bin/env python3
"""Export a production-shaped Llama stack through the same bridge as the
acceptance fixtures.

The two accepted models are 512-hidden toys.  This one takes the real
dimensions on the command line so the compiler is exercised at the width and
depth the paper's experiments run at.  Nothing here is compiler logic: the
model reaches TileMega only through the stable export bridge.

Weights are random.  The numerical reference is the same random weights run
through PyTorch, so a real checkpoint would add download cost and no evidence.
"""
from __future__ import annotations

import argparse
import array
import ctypes
import importlib.util
import json
import sys
import time
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


class LlamaStack(nn.Module):
    """`layers` decoder layers wired residual-to-residual, one KV cache pair
    per layer.  The signature is built with the caches flattened, matching the
    export bridge's `past_kN`/`past_vN` convention."""

    def __init__(self, layer_type, layers, hidden, intermediate, heads, kv_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [layer_type(hidden, intermediate, heads=heads, kv_heads=kv_heads)
             for _ in range(layers)]
        )

    def _run(self, hidden, caches):
        outputs = []
        for layer, (past_k, past_v) in zip(self.layers,
                                           zip(caches[0::2], caches[1::2])):
            hidden, full_k, full_v = layer(hidden, past_k, past_v)
            outputs.extend((full_k, full_v))
        return (hidden, *outputs)


def make_stack(layer_type, layers, hidden, intermediate, heads, kv_heads):
    """`torch.export` matches `dynamic_shapes` against the *declared* signature,
    so a `*caches` vararg is reported as one element and the export is refused.
    The signature is therefore generated with one named parameter per cache,
    which is also the `past_kN`/`past_vN` convention the bridge already reads."""
    names = []
    for index in range(layers):
        names.extend((f"past_k{index}", f"past_v{index}"))
    source = (
        "def forward(self, hidden, " + ", ".join(names) + "):\n"
        "    return self._run(hidden, (" + ", ".join(names) + ",))\n"
    )
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - a generated signature, no input
    stack = LlamaStack(layer_type, layers, hidden, intermediate, heads, kv_heads)
    bound = namespace["forward"].__get__(stack, type(stack))
    object.__setattr__(stack, "forward", bound)
    return stack


def write_tensor(path: Path, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    """The acceptance fixtures write through `array.array` over a Python list,
    which is fine for a 512-hidden toy and hopeless at 1.2e9 parameters: it
    materializes one Python object per element.  numpy is not installed here,
    so the bytes are copied straight out of the tensor's storage.  The layout
    is identical -- contiguous, little-endian, same dtype -- so the files are
    byte-for-byte what the old writer produced (asserted in `run.sh`)."""
    value = tensor.detach().cpu().contiguous().to(dtype)
    raw = (ctypes.c_char * (value.numel() * value.element_size())).from_address(
        value.data_ptr())
    with path.open("wb") as stream:
        stream.write(memoryview(raw))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="bf16")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seq-max", type=int, default=2048)
    parser.add_argument("--past-max", type=int, default=512)
    parser.add_argument("--fixture-seq", type=int, default=4)
    parser.add_argument("--fixture-past", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / "fixture"
    fixture.mkdir(exist_ok=True)

    probe = _load_probe(args.repo / "docs/experiments/V_H/export_probe.py")
    torch.manual_seed(20260906)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    head_dim = args.hidden // args.heads
    began = time.time()
    model = make_stack(probe.LlamaLayer, args.layers, args.hidden,
                       args.intermediate, args.heads,
                       args.kv_heads).eval().to(dtype=dtype)
    built = time.time()

    hidden = torch.randn(1, args.fixture_seq, args.hidden, dtype=dtype)
    caches = [torch.randn(1, args.kv_heads, args.fixture_past, head_dim,
                          dtype=dtype)
              for _ in range(2 * args.layers)]
    inputs = (hidden, *caches)
    seq = Dim("seq_len", min=1, max=args.seq_max)
    past = Dim("past_len", min=0, max=args.past_max)
    dynamic = ({1: seq}, *({2: past} for _ in caches))
    program = export(model, inputs, dynamic_shapes=dynamic, strict=True)
    exported = time.time()
    torch.export.save(program, args.out / "exported_program.pt2")

    with torch.no_grad():
        outputs = program.module()(*inputs)
    for spec, value in zip(
        (item for item in program.graph_signature.input_specs
         if item.kind.name == "USER_INPUT"), inputs
    ):
        write_tensor(fixture / f"input_{spec.arg.name}.bin", value, dtype)
    for name, value in program.state_dict.items():
        write_tensor(fixture / ("state_" + name.replace(".", "_") + ".bin"),
                     value, dtype)
    for index, value in enumerate(outputs):
        write_tensor(fixture / f"reference_{index}.bin", value, dtype)
    parameters = sum(item.numel() for item in program.state_dict.values())
    call_functions = sum(1 for node in program.graph.nodes
                         if node.op == "call_function")
    (fixture / "manifest.json").write_text(json.dumps({
        "seq": args.fixture_seq, "past": args.fixture_past,
        "layers": args.layers, "hidden": args.hidden,
        "intermediate": args.intermediate, "heads": args.heads,
        "kv_heads": args.kv_heads, "head_dim": head_dim,
        "dtype": str(dtype), "parameters": parameters,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "layers": args.layers, "hidden": args.hidden,
        "heads": args.heads, "kv_heads": args.kv_heads,
        "head_dim": head_dim, "parameters": parameters,
        "call_functions": call_functions, "outputs": len(outputs),
        "build_s": round(built - began, 3),
        "export_s": round(exported - built, 3),
    }))


if __name__ == "__main__":
    main()
